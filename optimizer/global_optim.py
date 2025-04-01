import json
import logging
import time
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

from .base_models import BaseGlobalOptimizer

logger = logging.getLogger(__name__)


class MultiCameraOptimizer(BaseGlobalOptimizer):
    def __init__(self, df, camera_model_cls, verbose=True):
        self.df = df
        self.camera_model_cls = camera_model_cls
        self.verbose = verbose

        self._reset_state()
        self.target_fp_reduction = None
        self.grouped = df.groupby(["store", "camera_id"])

        if not self.verbose:
            logger.setLevel(logging.WARNING)

    def _reset_state(self):
        self.cameras_info = []
        self.selected = []
        self.skipped = []
        self.total_fp_saved = 0
        self.total_tp_lost = 0

    def _fit_camera(self, store, cam_id, group, weight, method="cost", verbose=False):
        X = group["probability"].values.astype(float)
        y = group["is_theft"].values
        camera_name = f"{store}_cam{cam_id}"
        camera = self.camera_model_cls(camera_id=camera_name)

        try:
            camera.fit(X, y, method=method, weight=weight, verbose=verbose)
            gain = camera.report_gain()

            if gain["fp_saved"] <= 0:
                self.skipped.append(camera_name)
                return None

            return {
                "camera_id": camera_name,
                "fp_saved": gain["fp_saved"],
                "tp_lost": gain["tp_lost"],
                "cost_ratio": camera.cost_ratio,
                "threshold": camera.threshold,
                "X": X,
                "y": y,
            }

        except Exception as e:
            logger.warning(f"Error fitting camera {camera_name}: {e}")
            self.skipped.append(camera_name)
            return None

    def _evaluate_thresholds(self, X, y, num_thresholds=50):
        thresholds = np.linspace(0, 1, num_thresholds)
        base_fp = np.sum(y == 0)
        base_tp = np.sum(y == 1)
        fp_reductions = []
        tp_reductions = []

        for t in thresholds:
            preds = (X >= t).astype(int)
            fp = np.sum((preds == 1) & (y == 0))
            tp = np.sum((preds == 1) & (y == 1))
            fp_reductions.append(base_fp - fp)
            tp_reductions.append(base_tp - tp)

        return thresholds, fp_reductions, tp_reductions

    def _select_best_weight(self, store, cam_id, group, weights, method):
        best_info = None
        best_fp_saved = -1
        best_tp_lost = float("inf")

        for weight in weights:
            cam_info = self._fit_camera(store, cam_id, group, weight, method=method, verbose=False)
            if not cam_info:
                continue

            if (cam_info["fp_saved"] > best_fp_saved) or \
               (cam_info["fp_saved"] == best_fp_saved and cam_info["tp_lost"] < best_tp_lost):
                best_info = cam_info
                best_fp_saved = cam_info["fp_saved"]
                best_tp_lost = cam_info["tp_lost"]

        return best_info

    def _log_summary(self, title, total_fp, total_tp, elapsed):
        logger.info(
            f"\n[{title}]\n"
            f"Target FP reduction : {self.target_fp_reduction:.0f}\n"
            f"Total FP saved      : {self.total_fp_saved} / {total_fp}\n"
            f"Total TP lost       : {self.total_tp_lost} / {total_tp}\n"
            f"Used                : {len(self.selected)} cameras\n"
            f"Skipped             : {len(self.skipped)} cameras\n"
            f"Total optimization time: {elapsed:.4f} seconds"
        )

    def run(self, target_fp_reduction: float, strategy: str = "greedy"):
        logger.info(f"Running {strategy} optimization...")

        strategies = {
            "greedy": self._greedy_run,
            "global": self._global_run
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}")

        strategies[strategy](target_fp_reduction)

    def _greedy_run(self, target_fp_reduction: float, method: str = "cost"):
        start_time = time.time()
        self._reset_state()

        total_tp_at_0 = (self.df["is_theft"] == 1).sum()
        total_fp_at_0 = (self.df["is_theft"] == 0).sum()
        self.target_fp_reduction = target_fp_reduction * total_fp_at_0

        weights = [0.0001, 0.05, 0.1, 0.5]
        priority_list = sorted(
            self.grouped.groups.keys(),
            key=lambda key: -((self.grouped.get_group(key)["is_theft"] == 0).sum())
        )

        for (store, cam_id) in priority_list:
            if self.total_fp_saved >= self.target_fp_reduction:
                break

            group = self.grouped.get_group((store, cam_id))
            best_info = self._select_best_weight(store, cam_id, group, weights, method)

            if best_info:
                self.cameras_info.append(best_info)
                self.selected.append(best_info["camera_id"])
                self.total_fp_saved += best_info["fp_saved"]
                self.total_tp_lost += best_info["tp_lost"]
            else:
                self.skipped.append((store, cam_id))

        self._log_summary("Lazy Greedy Optimization Summary", total_fp_at_0, total_tp_at_0, time.time() - start_time)

    def _unpack_lp_solution(self, x, threshold_data, num_thresholds, verbose):
        for i in range(len(threshold_data)):
            for k in range(num_thresholds):
                if x[i][k].varValue > 0.5:
                    store = threshold_data[i]["store"]
                    cam_id = threshold_data[i]["cam_id"]
                    threshold = threshold_data[i]["thresholds"][k]
                    fp_saved = threshold_data[i]["fp_red"][k]
                    tp_lost = threshold_data[i]["tp_red"][k]

                    cam_info = {
                        "store": store,
                        "camera_id": cam_id,
                        "threshold": float(threshold),
                        "fp_saved": int(fp_saved),
                        "tp_lost": int(tp_lost)
                    }

                    self.cameras_info.append(cam_info)
                    self.selected.append(cam_id)
                    self.total_fp_saved += fp_saved
                    self.total_tp_lost += tp_lost

                    if verbose:
                        logger.info(
                            f"Fitted camera {cam_id}. Optimal th: {round(threshold, 4)}. "
                            f"FP saved: {fp_saved}, TP lost: {tp_lost}"
                        )
                    break

    def _global_run(self, target_fp_reduction: float, verbose=True):
        start_time = time.time()
        self._reset_state()

        total_tp_at_0 = (self.df["is_theft"] == 1).sum()
        total_fp_at_0 = (self.df["is_theft"] == 0).sum()
        self.target_fp_reduction = target_fp_reduction * total_fp_at_0

        cameras = list(self.grouped.groups.keys())
        N = len(cameras)
        threshold_data = {}
        num_thresholds = 50

        for i, (store, cam_id) in enumerate(cameras):
            group = self.grouped.get_group((store, cam_id))
            X, y = group["probability"].values, group["is_theft"].values
            thresholds, fp_red, tp_red = self._evaluate_thresholds(X, y, num_thresholds)
            threshold_data[i] = {
                "store": store,
                "cam_id": cam_id,
                "thresholds": thresholds,
                "fp_red": fp_red,
                "tp_red": tp_red
            }

        prob = LpProblem("Optimal_Threshold_Optimization", LpMinimize)
        x = {i: [LpVariable(f"x_{i}_{k}", 0, 1, cat="Binary") for k in range(num_thresholds)] for i in range(N)}

        prob += lpSum(threshold_data[i]["tp_red"][k] * x[i][k] for i in range(N) for k in range(num_thresholds))

        for i in range(N):
            prob += lpSum(x[i][k] for k in range(num_thresholds)) == 1, f"one_threshold_{i}"

        prob += lpSum(threshold_data[i]["fp_red"][k] * x[i][k] for i in range(N) for k in range(num_thresholds)) >= self.target_fp_reduction

        prob.solve()

        if LpStatus[prob.status] == "Optimal":
            self._unpack_lp_solution(x, threshold_data, num_thresholds, verbose)
        else:
            logger.warning(f"Optimization failed: {LpStatus[prob.status]}")
            self.skipped = [f"{store}_{cam_id}" for store, cam_id in cameras]

        self._log_summary("Optimal LP Optimization Summary", total_fp_at_0, total_tp_at_0, time.time() - start_time)

    def to_dict(self):
        return {
            "target_fp_reduction": int(self.target_fp_reduction) if self.target_fp_reduction is not None else None,
            "total_fp_saved": int(self.total_fp_saved),
            "total_tp_lost": int(self.total_tp_lost),
            "cameras_info": [
                {k: v for k, v in cam.items() if k not in {"X", "y", "camera"}}
                for cam in self.cameras_info
            ]
        }

    @classmethod
    def from_dict(cls, data, df):
        obj = cls(df, camera_model_cls=None)
        obj.__dict__.update(data)
        return obj

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, df):
        with open(path, "r") as f:
            return cls.from_dict(json.load(f), df)

    def __repr__(self):
        return (
            f"<MultiCameraOptimizer target={self.target_fp_reduction} "
            f"| FP saved={self.total_fp_saved}, TP lost={self.total_tp_lost}, "
            f"selected={len(self.selected)}, skipped={len(self.skipped)}>"
        )
