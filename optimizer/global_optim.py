import json
import logging
import time
import numpy as np
from scipy.optimize import basinhopping
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

from .base_models import BaseGlobalOptimizer

logger = logging.getLogger(__name__)

AGGRESSIVE = 0.0001
STRICT     = 0.05
BALANCED   = 0.1
CONSERVATIVE = 0.5


class MultiCameraOptimizer(BaseGlobalOptimizer):
    def __init__(self, df, camera_model_cls, verbose=True):
        self.df = df
        self.camera_model_cls = camera_model_cls
        self.verbose = verbose

        self.cameras_info = []
        self.selected = []
        self.skipped = []
        self.total_fp_saved = 0
        self.total_tp_lost = 0
        self.target_fp_reduction = None

        if not self.verbose:
            logger.setLevel(logging.WARNING)

    def _fit_camera(self, store, cam_id, group, weight, method="cost", verbose=False):
        """
        Fit and evaluate a single camera, returning gain info if valid.
        """
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

    def _fit_cameras(self, method, weight):
        for (store, cam_id), group in self.df.groupby(['store', 'camera_id']):
            cam_info = self._fit_camera(store, cam_id, group, method=method, weight=weight, verbose=self.verbose)
            if cam_info:
                self.cameras_info.append(cam_info)
     
    def run(self, target_fp_reduction: float, strategy: str = "greedy"):

        logger.info(f"Running {strategy} optimization...")

        if strategy == "greedy":
            self._greedy_run(target_fp_reduction)
        else:
            self._global_run(target_fp_reduction)

    def _greedy_run(self, target_fp_reduction: int, method: str = "cost"):
        start_time = time.time()
        
        total_tp_at_0 = (self.df["is_theft"] == 1).sum()
        total_fp_at_0 = (self.df["is_theft"] == 0).sum()

        self.target_fp_reduction = target_fp_reduction * total_fp_at_0

        grouped = self.df.groupby(["store", "camera_id"])

        priority_list = []
        for (store, cam_id), group in grouped:
            baseline_fp = (group["is_theft"] == 0).sum()
            priority_list.append(((store, cam_id), baseline_fp))
        priority_list.sort(key=lambda x: -x[1])

        self.selected = []
        self.total_fp_saved = 0
        self.total_tp_lost = 0
        self.cameras_info = []
        self.skipped = []

        # Choose weight strategy
        # We've found these 
        if target_fp_reduction >= 0.20:
            weight = AGGRESSIVE
        elif target_fp_reduction >= 0.15:
            weight = STRICT
        elif target_fp_reduction >= 0.10:
            weight = BALANCED
        else:
            weight = CONSERVATIVE

        weights =  [0.0001, 0.05, 0.1, 0.5]

        for (store, cam_id), _ in priority_list:
            if self.total_fp_saved >= self.target_fp_reduction:
                break

            group = grouped.get_group((store, cam_id))
            # The fit camera function find the optimal threshold and 
            # stats for that camera using the local optmization method
            cam_info = self._fit_camera(store, 
                                        cam_id, 
                                        group, 
                                        method=method, 
                                        weight=weight, 
                                        verbose=self.verbose
                        )

            if cam_info:
                self.cameras_info.append(cam_info)
                self.selected.append(cam_info["camera_id"])
                self.total_fp_saved += cam_info["fp_saved"]
                self.total_tp_lost += cam_info["tp_lost"]
            else:
                self.skipped.append((store, cam_id))

        elapsed = time.time() - start_time

        summary = (
            "\n[Lazy Greedy Optimization Summary]\n"
            f"Target FP reduction : {self.target_fp_reduction}\n"
            f"Total FP saved      : {self.total_fp_saved} / {total_fp_at_0}\n"
            f"Total TP lost       : {self.total_tp_lost} / {total_tp_at_0}\n"
            f"Used                : {len(self.selected)} cameras\n"
            f"Skipped             : {len(self.skipped)} cameras\n"
            f"Total optimization time: {elapsed:.4f} seconds"
        )
        logger.info(summary)

    def _global_run(self, target_fp_reduction: float, verbose=True):
        """
        Optimize thresholds across all cameras using linear programming to achieve
        a target FP reduction while minimizing TP reduction.
        
        Args:
            target_fp_reduction (float): Target fraction of total FP to reduce (0 to 1).
            method (str): Optimization method (default: "cost").
        """
        start_time = time.time()
        
        # Validate required columns
        required_cols = {"store", "camera_id", "probability", "is_theft"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}. Available: {self.df.columns.tolist()}")

        # Calculate baseline totals
        total_tp_at_0 = (self.df["is_theft"] == 1).sum()
        total_fp_at_0 = (self.df["is_theft"] == 0).sum()
        self.target_fp_reduction = target_fp_reduction * total_fp_at_0

        # Group data by camera
        grouped = self.df.groupby(["store", "camera_id"])
        cameras = list(grouped.groups.keys())
        N = len(cameras)

        # Precompute threshold effects
        threshold_data = {}
        num_thresholds = 50
        for i, (store, cam_id) in enumerate(cameras):
            group = grouped.get_group((store, cam_id))
            X, y = group["probability"].values, group["is_theft"].values
            thresholds = np.linspace(0, 1, num_thresholds)
            fp_reductions, tp_reductions = [], []
            base_fp = (group["is_theft"] == 0).sum()
            base_tp = (group["is_theft"] == 1).sum()

            for t in thresholds:
                preds = (X >= t).astype(int)
                fp = np.sum((preds == 1) & (y == 0))
                tp = np.sum((preds == 1) & (y == 1))
                fp_reductions.append(base_fp - fp)
                tp_reductions.append(base_tp - tp)

            threshold_data[i] = {
                "store": store,
                "cam_id": cam_id,
                "thresholds": thresholds,
                "fp_red": fp_reductions,
                "tp_red": tp_reductions
            }

        # Set up LP problem
        prob = LpProblem("Optimal_Threshold_Optimization", LpMinimize)
        x = {i: [LpVariable(f"x_{i}_{k}", 0, 1, cat="Binary") for k in range(num_thresholds)] 
            for i in range(N)}

        # Objective: Minimize TP reduction
        prob += lpSum(threshold_data[i]["tp_red"][k] * x[i][k] 
                    for i in range(N) for k in range(num_thresholds))

        # Constraints
        for i in range(N):
            prob += lpSum(x[i][k] for k in range(num_thresholds)) == 1, f"one_threshold_{i}"
        prob += (lpSum(threshold_data[i]["fp_red"][k] * x[i][k] 
                    for i in range(N) for k in range(num_thresholds)) >= self.target_fp_reduction,
                "fp_reduction_target")

        # Solve
        prob.solve()

        # Extract results
        self.selected = []
        self.total_fp_saved = 0
        self.total_tp_lost = 0
        self.cameras_info = []
        self.skipped = []

        if LpStatus[prob.status] == "Optimal":
            for i in range(N):
                for k in range(num_thresholds):
                    if x[i][k].varValue > 0.5:
                        store = threshold_data[i]["store"]
                        cam_id = threshold_data[i]["cam_id"]
                        threshold = threshold_data[i]["thresholds"][k]
                        fp_saved = threshold_data[i]["fp_red"][k]
                        tp_lost = threshold_data[i]["tp_red"][k]
                        cam_info = {
                            "store": store,
                            "camera_id": int(cam_id),
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
                                f"Fitted camera {cam_id}. "
                                f"Optimal th: {round(threshold, 4)}. "
                                f"FP saved: {fp_saved}, TP lost: {tp_lost}"
                            )
                        break
        else:
            logger.warning(f"Optimization failed: {LpStatus[prob.status]}")
            self.skipped = [f"{store}_{cam_id}" for store, cam_id in cameras]

        elapsed = time.time() - start_time

        summary = (
            "\n[Optimal LP Optimization Summary]\n"
            f"Target FP reduction : {self.target_fp_reduction}\n"
            f"Total FP saved      : {self.total_fp_saved} / {total_fp_at_0}\n"
            f"Total TP lost       : {self.total_tp_lost} / {total_tp_at_0}\n"
            f"Used                : {len(self.selected)} cameras\n"
            f"Skipped             : {len(self.skipped)} cameras\n"
            f"Total optimization time: {elapsed:.4f} seconds"
        )
        logger.info(summary)

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
        obj = cls(df)
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
        return (f"<GreedyCameraSelector target={self.target_fp_reduction} "
                f"| FP saved={self.total_fp_saved}, TP lost={self.total_tp_lost}, "
                f"selected={len(self.selected)}, skipped={len(self.skipped)}>")
