import json
import logging
import time

from .base_models import BaseGlobalOptimizer

logger = logging.getLogger(__name__)


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

    def _fit_camera(self, store, cam_id, group, method="cost", verbose=False):
        """
        Fit and evaluate a single camera, returning gain info if valid.
        """
        X = group["probability"].values.astype(float)
        y = group["is_theft"].values
        camera_name = f"{store}_cam{cam_id}"
        camera = self.camera_model_cls(camera_id=camera_name)

        try:
            camera.fit(X, y, method=method, verbose=verbose)
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
            }

        except Exception as e:
            logger.warning(f"Error fitting camera {camera_name}: {e}")
            self.skipped.append(camera_name)
            return None

    def _fit_cameras(self, method):
        for (store, cam_id), group in self.df.groupby(['store', 'camera_id']):
            cam_info = self._fit_camera(store, cam_id, group, method=method, verbose=self.verbose)
            if cam_info:
                self.cameras_info.append(cam_info)

    def run(self, target_fp_reduction: int, strategy: str = "greedy"):
        assert strategy in ["greedy", "lazy"], "Strategy must be 'greedy' or 'lazy'"

        logger.info(f"Running {strategy} optimization...")

        if strategy == "lazy":
            self._lazy_run(target_fp_reduction)
        else:
            self._greedy_run(target_fp_reduction)

    def _greedy_run(self, target_fp_reduction: int, method: str = "cost"):
        assert method in ['cost', 'optuna']
        start_time = time.time()

        self.target_fp_reduction = target_fp_reduction

        if not self.cameras_info:
            self._fit_cameras(method)

        self.cameras_info.sort(key=lambda x: x["cost_ratio"])

        self.selected = []
        self.total_fp_saved = 0
        self.total_tp_lost = 0

        for cam in self.cameras_info:
            if self.total_fp_saved >= target_fp_reduction:
                break
            self.selected.append(cam["camera_id"])
            self.total_fp_saved += cam["fp_saved"]
            self.total_tp_lost += cam["tp_lost"]

        elapsed = time.time() - start_time

        summary = (
            "\nOptimization Summary\n"
            f"Target FP reduction : {self.target_fp_reduction}\n"
            f"Total FP saved      : {self.total_fp_saved}\n"
            f"Total TP lost       : {self.total_tp_lost}\n"
            f"Used                : {len(self.selected)} / {len(self.cameras_info)} cameras"
            f"Total optimization time: {elapsed:.6f} seconds"
        )
        if self.skipped:
            summary += f"\nSkipped cameras      : {self.skipped}"

        logger.info(summary)

    def _lazy_run(self, target_fp_reduction: int, method: str = "cost"):
        start_time = time.time()
        self.target_fp_reduction = target_fp_reduction

        grouped = self.df.groupby(["store", "camera_id"])

        # Let's preselect the cameras that have an high numeber of normal events
        # Because they're likelly to produce a lot of FP.
        # We need to pay more attention to those.
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

        for (store, cam_id), _ in priority_list:

            # Let's fit thresholds until we get the target reduction
            if self.total_fp_saved >= target_fp_reduction:
                break

            group = grouped.get_group((store, cam_id))
            cam_info = self._fit_camera(store, cam_id, group, method=method, verbose=False)

            if cam_info:
                self.cameras_info.append(cam_info)
                self.selected.append(cam_info["camera_id"])
                self.total_fp_saved += cam_info["fp_saved"]
                self.total_tp_lost += cam_info["tp_lost"]

        elapsed = time.time() - start_time

        summary = (
            "\n[Lazy Greedy Optimization Summary]\n"
            f"Target FP reduction : {self.target_fp_reduction}\n"
            f"Total FP saved      : {self.total_fp_saved}\n"
            f"Total TP lost       : {self.total_tp_lost}\n"
            f"Used                : {len(self.selected)} cameras\n"
            f"Skipped             : {len(self.skipped)} cameras\n"
            f"Total optimization time: {elapsed:.4f} seconds"
        )
        logger.info(summary)

    def to_dict(self):
        return {
            "target_fp_reduction": int(self.target_fp_reduction),
            "total_fp_saved": int(self.total_fp_saved),
            "total_tp_lost": int(self.total_tp_lost),
            "cameras_info": self.cameras_info
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
