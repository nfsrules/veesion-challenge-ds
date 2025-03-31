import json
import logging
import time

from .base_models import BaseGlobalOptimizer

logger = logging.getLogger(__name__)


class MultiCameraOptimizer(BaseGlobalOptimizer):
    def __init__(self, df, camera_model_cls, verbose=True):
        """
        Args:
            df (pd.DataFrame): Must contain 'store', 'camera_id', 'probability', 'is_theft'.
            verbose (bool): Whether to print fitting logs.
        """
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

    def _fit_cameras(self):
        for (store, cam_id), group in self.df.groupby(['store', 'camera_id']):
            X = group['probability'].values.astype(float)
            y = group['is_theft'].values
            camera_name = f"{store}_cam{cam_id}"

            camera = self.camera_model_cls(camera_id=camera_name)

            try:
                camera.fit(X, y, method="cost")
                gain = camera.report_gain()

                if gain["fp_saved"] > 0:
                    self.cameras_info.append({
                        "camera_id": camera_name,
                        "fp_saved": gain["fp_saved"],
                        "tp_lost": gain["tp_lost"],
                        "cost_ratio": camera.cost_ratio,
                        "threshold": camera.threshold,
                    })
                else:
                    self.skipped.append(camera_name)

            except Exception as e:
                logger.warning(f"Error fitting camera {camera_name}: {e}")
                self.skipped.append(camera_name)

    def run(self, target_fp_reduction: int):
        """
        Run greedy optimization using fitted camera models.

        Args:
            target_fp_reduction (int): Desired FP reduction goal.
        """
        start_time = time.time()

        self.target_fp_reduction = target_fp_reduction

        if not self.cameras_info:
            self._fit_cameras()

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
