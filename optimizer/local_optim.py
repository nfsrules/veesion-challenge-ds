import os
import json
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

from .base_models import BaseCameraModel

logger = logging.getLogger(__name__)


class CameraModel(BaseCameraModel):
    """
    A threshold-based model for a single camera that learns an optimal threshold
    to classify theft events using a selected optimization strategy with included persistence.
    """

    def fit(self, X, y, method="cost", nrb_steps=2000, verbose=True):
        """
        Fit camera prediction model by selecting an optimal threshold based on prediction scores and labels.

        Available methods:
            - 'cost': Minimizes true positives lost per false positive saved.
                Optional kwargs:
                    - steps (int): Number of thresholds to evaluate (default: 500)
            - 'greedy': Greedily increases threshold to reduce cost until no further improvement.
                Optional kwargs:
                    - steps (int): Number of thresholds to evaluate (default: 500)

        Time Complexity:
            O(T × n), where:
                T = number of thresholds (default 500 or as set by `steps`)
                n = number of samples

        Args:
            X (np.ndarray): Prediction scores (probabilities), shape (n_samples,).
            y (np.ndarray): Binary labels (1 = theft, 0 = non-theft), shape (n_samples,).
            method (str): Optimization strategy to use ('cost' or 'greedy').
            **kwargs: Additional arguments passed to the selected method.
        """
        self.optim_method = method
        self._fitted = False

        self.baseline_tp, self.baseline_fp = self._compute_tp_fp(X, y, threshold=self.threshold)

        if method == "cost":
            best_th = self._fit_cost(X, y, steps=nrb_steps)
        elif method == "greedy":
            best_th = self._fit_greedy(X, y, steps=nrb_steps)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        self.threshold = best_th
        self.optimal_tp, self.optimal_fp = self._compute_tp_fp(X, y, threshold=best_th)
        self.tp_lost = self.baseline_tp - self.optimal_tp
        self.fp_saved = self.baseline_fp - self.optimal_fp
        self.cost_ratio = self._compute_cost_ratio(self.tp_lost, self.fp_saved)

        self._fitted = True
        self.updated_at = datetime.now()

        if verbose:
            logger.info(f"Finished fitting camera {self.camera_id}")
            logger.debug(f" - Best threshold: {self.threshold:.4f}")
            logger.debug(f" - TP lost: {self.tp_lost}")
            logger.debug(f" - FP saved: {self.fp_saved}")
            logger.debug(f" - Cost ratio: {self.cost_ratio:.4f}")

    def _compute_cost_ratio(self, tp_lost, fp_saved):
        """Compute the cost as true positives lost per false positive saved."""
        return float("inf") if fp_saved <= 0 else tp_lost / (fp_saved + 1e-8)

    def _fit_cost(self, X, y, steps=500):
        """Grid search for optimal threshold minimizing TP lost / FP saved."""
        thresholds = np.linspace(0, 1, steps)
        base_tp, base_fp = self.baseline_tp, self.baseline_fp

        if base_fp == 0 or base_tp == 0:
            return self.threshold

        best_score = np.inf
        best_th = self.threshold

        for th in thresholds:
            tp, fp = self._compute_tp_fp(X, y, threshold=th)
            delta_tp = base_tp - tp
            delta_fp = base_fp - fp

            if delta_fp <= 0:
                continue

            cost = delta_tp / (delta_fp + 1e-8)
            if cost < best_score or (cost == best_score and th > best_th):
                best_score = cost
                best_th = th

        return best_th

    def _fit_greedy(self, X, y, steps=500):
        """Greedily increase threshold to reduce cost until no further gain."""
        thresholds = np.linspace(0, 1, steps)
        base_tp, base_fp = self.baseline_tp, self.baseline_fp

        if base_fp == 0 or base_tp == 0:
            return self.threshold

        best_th = self.threshold
        best_cost = float("inf")

        for th in thresholds:
            tp, fp = self._compute_tp_fp(X, y, threshold=th)
            delta_tp = base_tp - tp
            delta_fp = base_fp - fp

            if delta_fp <= 0:
                continue

            cost = self._compute_cost_ratio(delta_tp, delta_fp)

            if cost < best_cost:
                best_cost = cost
                best_th = th
            else:
                break

        return best_th

    def _apply_threshold(self, X, threshold):
        """Apply binary threshold to scores."""
        return (X >= threshold).astype(int)

    def _compute_tp_fp(self, X, y, threshold):
        """Compute true positives and false positives for given threshold."""
        preds = self._apply_threshold(X, threshold)
        tp = np.sum((preds == 1) & (y == 1))
        fp = np.sum((preds == 1) & (y == 0))
        return tp, fp

    def report_gain(self):
        """Return dict of TP/FP gain/loss and selected threshold."""
        if not self._fitted:
            raise RuntimeError("Model must be fit before reporting gain.")
        return {
            "baseline_tp": self.baseline_tp,
            "baseline_fp": self.baseline_fp,
            "optimal_tp": self.optimal_tp,
            "optimal_fp": self.optimal_fp,
            "tp_lost": self.tp_lost,
            "fp_saved": self.fp_saved,
            "threshold": self.threshold,
            "cost_method": self.optim_method,
            "cost_ratio": self.cost_ratio
        }

    def predict(self, X):
        """Return binary predictions using the fitted threshold."""
        if not self._fitted:
            raise RuntimeError("Model must be fit before predicting.")
        return self._apply_threshold(X, self.threshold)

    def score(self, X, y, metric="recall"):
        """
        Compute a classification metric.

        Args:
            X (np.ndarray): Scores, shape (n_samples,).
            y (np.ndarray): Labels, shape (n_samples,).
            metric (str): 'recall', 'precision', or 'f1'.

        Returns:
            float: Metric score.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before scoring.")
        preds = self.predict(X)

        if metric == "recall":
            return recall_score(y, preds, zero_division=0)
        elif metric == "precision":
            return precision_score(y, preds, zero_division=0)
        elif metric == "f1":
            return f1_score(y, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def to_dict(self):
        """Serialize model to dictionary."""
        return {
            "camera_id": self.camera_id,
            "threshold": self.threshold,
            "cost_ratio": self.cost_ratio,
            "tp_lost": self.tp_lost,
            "fp_saved": self.fp_saved,
            "optim_method": self.optim_method,
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize model from dictionary."""
        obj = cls(camera_id=data["camera_id"])
        for attr in ["threshold", "cost_ratio", "tp_lost", "fp_saved", "optim_method"]:
            setattr(obj, attr, data[attr])
        obj.updated_at = datetime.fromisoformat(data["updated_at"])
        obj._fitted = True
        return obj

    def save(self, path):
        """Save model to disk as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load model from disk."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self):
        return (f"{self.__class__.__name__}(cam={self.camera_id}, "
                f"th={self.threshold:.2f}, cost_ratio={self.cost_ratio:.4f}, "
                f"method={self.optim_method}, "
                f"updated={self.updated_at.strftime('%Y-%m-%d %H:%M:%S')})")
