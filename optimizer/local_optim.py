import os
import json
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import optuna

from .base_models import BaseCameraModel

logger = logging.getLogger(__name__)


class CameraModel(BaseCameraModel):
    """
    A threshold-based model for a single camera that learns an optimal threshold
    to classify theft events using a selected optimization strategy with included persistence.
    """

    def fit(self, X, y, method="cost", verbose=True):
        """
        Fit camera prediction model by selecting an optimal threshold based on prediction scores and labels.

        Available methods:
            - 'cost': Minimizes true positives lost per false positive saved.
                Optional kwargs:
                    - steps (int): Number of thresholds to evaluate (default: 500)
            - 'optuna': Greedily increases threshold to reduce cost until no further improvement.
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
            best_th = self._fit_cost(X, y)
        elif method == "optuna":
            best_th = self._fit_optuna(X, y, n_trials=50)
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
            logger.info(
                f"Fitted camera {self.camera_id[:-5]}. "
                f"Optimal th: {round(self.threshold, 4)}. "
                f"FP saved: {self.fp_saved}, TP lost: {self.tp_lost}"
            )

    def _compute_cost_ratio(self, tp_lost, fp_saved):
        """Compute the cost as true positives lost per false positive saved."""
        return float("inf") if fp_saved <= 0 else tp_lost / (fp_saved + 1e-8)

    def _fit_cost(self, X, y):
        """Grid search for optimal threshold minimizing TP lost / FP saved."""
        base_tp, base_fp = self._compute_tp_fp(X, y, threshold=self.threshold)

        if base_fp == 0 or base_tp == 0:
            return self.threshold

        precision, recall, thresholds = precision_recall_curve(y, X)
        total_pos = np.sum(y)

        best_score = np.inf
        best_th = self.threshold

        for i in range(len(thresholds)):
            tp = recall[i] * total_pos

            # invert precision = TP / (TP + FP) to get FP
            fp = tp * (1 - precision[i]) / (precision[i] + 1e-8)

            delta_tp = base_tp - tp
            delta_fp = base_fp - fp

            if delta_fp <= 0:
                continue

            cost = delta_tp / (delta_fp + 1e-8)
            if cost < best_score or (cost == best_score and thresholds[i] > best_th):
                best_score = cost
                best_th = thresholds[i]

        return best_th

    def _fit_optuna(self, X, y, n_trials=100):
        base_tp, base_fp = self.baseline_tp, self.baseline_fp

        def objective(trial):
            threshold = trial.suggest_float("threshold", 0.0, 1.0)
            tp, fp = self._compute_tp_fp(X, y, threshold)
            delta_tp = base_tp - tp
            delta_fp = base_fp - fp
            if delta_fp <= 0:
                return float("inf")
            return delta_tp / (delta_fp + 1e-8)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params["threshold"]

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
            "baseline_tp": int(self.baseline_tp),
            "baseline_fp": int(self.baseline_fp),
            "optimal_tp": int(self.optimal_tp),
            "optimal_fp": int(self.optimal_fp),
            "tp_lost": int(self.tp_lost),
            "fp_saved": int(self.fp_saved),
            "threshold": float(self.threshold),
            "cost_method": self.optim_method,
            "cost_ratio": float(self.cost_ratio)
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
