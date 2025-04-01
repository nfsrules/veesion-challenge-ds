from abc import ABC, abstractmethod
from datetime import datetime


class BaseCameraModel(ABC):
    """
    Abstract base class for all camera models.
    Defines the interface for training, prediction, evaluation, and persistence.
    """

    def __init__(self, camera_id, store=None):
        self.camera_id = str(camera_id)
        self.store = store
        self.threshold = 0.0
        self.optim_method = None
        self._fitted = False
        self.cost_ratio = float("inf") 
        self.created_at = datetime.now()
        self.updated_at = self.created_at

    @abstractmethod
    def fit(self, X, y, method="greedy", **kwargs):
        pass

    @abstractmethod
    def eval(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}(cam={self.camera_id}, th={self.threshold:.2f}, "
                f"method={self.optim_method}, updated={self.updated_at.strftime('%Y-%m-%d %H:%M:%S')})")


class BaseGlobalOptimizer(ABC):
    """
    Abstract base class for global optimizers.
    Defines the interface for runing an optimization with a target #FP reduction and persistence.
    """
    @abstractmethod
    def run(self, target_fp_reduction: int, outdir: str = None):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass

