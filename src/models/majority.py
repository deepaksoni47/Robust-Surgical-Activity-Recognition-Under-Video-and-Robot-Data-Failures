from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy import stats

from src.evaluation.metrics import to_numpy_labels


class MajorityClassClassifier:
    """Trivial empirical majority-class baseline classifier for surgical gesture recognition.
    
    Establishes the empirical lower-bound benchmark for each task and LOUO fold.
    Strictly trained on training trials to adhere to zero-leakage constraints.
    """

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes
        self.majority_class_: Optional[int] = None
        self.class_priors_: Optional[np.ndarray] = None
        self.class_counts_: Dict[int, int] = {}
        self.is_fitted: bool = False

    def fit(self, y_train: Union[np.ndarray, torch.Tensor, List[int]]) -> "MajorityClassClassifier":
        """Learn majority gesture class and class prior distribution from training labels.
        
        Args:
            y_train: Ground truth integer labels from training fold.
            
        Returns:
            self
        """
        labels = to_numpy_labels(y_train)
        if len(labels) == 0:
            raise ValueError("Training labels cannot be empty.")

        unique, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique.tolist(), counts.tolist()))
        self.class_counts_ = {int(k): int(v) for k, v in count_dict.items()}

        # Mode calculation
        mode_res = stats.mode(labels, keepdims=False)
        self.majority_class_ = int(mode_res.mode)

        # Empirical prior distribution
        if self.num_classes is None:
            max_class = max(self.majority_class_, int(np.max(labels)))
            self.num_classes = max_class + 1

        priors = np.zeros(self.num_classes, dtype=np.float32)
        total_samples = len(labels)
        for cls_id, count in self.class_counts_.items():
            if cls_id < self.num_classes:
                priors[cls_id] = count / total_samples

        self.class_priors_ = priors
        self.is_fitted = True

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor, int]) -> np.ndarray:
        """Predict the learned majority gesture class for all samples in X."""
        if not self.is_fitted or self.majority_class_ is None:
            raise RuntimeError("Classifier must be fitted before calling predict().")

        if isinstance(X, int):
            n_samples = X
        elif isinstance(X, (np.ndarray, torch.Tensor)):
            n_samples = len(X)
        else:
            n_samples = len(list(X))

        return np.full(n_samples, fill_value=self.majority_class_, dtype=np.int64)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor, int]) -> np.ndarray:
        """Predict class prior probabilities for all samples in X."""
        if not self.is_fitted or self.class_priors_ is None:
            raise RuntimeError("Classifier must be fitted before calling predict_proba().")

        if isinstance(X, int):
            n_samples = X
        elif isinstance(X, (np.ndarray, torch.Tensor)):
            n_samples = len(X)
        else:
            n_samples = len(list(X))

        return np.tile(self.class_priors_, (n_samples, 1))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted classifier parameters."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted classifier.")
        return {
            "num_classes": self.num_classes,
            "majority_class": self.majority_class_,
            "class_counts": self.class_counts_,
            "class_priors": self.class_priors_.tolist() if self.class_priors_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MajorityClassClassifier":
        """Reconstruct classifier from serialized state dictionary."""
        clf = cls(num_classes=data["num_classes"])
        clf.majority_class_ = data["majority_class"]
        clf.class_counts_ = {int(k): v for k, v in data["class_counts"].items()}
        clf.class_priors_ = np.array(data["class_priors"], dtype=np.float32)
        clf.is_fitted = True
        return clf
