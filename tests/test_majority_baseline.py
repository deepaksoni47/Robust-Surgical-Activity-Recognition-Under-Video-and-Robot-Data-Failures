import numpy as np
import pytest
import torch

from src.data.dataset import get_fold_datasets
from src.evaluation.metrics import compute_classification_metrics
from src.models.majority import MajorityClassClassifier


def test_majority_class_fitting() -> None:
    """Test mode discovery and constant prediction."""
    y_train = [1, 1, 1, 2, 2, 3]
    clf = MajorityClassClassifier(num_classes=4)
    clf.fit(y_train)

    assert clf.is_fitted
    assert clf.majority_class_ == 1

    # Predict on 5 arbitrary inputs
    preds = clf.predict(5)
    assert len(preds) == 5
    assert np.all(preds == 1)


def test_class_priors_distribution() -> None:
    """Test that predicted class probabilities reflect training frequencies and sum to 1."""
    y_train = [0, 1, 1, 1]
    clf = MajorityClassClassifier(num_classes=2)
    clf.fit(y_train)

    assert pytest.approx(clf.class_priors_[0]) == 0.25
    assert pytest.approx(clf.class_priors_[1]) == 0.75
    assert pytest.approx(float(np.sum(clf.class_priors_))) == 1.0

    probas = clf.predict_proba(3)
    assert probas.shape == (3, 2)
    assert np.allclose(probas.sum(axis=-1), 1.0)


def test_unfitted_raises_runtime_error() -> None:
    """Verify that calling predict before fit raises RuntimeError."""
    clf = MajorityClassClassifier()
    with pytest.raises(RuntimeError, match="must be fitted"):
        clf.predict(10)


def test_serialization_roundtrip() -> None:
    """Test serialization and deserialization of fitted classifier."""
    y = [2, 2, 2, 1, 0]
    clf1 = MajorityClassClassifier(num_classes=3).fit(y)
    data = clf1.to_dict()

    clf2 = MajorityClassClassifier.from_dict(data)
    assert clf2.is_fitted
    assert clf2.majority_class_ == 2
    assert np.allclose(clf1.class_priors_, clf2.class_priors_)


def test_majority_baseline_suturing_fold1() -> None:
    """Test running majority baseline on actual Suturing fold 1 data."""
    fold_data = get_fold_datasets(
        task="Suturing",
        fold_num=1,
        window_size=16,
        train_stride=16,
        eval_stride=16,
        dataset_dir="dataset",
    )

    train_targets = [sample["target"].item() for sample in fold_data["train"]]
    test_targets = [sample["target"].item() for sample in fold_data["test"]]

    clf = MajorityClassClassifier(num_classes=len(fold_data["vocab"]))
    clf.fit(train_targets)

    assert clf.is_fitted
    preds = clf.predict(len(test_targets))

    metrics = compute_classification_metrics(
        y_true=test_targets,
        y_pred=preds,
        num_classes=len(fold_data["vocab"]),
        class_names=fold_data["inv_vocab"],
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["total_samples"] == len(test_targets)
