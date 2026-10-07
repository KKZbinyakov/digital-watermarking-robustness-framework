"""Tests for detector classification, AUC, and empirical p-value metrics."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .helpers import DETECTOR_LABEL_METRICS, evaluate


def test_classification_metrics_match_confusion_matrix_definition(detector):
    y_true = detector["y_true"]
    y_pred = detector["y_pred"]
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    expected = {
        "Accuracy": (true_positive + true_negative) / y_true.size,
        "F1": 2 * precision * recall / (precision + recall),
        "Precision": precision,
        "Recall": recall,
    }

    for name, expected_value in expected.items():
        assert evaluate(name, y_true=y_true, y_pred=y_pred) == pytest.approx(expected_value)


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("Accuracy", accuracy_score),
        ("F1", f1_score),
        ("Precision", precision_score),
        ("Recall", recall_score),
    ],
)
def test_classification_metrics_match_scikit_learn(name, reference, detector):
    assert evaluate(name, y_true=detector["y_true"], y_pred=detector["y_pred"]) == pytest.approx(
        reference(detector["y_true"], detector["y_pred"])
    )


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([0, 0, 0], [0, 0, 0], {"Accuracy": 1.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}),
        ([1, 1, 1], [0, 0, 0], {"Accuracy": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}),
        ([1, 1, 1], [1, 1, 1], {"Accuracy": 1.0, "F1": 1.0, "Precision": 1.0, "Recall": 1.0}),
    ],
)
def test_classification_metrics_define_zero_denominator_cases(y_true, y_pred, expected):
    for name, expected_value in expected.items():
        assert evaluate(name, y_true=y_true, y_pred=y_pred) == pytest.approx(expected_value)


@pytest.mark.parametrize("name", sorted(DETECTOR_LABEL_METRICS))
def test_classification_metric_rejects_shape_mismatch(name):
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate(name, y_true=np.array([1, 0, 1]), y_pred=np.array([1, 0]))


@pytest.mark.parametrize("name", sorted(DETECTOR_LABEL_METRICS))
def test_classification_metric_rejects_empty_labels(name):
    with pytest.raises(ValueError, match="empty"):
        evaluate(name, y_true=np.array([]), y_pred=np.array([]))


@pytest.mark.xfail(
    strict=True,
    reason="detector metrics document binary labels but currently accept values outside {0, 1}",
)
@pytest.mark.parametrize("name", sorted(DETECTOR_LABEL_METRICS))
def test_classification_metric_rejects_non_binary_labels(name):
    with pytest.raises(ValueError, match="0.*1|binary"):
        evaluate(name, y_true=np.array([0, 1, 2]), y_pred=np.array([0, 1, 2]))


def test_auc_matches_pairwise_probability(detector):
    y_true = detector["y_true"]
    y_scores = detector["y_scores"]
    positives = y_scores[y_true == 1]
    negatives = y_scores[y_true == 0]
    expected = float(
        np.mean(
            [
                float(positive > negative) + 0.5 * float(positive == negative)
                for positive in positives
                for negative in negatives
            ]
        )
    )

    assert evaluate("AUC", y_true=y_true, y_scores=y_scores) == pytest.approx(expected)


def test_auc_matches_scikit_learn(detector):
    expected = roc_auc_score(detector["y_true"], detector["y_scores"])

    assert evaluate("AUC", y_true=detector["y_true"], y_scores=detector["y_scores"]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([0.1, 0.2, 0.8, 0.9], 1.0),
        ([0.9, 0.8, 0.2, 0.1], 0.0),
        ([0.5, 0.5, 0.5, 0.5], 0.5),
    ],
)
def test_auc_boundary_rankings(scores, expected):
    y_true = np.array([0, 0, 1, 1])

    assert evaluate("AUC", y_true=y_true, y_scores=np.array(scores)) == pytest.approx(expected)


@pytest.mark.parametrize("label", [0, 1])
def test_auc_is_nan_when_one_class_is_missing(label):
    value = evaluate("AUC", y_true=np.full(3, label), y_scores=np.array([0.1, 0.5, 0.9]))

    assert np.isnan(value)


def test_auc_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        evaluate("AUC", y_true=np.array([0, 1]), y_scores=np.array([0.1]))


@pytest.mark.xfail(
    strict=True,
    reason="AUC documents binary labels but currently accepts values outside {0, 1}",
)
def test_auc_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0.*1|binary"):
        evaluate("AUC", y_true=np.array([0, 1, 2]), y_scores=np.array([0.1, 0.9, 0.5]))


@pytest.mark.parametrize("size", [1, 10, 1000])
def test_p_value_has_finite_sample_lower_bound(size):
    value = evaluate("P_Value", statistic=1e9, null_samples=np.zeros(size))

    assert value == pytest.approx(1 / (size + 1))
    assert value > 0


def test_p_value_counts_right_tail_including_ties():
    null_samples = np.array([-2.0, -1.0, 0.0, 0.0, 3.0])

    assert evaluate("P_Value", statistic=0.0, null_samples=null_samples) == pytest.approx(4 / 6)


def test_p_value_is_nan_for_empty_null_sample():
    assert np.isnan(evaluate("P_Value", statistic=0.0, null_samples=np.array([])))
