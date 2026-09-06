import pytest

from fata.detection.metrics import detection_metrics, fit_fpr_threshold


def test_calibration_threshold_and_perfect_detection():
    negatives = list(range(20))
    threshold = fit_fpr_threshold(negatives, max_fpr=.05)
    result = detection_metrics(negatives, [30, 31, 32], threshold=threshold)
    assert result["fpr"] <= .05
    assert result["auroc"] == pytest.approx(1.0)
    assert result["aupr"] == pytest.approx(1.0)
    assert result["tpr_at_threshold"] == pytest.approx(1.0)


def test_ties_cannot_exceed_fpr_allowance():
    threshold = fit_fpr_threshold([0] * 19 + [1, 1], max_fpr=.05)
    assert sum(score >= threshold for score in [0] * 19 + [1, 1]) == 0
