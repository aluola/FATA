"""A cached-feature-sized direction -> score -> metric smoke chain."""

import numpy as np

from fata.detection.metrics import detection_metrics, fit_fpr_threshold


def test_three_stage_direction_score_metric_chain():
    rng = np.random.default_rng(42)
    clean_fit = rng.normal(0, .1, (20, 3, 4))
    base_fit = clean_fit + np.array([[[1, 0, 0, 0]]] * 20)
    directions = base_fit.mean(0) - clean_fit.mean(0)  # stage x hidden
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    def score(features):
        stage_scores = np.einsum("nsh,sh->ns", features, directions)
        return stage_scores.mean(axis=1)  # aggregate only after stage scores

    negatives = rng.normal(0, .1, (100, 3, 4))
    unseen_fata = negatives[:30] + directions[None, :, :] * .8
    threshold = fit_fpr_threshold(score(negatives[:40]), .05)
    metrics = detection_metrics(score(negatives[40:]), score(unseen_fata), threshold=threshold)
    assert metrics["negative_count"] == 60
    assert metrics["positive_count"] == 30
    assert metrics["auroc"] > .95
