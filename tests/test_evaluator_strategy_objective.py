from __future__ import annotations

from cv_agent.decision.strategy import ObjectiveWeights
from cv_agent.trainer.evaluator import compute_weighted_score


def test_compute_weighted_score_uses_recall_and_overfit_penalty():
    metrics = {
        "mAP50-95": 0.5,
        "mAP50": 0.7,
        "recall": 0.2,
        "precision": 0.8,
        "overfit_penalty": 0.3,
    }
    weights = ObjectiveWeights(
        map50_95=0.4,
        map50=0.1,
        recall=0.4,
        precision=0.1,
        overfit_penalty=0.2,
    )

    score = compute_weighted_score(metrics, weights)

    # ObjectiveWeights.normalized() divides by all objective fields, including
    # penalty fields, so the supplied weights sum to 1.2 before normalization.
    assert round(score, 3) == 0.308


def test_compute_weighted_score_supports_ultralytics_keys_and_default_penalties():
    metrics = {
        "metrics/mAP50-95(B)": 0.6,
        "metrics/mAP50(B)": 0.8,
        "metrics/recall(B)": 0.4,
        "metrics/precision(B)": 0.5,
    }
    weights = ObjectiveWeights(
        map50_95=1.0,
        map50=1.0,
        recall=1.0,
        precision=1.0,
        overfit_penalty=1.0,
        cost_penalty=1.0,
    )

    score = compute_weighted_score(metrics, weights)

    assert round(score, 3) == 0.383
