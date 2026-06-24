from __future__ import annotations

from cv_agent.core.config import DecisionConfig, HyperParams
from cv_agent.decision.three_state import ThreeStateDecisionEngine
from cv_agent.trainer.evaluator import EvaluationComparison, RoundResult
from pathlib import Path


def _comparison(delta: float, overfit: bool = False, underfit: bool = False) -> EvaluationComparison:
    current = RoundResult(round_num=2, run_dir=Path("r"), score=0.5)
    best = RoundResult(round_num=1, run_dir=Path("r"), score=0.48)
    return EvaluationComparison(
        current=current,
        best_historical=best,
        delta_percent=delta,
        overfitting=overfit,
        underfitting=underfit,
    )


def test_custom_green_threshold():
    engine = ThreeStateDecisionEngine(DecisionConfig(green_threshold_pct=1.0))
    decision = engine.decide(_comparison(1.5), red_count=0, current_params=HyperParams())
    assert decision.color == "green"


def test_custom_red_threshold():
    engine = ThreeStateDecisionEngine(DecisionConfig(red_threshold_pct=-2.0))
    decision = engine.decide(_comparison(-3.0), red_count=0, current_params=HyperParams())
    assert decision.color == "red"


def test_yellow_between_thresholds():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(-1.0), red_count=0, current_params=HyperParams())
    assert decision.color == "yellow"
    assert decision.action == "escape_local_optimum"
