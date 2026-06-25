from __future__ import annotations

from pathlib import Path

from cv_agent.core.config import DecisionConfig, HyperParams
from cv_agent.core.state_machine import DecisionAction
from cv_agent.decision.three_state import ThreeStateDecisionEngine
from cv_agent.trainer.evaluator import EvaluationComparison, RoundResult


def _comparison(delta_pct: float, overfit: bool = False, underfit: bool = False) -> EvaluationComparison:
    best_score = 0.48
    delta_abs = best_score * (delta_pct / 100.0)
    current_score = best_score + delta_abs
    current = RoundResult(round_num=2, run_dir=Path("r"), score=current_score)
    best = RoundResult(round_num=1, run_dir=Path("r"), score=best_score)
    return EvaluationComparison(
        current=current,
        best_historical=best,
        delta_percent=delta_pct,
        delta_abs=delta_abs,
        overfitting=overfit,
        underfitting=underfit,
    )


def test_custom_green_threshold():
    engine = ThreeStateDecisionEngine(DecisionConfig(green_threshold_pct=1.0))
    decision = engine.decide(_comparison(1.5), current_params=HyperParams())
    assert decision.color == "green"
    assert decision.metadata["green_tier"] == "hard"


def test_custom_red_threshold():
    engine = ThreeStateDecisionEngine(DecisionConfig(red_threshold_pct=-2.0))
    decision = engine.decide(_comparison(-3.0), current_params=HyperParams())
    assert decision.color == "red"
    assert decision.metadata["red_tier"] == "hard"


def test_yellow_between_thresholds():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(-1.0), current_params=HyperParams())
    assert decision.color == "yellow"
    assert decision.action == DecisionAction.ESCAPE_LOCAL_OPTIMUM.value


def test_marginal_green():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(2.0), current_params=HyperParams())
    assert decision.color == "green"
    assert decision.metadata["green_tier"] == "marginal"


def test_soft_red_overfit_mild_regularize():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(-4.0, overfit=True), current_params=HyperParams())
    assert decision.color == "red"
    assert decision.action == DecisionAction.MILD_REGULARIZE.value
    assert decision.should_rollback is False
    assert decision.metadata["red_tier"] == "soft"


def test_yellow_overfit_mild_regularize():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(-1.0, overfit=True), current_params=HyperParams())
    assert decision.color == "yellow"
    assert decision.action == DecisionAction.MILD_REGULARIZE.value


def test_hard_red_underfit_rolls_back():
    engine = ThreeStateDecisionEngine()
    decision = engine.decide(_comparison(-6.0, underfit=True), current_params=HyperParams())
    assert decision.color == "red"
    assert decision.action == DecisionAction.AGGRESSIVE_LR_ADJUST.value
    assert decision.should_rollback is True
