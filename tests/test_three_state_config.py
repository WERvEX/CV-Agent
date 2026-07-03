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


def _history(*scores: float) -> list[RoundResult]:
    return [
        RoundResult(round_num=i + 1, run_dir=Path("r"), score=score, metrics={"mAP50": score})
        for i, score in enumerate(scores)
    ]


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


def test_dynamic_thresholds_use_percent_schedule_for_phase():
    engine = ThreeStateDecisionEngine(
        DecisionConfig(
            dynamic_thresholds=True,
            phase_schedule={
                "exploration_until_pct": 0.34,
                "exploitation_until_pct": 0.75,
            },
        )
    )

    early = engine.decide(
        _comparison(-6.0),
        current_params=HyperParams(),
        round_num=2,
        max_rounds=6,
        history=_history(0.48),
    )
    late = engine.decide(
        _comparison(-6.0),
        current_params=HyperParams(),
        round_num=5,
        max_rounds=6,
        history=_history(0.48),
    )

    assert early.metadata["decision_phase"] == "exploration"
    assert early.metadata["effective_thresholds"]["red_threshold_pct"] == -10.0
    assert early.color == "red"
    assert early.metadata["red_tier"] == "soft"
    assert early.should_rollback is False
    assert late.metadata["decision_phase"] == "convergence"
    assert late.metadata["effective_thresholds"]["red_threshold_pct"] == -3.0
    assert late.color == "red"
    assert late.should_rollback is True


def test_recent_median_prevents_red_when_current_matches_recent_performance():
    engine = ThreeStateDecisionEngine(DecisionConfig(dynamic_thresholds=True))
    comparison = _comparison(-6.0)
    comparison.current.score = 0.4512
    comparison.delta_abs = comparison.current.score - comparison.best_historical.score

    decision = engine.decide(
        comparison,
        current_params=HyperParams(),
        round_num=4,
        max_rounds=6,
        history=_history(0.48, 0.45, 0.451, 0.449),
    )

    assert decision.color == "yellow"
    assert decision.metadata["recent_median_guard"] is True
    assert decision.metadata["delta_vs_recent_median_pct"] > 0


def test_high_volatility_relaxes_red_threshold():
    engine = ThreeStateDecisionEngine(DecisionConfig(dynamic_thresholds=True))

    decision = engine.decide(
        _comparison(-4.0),
        current_params=HyperParams(),
        round_num=5,
        max_rounds=6,
        history=_history(0.48, 0.42, 0.49, 0.43),
    )

    assert decision.metadata["recent_volatility"] >= 0.01
    assert decision.metadata["effective_thresholds"]["red_threshold_pct"] == -5.0
    assert decision.color == "red"
    assert decision.metadata["red_tier"] == "soft"
    assert decision.should_rollback is False
