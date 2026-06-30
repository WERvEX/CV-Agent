from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import DecisionAction, DecisionColor
from cv_agent.decision.strategy import StrategyPatch, StrategyPhase
from cv_agent.decision.three_state import Decision
from cv_agent.interaction.types import DecisionReview
from cv_agent.trainer.evaluator import EvaluationComparison, RoundResult


def _engine_with_strategy_state(
    tmp_path: Path,
    config: TrainConfig | None = None,
) -> TrainingEngine:
    engine = TrainingEngine()
    engine._config = config or TrainConfig(output_root=tmp_path, max_rounds=2)
    engine._run_dir = tmp_path
    engine._round_num = 1
    engine._current_params = HyperParams()
    engine._history = [
        RoundResult(
            round_num=1,
            run_dir=tmp_path,
            score=0.5,
            metrics={"recall": 0.3, "mAP50-95": 0.4},
        )
    ]
    engine._last_round_result = engine._history[0]
    engine._decision_log = []
    engine._llm_advisor = MagicMock()
    engine._optuna = MagicMock()
    return engine


def test_engine_applies_strategy_patch_before_next_proposal(tmp_path: Path):
    engine = _engine_with_strategy_state(tmp_path)
    patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="test recovery",
        search_space_patch={"lr0": (0.001, 0.002)},
    )
    engine._llm_advisor.plan_strategy.return_value = patch

    result = engine._plan_strategy(
        {"color": "red", "action": "rollback", "reason": "drop"}
    )

    assert result == patch
    engine._optuna.set_strategy_patch.assert_called_once_with(patch)


def test_engine_strategy_disabled_does_not_call_planner(tmp_path: Path):
    config = TrainConfig(output_root=tmp_path, strategy={"enabled": False})
    engine = _engine_with_strategy_state(tmp_path, config=config)

    result = engine._plan_strategy({"color": "red"})

    assert result is None
    engine._llm_advisor.plan_strategy.assert_not_called()
    engine._optuna.set_strategy_patch.assert_not_called()


def test_engine_low_confidence_strategy_keeps_active_patch(tmp_path: Path):
    config = TrainConfig(output_root=tmp_path, strategy={"min_confidence": 0.8})
    engine = _engine_with_strategy_state(tmp_path, config=config)
    active_patch = StrategyPatch(phase=StrategyPhase.EXPLORATION, reason="existing")
    engine._active_strategy_patch = active_patch
    engine._llm_advisor.plan_strategy.return_value = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="too uncertain",
        confidence=0.2,
    )

    result = engine._plan_strategy({"color": "red"})

    assert result == active_patch
    engine._optuna.set_strategy_patch.assert_not_called()


def test_engine_strategy_cadence_reuses_active_patch_without_planning(tmp_path: Path):
    config = TrainConfig(output_root=tmp_path, strategy={"planner_cadence": 2})
    engine = _engine_with_strategy_state(tmp_path, config=config)
    active_patch = StrategyPatch(phase=StrategyPhase.EXPLORATION, reason="existing")
    engine._active_strategy_patch = active_patch

    result = engine._plan_strategy({"color": "green"})

    assert result == active_patch
    engine._llm_advisor.plan_strategy.assert_not_called()
    engine._optuna.set_strategy_patch.assert_not_called()


def test_do_decide_applies_strategy_patch_before_optuna_proposal(
    tmp_path: Path,
    monkeypatch,
):
    import cv_agent.core.engine as engine_module

    engine = _engine_with_strategy_state(tmp_path)
    patch = StrategyPatch(
        phase=StrategyPhase.EXPLOITATION,
        reason="narrow next proposal",
        search_space_patch={"lr0": (0.001, 0.002)},
    )
    calls: list[str] = []
    engine._llm_advisor.plan_strategy.return_value = patch
    engine._optuna.set_strategy_patch.side_effect = lambda strategy_patch: calls.append(
        f"patch:{strategy_patch.phase.value}"
    )
    engine._optuna.propose_next.side_effect = lambda *args, **kwargs: (
        calls.append("proposal") or (HyperParams(lr0=0.0015), True)
    )
    engine._optuna.trial_count = 1
    engine._decision_engine = MagicMock()
    engine._decision_engine.decide.return_value = Decision(
        color=DecisionColor.GREEN.value,
        action=DecisionAction.ACCEPT.value,
        reason="accept",
        next_hyperparams=engine._current_params,
        metadata={"green_tier": "hard"},
    )
    engine._last_comparison = EvaluationComparison(
        current=engine._last_round_result,
        best_historical=None,
    )
    engine._interaction = MagicMock()
    engine._interaction.review_decision.return_value = DecisionReview(
        apply_recommendation=True,
        rollback_approved=False,
    )
    engine._mlflow = MagicMock()
    engine._save_session_state = MagicMock()
    monkeypatch.setattr(engine_module, "offer_mode_control", lambda **kwargs: "auto")

    engine._do_decide()

    assert calls == ["patch:exploitation", "proposal"]
    assert engine._decision_log[0]["metadata"]["strategy_patch"]["phase"] == "exploitation"
