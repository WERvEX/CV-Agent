from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import DecisionAction, DecisionColor
from cv_agent.decision.strategy import ObjectiveWeights, StrategyPatch, StrategyPhase
from cv_agent.decision.strategy_memory import StrategyMemory
from cv_agent.decision.three_state import Decision
from cv_agent.interaction.types import DecisionReview
from cv_agent.tracking.run_dir import load_strategy_log, save_session_state
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


def test_strategy_memory_baseline_recomputed_with_patch_objective_weights(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    patch = StrategyPatch(
        phase=StrategyPhase.EXPLOITATION,
        reason="prefer recall",
        objective_weights=ObjectiveWeights(
            map50_95=0.0,
            map50=0.0,
            recall=1.0,
            precision=0.0,
            overfit_penalty=0.0,
            cost_penalty=0.0,
        ),
    )
    engine._last_round_result = RoundResult(
        round_num=1,
        run_dir=tmp_path,
        score=0.9,
        metrics={"mAP50": 0.9, "recall": 0.25},
    )
    engine._history = [engine._last_round_result]
    engine._llm_advisor.plan_strategy.return_value = patch

    result = engine._plan_strategy({"color": "green"})

    assert result == patch
    assert engine._strategy_memory_baseline_score == 0.25


def test_strategy_memory_baseline_applies_overfit_penalty_when_metric_missing(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="penalize prior overfit",
        objective_weights=ObjectiveWeights(
            map50_95=0.0,
            map50=1.0,
            recall=0.0,
            precision=0.0,
            overfit_penalty=1.0,
            cost_penalty=0.0,
        ),
    )
    engine._last_round_result = RoundResult(
        round_num=1,
        run_dir=tmp_path,
        score=0.9,
        metrics={"mAP50": 0.8},
        overfitting=True,
    )
    engine._history = [engine._last_round_result]
    engine._llm_advisor.plan_strategy.return_value = patch

    result = engine._plan_strategy({"color": "red"})

    assert result == patch
    assert round(engine._strategy_memory_baseline_score, 3) == -0.1


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


def test_resume_restores_active_strategy_patch_and_applies_to_optuna(
    tmp_path: Path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "args.yaml").write_text("lr0: 0.01\n", encoding="utf-8")
    patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="resume constraints",
        search_space_patch={"lr0": (0.001, 0.002)},
    )
    save_session_state(
        run_dir,
        {
            "round_num": 1,
            "best_score": 0.5,
            "best_round": 1,
            "best_checkpoint": None,
            "history_scores": [0.5],
            "current_params": HyperParams(lr0=0.01).model_dump(),
            "active_strategy_patch": patch.model_dump(mode="json"),
            "strategy_memory_baseline_score": 0.37,
        },
    )
    engine = TrainingEngine()
    optuna = MagicMock()
    mlflow = MagicMock()

    def fake_setup(self, config):
        self._optuna = optuna
        self._mlflow = mlflow
        self._red_tracker.count = 0

    monkeypatch.setattr(TrainingEngine, "_setup_subsystems", fake_setup)
    monkeypatch.setattr(TrainingEngine, "_main_loop", lambda self: None)
    monkeypatch.setattr(TrainingEngine, "_print_summary", lambda self: None)

    engine.resume(run_dir, TrainConfig(output_root=tmp_path, max_rounds=2))

    assert engine._active_strategy_patch == patch
    assert engine._strategy_memory_baseline_score == 0.37
    optuna.set_strategy_patch.assert_called_once_with(patch)


def test_resume_ignores_malformed_active_strategy_patch(
    tmp_path: Path,
    monkeypatch,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "args.yaml").write_text("lr0: 0.01\n", encoding="utf-8")
    save_session_state(
        run_dir,
        {
            "round_num": 1,
            "best_score": 0.5,
            "best_round": 1,
            "best_checkpoint": None,
            "history_scores": [0.5],
            "current_params": HyperParams(lr0=0.01).model_dump(),
            "active_strategy_patch": {"phase": "bogus", "reason": "bad state"},
        },
    )
    engine = TrainingEngine()
    optuna = MagicMock()
    mlflow = MagicMock()
    loop_reached = False

    def fake_setup(self, config):
        self._optuna = optuna
        self._mlflow = mlflow
        self._red_tracker.count = 0

    def fake_main_loop(self):
        nonlocal loop_reached
        loop_reached = True

    monkeypatch.setattr(TrainingEngine, "_setup_subsystems", fake_setup)
    monkeypatch.setattr(TrainingEngine, "_main_loop", fake_main_loop)
    monkeypatch.setattr(TrainingEngine, "_print_summary", lambda self: None)

    engine.resume(run_dir, TrainConfig(output_root=tmp_path, max_rounds=2))

    assert loop_reached
    assert engine._active_strategy_patch is None
    optuna.set_strategy_patch.assert_not_called()


def test_save_session_state_persists_active_strategy_patch_and_strategy_log(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    patch = StrategyPatch(phase=StrategyPhase.EXPLORATION, reason="persist me")
    engine._active_strategy_patch = patch
    engine._strategy_memory_baseline_score = 0.42
    engine._strategy_log = [patch.model_dump(mode="json")]
    engine._config = TrainConfig(output_root=tmp_path, max_rounds=2)

    engine._save_session_state()

    state = (tmp_path / "session_state.json").read_text(encoding="utf-8")
    assert "active_strategy_patch" in state
    assert '"strategy_memory_baseline_score": 0.42' in state
    assert load_strategy_log(tmp_path) == [patch.model_dump(mode="json")]


def test_strategy_memory_records_active_patch_outcome_after_evaluation(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")
    patch = StrategyPatch(phase=StrategyPhase.EXPLOITATION, reason="narrow lr")
    engine._active_strategy_patch = patch
    engine._strategy_memory = StrategyMemory(max_items=3)
    engine._strategy_memory_baseline_score = 0.5
    engine._config = TrainConfig(
        output_root=tmp_path,
        max_rounds=2,
        data={"data_yaml": data_yaml},
    )
    engine._current_params = HyperParams(lr0=0.002)
    engine._last_artifacts = MagicMock()
    engine._last_artifacts.results_csv = tmp_path / "results.csv"
    engine._last_artifacts.best_pt = MagicMock()
    engine._last_artifacts.last_pt = tmp_path / "last.pt"
    engine._last_artifacts.best_pt.exists.return_value = False
    engine._evaluator = MagicMock()
    round_result = RoundResult(
        round_num=2,
        run_dir=tmp_path,
        score=0.6,
        metrics={"mAP50": 0.6},
    )
    engine._evaluator.extract_metrics.return_value = round_result
    engine._evaluator.enrich_from_validation.return_value = round_result
    engine._evaluator.compare.return_value = EvaluationComparison(
        current=round_result,
        best_historical=engine._history[-1],
    )
    engine._mlflow = MagicMock()
    engine._optuna = MagicMock()
    engine._checkpoint_manager = MagicMock()

    engine._do_evaluate()

    assert engine._strategy_memory.effective_patterns
    assert "narrow lr" in engine._strategy_memory.effective_patterns[0]
    assert engine._strategy_memory_baseline_score is None


def test_active_strategy_objective_weights_override_score_before_comparison_and_history(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")
    engine._config = TrainConfig(
        output_root=tmp_path,
        max_rounds=2,
        data={"data_yaml": data_yaml},
    )
    engine._round_num = 2
    engine._active_strategy_patch = StrategyPatch(
        phase=StrategyPhase.DATA_GAP,
        reason="prefer recall",
        objective_weights=ObjectiveWeights(
            map50_95=0.0,
            map50=0.0,
            recall=1.0,
            precision=0.0,
            overfit_penalty=0.0,
            cost_penalty=0.0,
        ),
    )
    engine._last_artifacts = MagicMock()
    engine._last_artifacts.results_csv = tmp_path / "results.csv"
    engine._last_artifacts.best_pt = MagicMock()
    engine._last_artifacts.last_pt = tmp_path / "last.pt"
    engine._last_artifacts.best_pt.exists.return_value = False
    engine._evaluator = MagicMock()
    round_result = RoundResult(
        round_num=2,
        run_dir=tmp_path,
        score=0.9,
        metrics={"mAP50": 0.9, "recall": 0.25},
    )
    engine._evaluator.extract_metrics.return_value = round_result
    engine._evaluator.enrich_from_validation.return_value = round_result
    engine._evaluator.compare.return_value = EvaluationComparison(
        current=round_result,
        best_historical=engine._history[-1],
    )
    engine._mlflow = MagicMock()
    engine._optuna = MagicMock()
    engine._checkpoint_manager = MagicMock()

    engine._do_evaluate()

    compare_current = engine._evaluator.compare.call_args.args[0]
    assert compare_current.score == 0.25
    assert engine._history[-1].score == 0.25
    assert engine._last_round_result.score == 0.25


def test_do_evaluate_injects_overfit_penalty_before_strategy_weighted_score(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")
    engine._config = TrainConfig(
        output_root=tmp_path,
        max_rounds=2,
        data={"data_yaml": data_yaml},
    )
    engine._round_num = 2
    engine._active_strategy_patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="penalize overfit",
        objective_weights=ObjectiveWeights(
            map50_95=0.0,
            map50=1.0,
            recall=0.0,
            precision=0.0,
            overfit_penalty=1.0,
            cost_penalty=0.0,
        ),
    )
    engine._last_artifacts = MagicMock()
    engine._last_artifacts.results_csv = tmp_path / "results.csv"
    engine._last_artifacts.best_pt = MagicMock()
    engine._last_artifacts.last_pt = tmp_path / "last.pt"
    engine._last_artifacts.best_pt.exists.return_value = False
    engine._evaluator = MagicMock()
    round_result = RoundResult(
        round_num=2,
        run_dir=tmp_path,
        score=0.8,
        metrics={"mAP50": 0.8},
        overfitting=True,
    )
    engine._evaluator.extract_metrics.return_value = round_result
    engine._evaluator.enrich_from_validation.return_value = round_result
    engine._evaluator.compare.return_value = EvaluationComparison(
        current=round_result,
        best_historical=engine._history[-1],
    )
    engine._mlflow = MagicMock()
    engine._optuna = MagicMock()
    engine._checkpoint_manager = MagicMock()

    engine._do_evaluate()

    assert round_result.metrics["overfit_penalty"] == 1.0
    assert round_result.metrics["cost_penalty"] == 0.0
    assert round(round_result.score, 3) == -0.1


def test_strategy_scoring_saves_audit_metrics_but_logs_only_numeric_metrics(
    tmp_path: Path,
):
    engine = _engine_with_strategy_state(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")
    engine._config = TrainConfig(
        output_root=tmp_path,
        max_rounds=2,
        data={"data_yaml": data_yaml},
    )
    engine._round_num = 2
    engine._active_strategy_patch = StrategyPatch(
        phase=StrategyPhase.DATA_GAP,
        reason="audit weights",
        objective_weights=ObjectiveWeights(
            map50_95=0.0,
            map50=0.0,
            recall=1.0,
            precision=0.0,
            overfit_penalty=0.0,
            cost_penalty=0.0,
        ),
        metadata={"source": "unit-test"},
    )
    engine._last_artifacts = MagicMock()
    engine._last_artifacts.results_csv = tmp_path / "results.csv"
    engine._last_artifacts.best_pt = MagicMock()
    engine._last_artifacts.last_pt = tmp_path / "last.pt"
    engine._last_artifacts.best_pt.exists.return_value = False
    engine._evaluator = MagicMock()
    round_result = RoundResult(
        round_num=2,
        run_dir=tmp_path,
        score=0.9,
        metrics={"mAP50": 0.9, "recall": 0.3},
    )
    engine._evaluator.extract_metrics.return_value = round_result
    engine._evaluator.enrich_from_validation.return_value = round_result
    engine._evaluator.compare.return_value = EvaluationComparison(
        current=round_result,
        best_historical=engine._history[-1],
    )
    engine._mlflow = MagicMock()
    engine._optuna = MagicMock()
    engine._checkpoint_manager = MagicMock()

    engine._do_evaluate()

    logged_metrics = engine._mlflow.log_metrics.call_args_list[0].args[0]
    assert all(isinstance(value, int | float) for value in logged_metrics.values())
    assert logged_metrics["strategy_weight_recall"] == 1.0

    import json

    saved_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["strategy_objective_weights"]["recall"] == 1.0
    assert saved_metrics["strategy_patch"]["phase"] == "data_gap"
    assert saved_metrics["strategy_patch"]["reason"] == "audit weights"


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
