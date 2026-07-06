from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import TrainingLoopState
from cv_agent.trainer.early_stop import EarlyStopDecision, resolve_early_stop_metric
from cv_agent.trainer.evaluator import EvaluationComparison, RoundResult


def test_resolve_early_stop_metric_supports_score_global_and_class_name(tmp_path: Path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [person, bicycle, car]\n", encoding="utf-8")
    result = RoundResult(
        round_num=1,
        run_dir=tmp_path,
        score=0.72,
        metrics={
            "mAP50": 0.70,
            "mAP50_95": 0.45,
            "mAP50_class_2": 0.81,
        },
    )

    assert resolve_early_stop_metric("score", result, data_yaml) == ("score", 0.72)
    assert resolve_early_stop_metric("mAP50", result, data_yaml) == ("mAP50", 0.70)
    assert resolve_early_stop_metric("mAP50_class:car", result, data_yaml) == (
        "mAP50_class_2",
        0.81,
    )


def test_do_evaluate_stops_when_early_stop_target_is_reached(tmp_path: Path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [car]\n", encoding="utf-8")
    engine = TrainingEngine()
    engine._config = TrainConfig(
        output_root=tmp_path,
        max_rounds=3,
        data={"data_yaml": data_yaml},
        early_stop={"enabled": True, "metric": "mAP50", "target": 0.75},
    )
    engine._run_dir = tmp_path
    engine._round_num = 1
    engine._current_params = HyperParams()
    engine._history = []
    engine._decision_log = []
    engine._last_artifacts = MagicMock()
    engine._last_artifacts.results_csv = tmp_path / "results.csv"
    engine._last_artifacts.best_pt = tmp_path / "weights" / "best.pt"
    engine._last_artifacts.last_pt = tmp_path / "weights" / "last.pt"
    engine._evaluator = MagicMock()
    round_result = RoundResult(
        round_num=1,
        run_dir=tmp_path,
        score=0.76,
        metrics={"mAP50": 0.76},
    )
    engine._evaluator.extract_metrics.return_value = round_result
    engine._evaluator.enrich_from_validation.return_value = round_result
    engine._evaluator.compare.return_value = EvaluationComparison(current=round_result, best_historical=None)
    engine._mlflow = MagicMock()
    engine._optuna = MagicMock()
    engine._checkpoint_manager = MagicMock()

    engine._do_evaluate()

    assert engine._state is TrainingLoopState.DONE
    assert isinstance(engine._early_stop_decision, EarlyStopDecision)
    assert engine._early_stop_decision.reached is True
    assert engine._early_stop_decision.metric_value == 0.76
    engine._optuna.report_result.assert_called_once_with(0.76, engine._current_params)


def test_export_final_best_model_writes_stable_final_artifacts(tmp_path: Path):
    best_snapshot = tmp_path / "best_snapshots" / "round_001_best.pt"
    best_snapshot.parent.mkdir()
    best_snapshot.write_bytes(b"weights")
    engine = TrainingEngine()
    engine._run_dir = tmp_path
    engine._best_checkpoint = best_snapshot
    engine._best_round = 1
    engine._best_score = 0.83
    engine._round_num = 2
    engine._early_stop_decision = EarlyStopDecision(
        reached=True,
        metric="mAP50",
        metric_value=0.83,
        target=0.80,
    )

    exported = engine._export_final_best_model()

    assert exported == tmp_path / "final" / "best.pt"
    assert exported.read_bytes() == b"weights"
    summary = json.loads((tmp_path / "final" / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_round"] == 1
    assert summary["best_score"] == 0.83
    assert summary["early_stop"]["reached"] is True
