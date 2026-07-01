from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import TrainingLoopState
from cv_agent.trainer.yolo_trainer import TrainArtifacts


def test_training_failure_does_not_consume_successful_round(tmp_path: Path):
    engine = TrainingEngine()
    engine._config = TrainConfig(max_rounds=2, output_root=tmp_path)
    engine._run_dir = tmp_path
    engine._current_params = HyperParams()
    engine._round_num = 0
    engine._best_checkpoint = None
    engine._decision_log = []
    engine._mlflow = MagicMock()
    engine._yolo_trainer = MagicMock()
    engine._yolo_trainer.train.side_effect = RuntimeError("cuda transient")

    engine._do_train()

    assert engine._round_num == 0
    assert engine._state is TrainingLoopState.TRAIN
    assert engine._decision_log[0]["action"] == "training_crash"


def test_repeated_training_failures_eventually_stop_without_counting_rounds(tmp_path: Path):
    engine = TrainingEngine()
    engine._config = TrainConfig(max_rounds=2, max_train_failures=2, output_root=tmp_path)
    engine._run_dir = tmp_path
    engine._current_params = HyperParams()
    engine._round_num = 0
    engine._best_checkpoint = None
    engine._decision_log = []
    engine._mlflow = MagicMock()
    engine._yolo_trainer = MagicMock()
    engine._yolo_trainer.train.side_effect = RuntimeError("cuda deterministic")

    engine._do_train()
    engine._do_train()

    assert engine._round_num == 0
    assert engine._state is TrainingLoopState.DONE
    assert [entry["action"] for entry in engine._decision_log] == [
        "training_crash",
        "training_crash",
    ]


def test_successful_training_resets_consecutive_failure_count(tmp_path: Path):
    engine = TrainingEngine()
    engine._config = TrainConfig(max_rounds=2, output_root=tmp_path)
    engine._run_dir = tmp_path
    engine._current_params = HyperParams()
    engine._round_num = 0
    engine._train_failures = 1
    engine._mlflow = MagicMock()
    engine._yolo_trainer = MagicMock()
    engine._yolo_trainer.train.return_value = TrainArtifacts(
        best_pt=tmp_path / "best.pt",
        last_pt=tmp_path / "last.pt",
        args_yaml=tmp_path / "args.yaml",
        results_csv=tmp_path / "results.csv",
        run_dir=tmp_path,
    )

    engine._do_train()

    assert engine._round_num == 1
    assert engine._train_failures == 0
    assert engine._state is TrainingLoopState.EVALUATE
