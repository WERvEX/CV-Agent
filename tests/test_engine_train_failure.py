from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import TrainingLoopState


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
