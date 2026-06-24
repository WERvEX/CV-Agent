from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.tracking.checkpoint_manager import CheckpointInfo


def test_resolve_initial_weights_uses_fork_on_round_one(tmp_path):
    engine = TrainingEngine()
    fork = tmp_path / "fork.pt"
    fork.write_text("weights", encoding="utf-8")
    engine._fork_weights = fork
    engine._round_num = 1
    engine._run_dir = tmp_path

    resolved = engine._resolve_initial_weights()
    assert resolved == fork


def test_run_from_checkpoint_sets_params_before_loop(tmp_path):
    weights = tmp_path / "ckpt.pt"
    weights.write_text("w", encoding="utf-8")
    info = CheckpointInfo(
        id="test:manual:x",
        run_dir=tmp_path,
        weights_path=weights,
        score=0.42,
        round=3,
        hyperparams=HyperParams(lr0=0.02, batch=8).model_dump(),
        kind="manual",
        label="test",
    )
    config = TrainConfig(output_root=tmp_path / "runs", max_rounds=1)

    engine = TrainingEngine()
    with patch.object(engine, "_begin_session") as mock_begin:
        engine.run_from_checkpoint(config, info)
    assert engine._fork_weights == weights
    assert engine._current_params.lr0 == 0.02
    assert engine._current_params.batch == 8
    mock_begin.assert_called_once()
