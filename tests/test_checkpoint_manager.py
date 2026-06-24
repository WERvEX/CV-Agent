from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_agent.core.config import CheckpointConfig, HyperParams
from cv_agent.tracking.checkpoint_manager import (
    CheckpointManager,
    find_checkpoint_by_id,
    list_checkpoints,
)


def _write_weights(path: Path, content: str = "w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_top_n_records_and_prunes(tmp_path):
    run_dir = tmp_path / "exp_test"
    run_dir.mkdir()
    weights_dir = run_dir / "weights"
    cfg = CheckpointConfig(top_n=2, auto_save_top=True)

    mgr = CheckpointManager(run_dir, cfg)

    w1 = _write_weights(weights_dir / "best.pt", "r1")
    mgr.record_score(w1, score=0.30, round_num=1, hyperparams={"lr0": 0.01, "batch": 8})

    w2 = _write_weights(weights_dir / "best.pt", "r2")
    mgr.record_score(w2, score=0.50, round_num=2, hyperparams={"lr0": 0.02, "batch": 8})

    w3 = _write_weights(weights_dir / "best.pt", "r3")
    mgr.record_score(w3, score=0.40, round_num=3, hyperparams={"lr0": 0.03, "batch": 8})

    board_path = run_dir / "checkpoints" / "leaderboard.json"
    with open(board_path, encoding="utf-8") as fh:
        board = json.load(fh)
    assert len(board["entries"]) == 2
    scores = sorted([e["score"] for e in board["entries"]], reverse=True)
    assert scores == [0.50, 0.40]
    assert not any(e["round"] == 1 for e in board["entries"])


def test_manual_save_and_list(tmp_path):
    run_dir = tmp_path / "exp_manual"
    run_dir.mkdir()
    weights = _write_weights(run_dir / "weights" / "best.pt", "manual")

    mgr = CheckpointManager(run_dir, CheckpointConfig())
    mgr.save_manual(
        name="my_run",
        weights_src=weights,
        score=0.55,
        round_num=2,
        hyperparams=HyperParams(lr0=0.01, batch=16).model_dump(),
    )

    entries = list_checkpoints(tmp_path)
    manual = [e for e in entries if e.kind == "manual"]
    assert len(manual) == 1
    assert manual[0].id.endswith(":manual:my_run")
    assert find_checkpoint_by_id(tmp_path, manual[0].id) is not None


def test_manual_save_rejects_duplicate_name(tmp_path):
    run_dir = tmp_path / "exp_dup"
    run_dir.mkdir()
    weights = _write_weights(run_dir / "weights" / "best.pt")
    mgr = CheckpointManager(run_dir, CheckpointConfig())
    mgr.save_manual("dup", weights, 0.1, 1, {})
    with pytest.raises(ValueError, match="already exists"):
        mgr.save_manual("dup", weights, 0.2, 2, {})
