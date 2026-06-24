from __future__ import annotations

import json

from cv_agent.core.config import HyperParams
from cv_agent.tracking.run_dir import (
    hyperparams_from_args_yaml,
    load_session_state,
    restore_session_state,
    save_session_state,
    snapshot_best_checkpoint,
)


def test_save_and_load_session_state(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    state = {
        "round_num": 2,
        "best_score": 0.42,
        "best_round": 1,
        "best_checkpoint": "best_snapshots/round_001_best.pt",
        "history_scores": [0.3, 0.42],
        "current_params": HyperParams(lr0=0.01, batch=8).model_dump(),
        "interaction_mode": "auto",
    }
    save_session_state(run_dir, state)

    loaded = load_session_state(run_dir)
    assert loaded == state


def test_restore_session_state_from_legacy_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_text("best", encoding="utf-8")

    (run_dir / "decision_log.json").write_text(
        json.dumps([{"round": 1, "color": "green"}]),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"score": 0.5, "best_score": 0.5}),
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text("lr0: 0.02\nbatch: 16\n", encoding="utf-8")

    snapshot_best_checkpoint(run_dir=run_dir, round_num=1)

    state = restore_session_state(run_dir)
    assert state is not None
    assert state["round_num"] == 1
    assert state["best_score"] == 0.5
    assert state["current_params"]["lr0"] == 0.02
    assert state["current_params"]["batch"] == 16


def test_hyperparams_from_args_yaml_filters_unknown_keys(tmp_path):
    args_path = tmp_path / "args.yaml"
    args_path.write_text(
        "lr0: 0.03\nbatch: 32\nunknown_key: 99\n",
        encoding="utf-8",
    )

    data = hyperparams_from_args_yaml(args_path)
    assert data["lr0"] == 0.03
    assert data["batch"] == 32
    assert "unknown_key" not in data
