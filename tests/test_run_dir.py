from __future__ import annotations

import json

from cv_agent.tracking.run_dir import save_artifacts, snapshot_best_checkpoint


def test_save_artifacts_replaces_decision_log_without_duplicate_history(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    save_artifacts(run_dir=run_dir, decision_log=[{"round": 1}])
    save_artifacts(run_dir=run_dir, decision_log=[{"round": 1}, {"round": 2}])

    with open(run_dir / "decision_log.json", encoding="utf-8") as fh:
        decision_log = json.load(fh)

    assert decision_log == [{"round": 1}, {"round": 2}]


def test_snapshot_best_checkpoint_preserves_historical_best_when_best_pt_is_overwritten(tmp_path):
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("round-one-best", encoding="utf-8")

    snapshot = snapshot_best_checkpoint(run_dir=run_dir, round_num=1)

    best_pt.write_text("round-two-worse", encoding="utf-8")

    assert snapshot.read_text(encoding="utf-8") == "round-one-best"
    assert snapshot != best_pt
