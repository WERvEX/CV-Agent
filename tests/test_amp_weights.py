from __future__ import annotations

from pathlib import Path

from cv_agent.trainer.amp_weights import AMP_CHECK_WEIGHT, ensure_amp_check_weights


def test_ensure_amp_check_weights_copies_from_weights_dir(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    source = weights_dir / AMP_CHECK_WEIGHT
    source.write_bytes(b"x" * 2_000_000)

    ensure_amp_check_weights(tmp_path)

    assert (tmp_path / AMP_CHECK_WEIGHT).exists()
    assert (tmp_path / AMP_CHECK_WEIGHT).stat().st_size == 2_000_000
