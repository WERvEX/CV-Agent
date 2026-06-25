from __future__ import annotations

from pathlib import Path

import pytest

from cv_agent.trainer.evaluator import Evaluator, RoundResult


def test_compare_includes_delta_abs():
    evaluator = Evaluator()
    best = RoundResult(round_num=1, run_dir=Path("r"), score=0.5)
    current = RoundResult(round_num=2, run_dir=Path("r"), score=0.55)
    history = [best]

    comparison = evaluator.compare(current, history)

    assert comparison.delta_abs == pytest.approx(0.05)
    assert comparison.delta_percent == pytest.approx(10.0)
