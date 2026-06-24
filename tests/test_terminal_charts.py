from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cv_agent.ui.terminal_charts import parse_results_history, sparkline


def test_sparkline_empty():
    assert sparkline([]) == "-"


def test_sparkline_constant_series():
    assert len(sparkline([1.0, 1.0, 1.0])) == 3


def test_sparkline_trending_series():
    line = sparkline([0.1, 0.2, 0.5, 0.9], width=10)
    assert len(line) == 4
    assert line[-1] > line[0]


def test_parse_results_history_reads_csv(tmp_path):
    csv_path = tmp_path / "results.csv"
    df = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train/box_loss": [1.0, 0.8],
            "train/cls_loss": [0.5, 0.4],
            "train/dfl_loss": [0.2, 0.1],
            "val/box_loss": [1.1, 1.0],
            "val/cls_loss": [0.6, 0.5],
            "val/dfl_loss": [0.3, 0.2],
            "metrics/mAP50(B)": [0.3, 0.4],
            "metrics/mAP50-95(B)": [0.2, 0.3],
            "lr/pg0": [0.01, 0.009],
        }
    )
    df.to_csv(csv_path, index=False)

    hist = parse_results_history(csv_path)

    assert hist["train_loss"][0] == 1.7
    assert hist["train_loss"][1] == pytest.approx(1.3)
    assert hist["val_loss"][0] == 2.0
    assert hist["val_loss"][1] == pytest.approx(1.7)
    assert hist["map50"] == [0.3, 0.4]
    assert hist["lr"] == [0.01, 0.009]
