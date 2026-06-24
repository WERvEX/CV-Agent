"""Lightweight terminal chart helpers (sparklines, CSV history)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 24) -> str:
    """Render a Unicode block sparkline for a numeric series."""
    if not values:
        return "-"
    pts = values[-width:]
    if len(pts) == 1:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2]

    lo, hi = min(pts), max(pts)
    if hi == lo:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * len(pts)

    span = hi - lo
    last_idx = len(_SPARK_CHARS) - 1
    return "".join(
        _SPARK_CHARS[min(last_idx, int((v - lo) / span * last_idx))]
        for v in pts
    )


def parse_results_history(csv_path: Path, max_points: int = 40) -> dict[str, list[float]]:
    """Read epoch-level series from Ultralytics results.csv."""
    empty: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "map50": [],
        "map50_95": [],
        "lr": [],
    }
    if not csv_path.exists():
        return empty
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return empty
    if df.empty:
        return empty

    df.columns = [c.strip() for c in df.columns]
    if max_points > 0 and len(df) > max_points:
        df = df.iloc[-max_points:]

    def _col_series(col: str) -> list[float]:
        if col not in df.columns:
            return []
        out: list[float] = []
        for val in df[col]:
            if pd.notna(val):
                try:
                    out.append(float(val))
                except (TypeError, ValueError):
                    pass
        return out

    train_loss: list[float] = []
    val_loss: list[float] = []
    for _, row in df.iterrows():
        box = row.get("train/box_loss")
        cls = row.get("train/cls_loss")
        dfl = row.get("train/dfl_loss")
        if pd.notna(box) and pd.notna(cls) and pd.notna(dfl):
            train_loss.append(float(box) + float(cls) + float(dfl))

        vbox = row.get("val/box_loss")
        vcls = row.get("val/cls_loss")
        vdfl = row.get("val/dfl_loss")
        if pd.notna(vbox) and pd.notna(vcls) and pd.notna(vdfl):
            val_loss.append(float(vbox) + float(vcls) + float(vdfl))

    epochs = _col_series("epoch")
    if epochs:
        epochs = [float(e) for e in epochs]

    return {
        "epoch": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "map50": _col_series("metrics/mAP50(B)"),
        "map50_95": _col_series("metrics/mAP50-95(B)"),
        "lr": _col_series("lr/pg0"),
    }


def format_delta(delta_percent: float | None) -> str:
    """Format a percent delta with color markup for Rich."""
    if delta_percent is None:
        return "[dim]—[/dim]"
    if delta_percent > 0:
        return f"[bold green]{delta_percent:+.2f}%[/bold green]"
    if delta_percent < 0:
        return f"[bold red]{delta_percent:+.2f}%[/bold red]"
    return f"[dim]{delta_percent:+.2f}%[/dim]"
