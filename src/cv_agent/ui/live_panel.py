"""Rich Live progress panel for cv_agent training sessions.

Renders a single self-refreshing terminal panel showing:
- Header: run dir, model variant, interaction mode, current pipeline stage.
- Body (left): current round, epoch, train/val loss, mAP/precision/recall, lr —
  polled from ``<run_dir>/results.csv`` (Ultralytics appends one row per epoch).
- Body (right): completed rounds, best score so far, Green/Yellow/Red decision
  counts, key current hyperparameters.
- Footer: a scrolling log of recent ``log_*`` / ``print_section`` lines.

The engine's ``_do_train`` blocks inside ``model.train()`` for the whole round,
so per-epoch updates are driven by a background daemon thread that re-reads the
CSV every ~1.5 s. The thread only reads engine state; it never mutates it.

Conflict avoidance: while a Live panel is active, :mod:`cv_agent.ui.console`'s
``log_*`` / ``print_section`` helpers divert their output into ``_log_buffer``
instead of printing directly, so they never corrupt the Live region. This is
gated by the module-level :data:`_live_active` flag toggled here.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cv_agent.ui.console import console
from cv_agent.ui.terminal_charts import parse_results_history, sparkline

if TYPE_CHECKING:
    from cv_agent.core.engine import TrainingEngine

# ---------------------------------------------------------------------------
# Log buffer — shared with cv_agent.ui.console so log_* lines route here while
# a Live panel is active.
# ---------------------------------------------------------------------------

#: Maximum log lines retained in the footer.
LOG_BUFFER_SIZE = 8

#: Module-level flag: True while a LivePanel is active. console.log_* consults
#: this and :data:`_log_buffer` to decide whether to print or buffer.
_live_active: bool = False

#: Ring buffer of recent log/section lines (renderable markup strings).
_log_buffer: deque[str] = deque(maxlen=LOG_BUFFER_SIZE)


def push_log_line(line: str) -> None:
    """Append a formatted log/section line to the footer buffer."""
    _log_buffer.append(line)


def is_live_active() -> bool:
    """Return True if a LivePanel is currently rendering."""
    return _live_active


# ---------------------------------------------------------------------------
# results.csv parsing — pure function, unit-testable without a Live region.
# ---------------------------------------------------------------------------

def parse_results_row(csv_path: Path) -> dict[str, Any]:
    """Read the latest row from a Ultralytics results.csv.

    Tolerates a missing or empty file (returns an empty dict). Column names
    follow Ultralytics' convention; spaces are stripped (as in evaluator.py).

    Returns:
        Dict with keys: epoch, train_loss, val_loss, map50, map50_95,
        precision, recall, lr. Missing values are omitted.
    """
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    if df.empty:
        return {}

    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[-1]

    def _get(col: str) -> float | None:
        if col in df.columns and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                return None
        return None

    out: dict[str, Any] = {}
    epoch = _get("epoch")
    if epoch is not None:
        out["epoch"] = int(epoch)

    box = _get("train/box_loss")
    cls = _get("train/cls_loss")
    dfl = _get("train/dfl_loss")
    if None not in (box, cls, dfl):
        out["train_loss"] = box + cls + dfl

    vbox = _get("val/box_loss")
    vcls = _get("val/cls_loss")
    vdfl = _get("val/dfl_loss")
    if None not in (vbox, vcls, vdfl):
        out["val_loss"] = vbox + vcls + vdfl

    out["map50"] = _get("metrics/mAP50(B)")
    out["map50_95"] = _get("metrics/mAP50-95(B)")
    out["precision"] = _get("metrics/precision(B)")
    out["recall"] = _get("metrics/recall(B)")
    out["lr"] = _get("lr/pg0")

    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Live panel
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 1.5  # seconds between results.csv re-reads


class LivePanel:
    """Context manager that renders a live training dashboard.

    Usage::

        with LivePanel(engine):
            engine._main_loop()

    The panel self-refreshes; engine code does not need to call update(). A
    background thread polls ``engine._run_dir / "results.csv"`` for per-epoch
    progress while ``_do_train`` blocks inside ``model.train()``.
    """

    def __init__(self, engine: "TrainingEngine") -> None:
        self._engine = engine
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_csv: dict[str, Any] = {}
        self._latest_history: dict[str, list[float]] = {}
        self._live: Live | None = None
        self._stage: str = "INIT"

    # -- public API -------------------------------------------------------

    def set_stage(self, stage: str) -> None:
        """Update the pipeline stage shown in the header (TRAIN/EVAL/DECIDE...)."""
        self._stage = stage

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "LivePanel":
        global _live_active
        # Degrade gracefully when not attached to a real terminal: skip the
        # Live region entirely so log_* print normally (piped output, CI, etc.)
        if not console.is_terminal:
            return self
        _live_active = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="cv_agent-live-panel", daemon=True
        )
        self._thread.start()
        self._live = Live(
            self._render(), console=console, refresh_per_second=2,
            screen=False, transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _live_active
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
        _live_active = False
        _log_buffer.clear()

    # -- background poller ------------------------------------------------

    def _poll_loop(self) -> None:
        run_dir = getattr(self._engine, "_run_dir", None)
        csv_path = (run_dir / "results.csv") if run_dir else None
        while not self._stop.is_set():
            if csv_path is not None:
                self._latest_csv = parse_results_row(csv_path)
                self._latest_history = parse_results_history(csv_path)
            self._stop.wait(_POLL_INTERVAL)

    # -- rendering --------------------------------------------------------

    def _render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), name="header", size=4),
            Layout(self._render_body(), name="body"),
            Layout(self._render_footer(), name="footer", size=LOG_BUFFER_SIZE + 2),
        )
        return Panel(layout, border_style="cyan", title="[bold cyan]cv_agent[/bold cyan]")

    def _render_header(self) -> Text:
        eng = self._engine
        cfg = eng._config
        run_dir = getattr(eng, "_run_dir", None)
        lines = [
            f"[bold]cv_agent[/bold]  [dim]run_dir=[/dim]{run_dir or '-'}",
            f"[dim]model=[/dim]{cfg.model_variant}  [dim]mode=[/dim]{cfg.interaction_mode}"
            f"  [dim]rounds=[/dim]{eng._round_num}/{cfg.max_rounds}  [dim]stage=[/dim][bold yellow]{self._stage}[/bold yellow]",
        ]
        if cfg.optimize_for_class:
            lines.append(f"[dim]optimize-for=[/dim][bold cyan]{cfg.optimize_for_class}[/bold cyan]")
        else:
            lines.append("")
        return Text.from_markup("\n".join(lines))

    def _render_body(self) -> Group:
        return Group(self._render_left_table(), self._render_right_table())
        # NOTE: rendered side by side via two stacked tables for terminal-width safety;
        # a two-column Layout could overflow on narrow terminals.

    def _render_left_table(self) -> Table:
        t = Table(title="Training", title_style="bold cyan", expand=True, show_header=True)
        t.add_column("Field", style="bold", width=16)
        t.add_column("Value", style="white")
        csv = self._latest_csv
        hist = getattr(self, "_latest_history", {})
        cfg = self._engine._config

        epoch = csv.get("epoch")
        epoch_str = f"{epoch}/{cfg.epochs_per_round}" if epoch is not None else f"-/{cfg.epochs_per_round}"
        t.add_row("epoch", epoch_str)

        def _metric_row(label: str, csv_key: str, hist_key: str, *, fmt: str = ".4f") -> None:
            if csv_key not in csv:
                t.add_row(label, "-")
                return
            val = csv[csv_key]
            line = f"{val:{fmt}}"
            series = hist.get(hist_key, [])
            if len(series) > 1:
                line += f"  [dim]{sparkline(series)}[/dim]"
            t.add_row(label, line)

        _metric_row("train_loss", "train_loss", "train_loss")
        _metric_row("val_loss", "val_loss", "val_loss")
        _metric_row("mAP50", "map50", "map50")
        _metric_row("mAP50-95", "map50_95", "map50_95")
        if "precision" in csv:
            t.add_row("precision", f"{csv['precision']:.4f}")
        if "recall" in csv:
            t.add_row("recall", f"{csv['recall']:.4f}")
        _metric_row("lr", "lr", "lr", fmt=".5f")
        return t

    def _render_right_table(self) -> Table:
        eng = self._engine
        t = Table(title="Decisions", title_style="bold cyan", expand=True, show_header=True)
        t.add_column("Field", style="bold", width=16)
        t.add_column("Value", style="white")

        # Decision color counts
        counts = {"green": 0, "yellow": 0, "red": 0}
        for d in eng._decision_log:
            c = d.get("color", "") if isinstance(d, dict) else ""
            if c in counts:
                counts[c] += 1

        t.add_row("completed", str(len(eng._history)))
        t.add_row("best_round", str(eng._best_round))
        t.add_row("best_score", f"{eng._best_score:.4f}")
        if eng._history:
            round_scores = [r.score for r in eng._history]
            t.add_row("score trend", sparkline(round_scores))
        t.add_row("🟢 green", str(counts["green"]))
        t.add_row("🟡 yellow", str(counts["yellow"]))
        t.add_row("🔴 red", str(counts["red"]))
        t.add_row("red_streak", str(eng._red_tracker.count))

        p = eng._current_params
        if p is not None:
            t.add_row("lr0", f"{p.lr0:.5f}")
            t.add_row("batch", str(p.batch))
            t.add_row("mosaic", f"{p.mosaic:.3f}")
            t.add_row("mixup", f"{p.mixup:.3f}")
        return t

    def _render_footer(self) -> Panel:
        lines = list(_log_buffer)
        body = "\n".join(lines) if lines else "[dim](no log output yet)[/dim]"
        return Panel(Text.from_markup(body), title="[dim]log[/dim]", border_style="dim", padding=(0, 1))
