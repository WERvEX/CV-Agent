"""Rich terminal UI components.

Provides a module-level Console singleton and styled helper functions
for consistent, color-coded CLI output throughout cv_agent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from cv_agent.ui.terminal_charts import format_delta, sparkline

# Singleton console for the entire application
console = Console()


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return current timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Styled log helpers
# ---------------------------------------------------------------------------

def _emit(line: str) -> None:
    """Print a formatted log line, or buffer it when a Live panel is active.

    When cv_agent.ui.live_panel has a Live region open, direct console.print
    would corrupt the panel; instead the line is appended to the panel's
    scrolling footer via push_log_line.
    """
    try:
        from cv_agent.ui import live_panel
        if live_panel.is_live_active():
            live_panel.push_log_line(line)
            return
    except ImportError:
        pass
    console.print(line)


def log_info(msg: str) -> None:
    _emit(f"[dim]{_ts()}[/dim] [bold blue]INFO[/bold blue]    {msg}")


def log_success(msg: str) -> None:
    _emit(f"[dim]{_ts()}[/dim] [bold green]SUCCESS[/bold green] {msg}")


def log_warning(msg: str) -> None:
    _emit(f"[dim]{_ts()}[/dim] [bold yellow]WARNING[/bold yellow] {msg}")


def log_error(msg: str) -> None:
    _emit(f"[dim]{_ts()}[/dim] [bold red]ERROR[/bold red]   {msg}")


def log_decision(color: str, msg: str) -> None:
    """Log a decision with color-coded label."""
    color_map = {
        "green": "bold green",
        "yellow": "bold yellow",
        "red": "bold red",
    }
    style = color_map.get(color, "bold white")
    _emit(f"[dim]{_ts()}[/dim] [{style}]DECIDE[/{style}]  {msg}")


# ---------------------------------------------------------------------------
# Banner and section headers
# ---------------------------------------------------------------------------

def print_banner(version: str) -> None:
    """Print the cv_agent startup banner."""
    banner = Text(
        rf"""
   ___       __   ___                __
  / _ \ _  _/ /_ / _ | ___ ____  ___/ /_
 / ___// / / __// __ |/ _ `/ _ \/ _  / -_)
/_/    \_,_/\__//_/ |_|\_,_/_//_/\_,_/\__/
""",
        style="bold cyan",
    )
    console.print(banner)
    console.print(f"  [bold]cv_agent[/bold] v{version} — Automated Closed-Loop YOLO Training")
    console.print("  " + "─" * 55)
    console.print()


def print_section(title: str) -> None:
    """Print a section header."""
    line = f"[bold white on blue] {title} [/bold white on blue]"
    try:
        from cv_agent.ui import live_panel
        if live_panel.is_live_active():
            live_panel.push_log_line(line)
            return
    except ImportError:
        pass
    console.print()
    console.print(line)
    console.print()


def print_final_summary(
    rounds_run: int,
    best_round: int,
    best_score: float,
    run_dir: Path,
    *,
    decision_log: list[dict[str, Any]] | None = None,
    round_scores: list[tuple[int, float]] | None = None,
) -> None:
    """Print a final summary table at the end of a session."""
    table = Table(title="Training Session Summary", title_style="bold cyan")
    table.add_column("Metric", style="bold", width=25)
    table.add_column("Value", style="bright_white")

    table.add_row("Rounds completed", str(rounds_run))
    table.add_row("Best round", f"#{best_round}")
    table.add_row("Best reward score", f"{best_score:.4f}")
    table.add_row("Output directory", str(run_dir))
    if round_scores:
        scores = [s for _, s in round_scores]
        table.add_row("Score trend", sparkline(scores))

    console.print()
    console.print(table)

    if decision_log:
        print_decision_timeline(decision_log, round_scores=round_scores)

    console.print()


# ---------------------------------------------------------------------------
# Decision / metric tables
# ---------------------------------------------------------------------------

def print_decision_table(decision: dict[str, Any], round_num: int) -> None:
    """Print a color-coded decision summary table."""
    color = decision.get("color", "white")
    color_style = {"green": "green", "yellow": "yellow", "red": "red"}.get(color, "white")

    table = Table(
        title=f"Round {round_num} Decision — [bold {color_style}]{color.upper()}[/bold {color_style}]",
        title_style=f"bold {color_style}",
    )
    table.add_column("Field", style="bold", width=18)
    table.add_column("Value", style="bright_white")

    for key, val in decision.items():
        if isinstance(val, (dict, list)):
            val = json.dumps(val, indent=2)
        table.add_row(str(key), str(val))

    console.print()
    console.print(table)
    console.print()


def print_decision_recommendation(
    decision: dict[str, Any],
    round_num: int,
    current_params: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """Print a concise, human-readable controller / Optuna recommendation."""
    color = decision.get("color", "white")
    color_style = {"green": "green", "yellow": "yellow", "red": "red"}.get(color, "white")
    action = decision.get("action", "?")
    reason = decision.get("reason", "")
    should_rollback = decision.get("should_rollback", False)

    table = Table(
        title=f"Round {round_num} — [bold {color_style}]Decision ({color.upper()})[/bold {color_style}]",
        title_style=f"bold {color_style}",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Field", style="bold cyan", width=22)
    table.add_column("Value", style="bright_white")

    table.add_row("Action", str(action))
    table.add_row("Reason", reason)

    if should_rollback:
        rollback_target = checkpoint_path or decision.get("rollback_checkpoint") or "best checkpoint"
        table.add_row("Rollback", f"[bold yellow]Yes[/bold yellow] → {rollback_target}")
    else:
        table.add_row("Rollback", "[dim]No[/dim]")

    next_params = decision.get("proposed_hyperparams") or decision.get("next_hyperparams") or {}
    has_param_diff = False
    if current_params and isinstance(next_params, dict):
        has_param_diff = any(
            current_params.get(key) != new_val for key, new_val in next_params.items()
        )
        if not has_param_diff:
            table.add_row("Param changes", "[dim]None (next round keeps current params)[/dim]")
    elif next_params:
        table.add_row("Param changes", json.dumps(next_params, indent=2))

    console.print()
    console.print(Panel(table, border_style=color_style))
    if has_param_diff and current_params and isinstance(next_params, dict):
        from cv_agent.interaction.diff_renderer import render_diff

        console.print(render_diff(current_params, next_params, title="Hyperparameter Diff"))
    console.print()


def print_round_evaluation(
    *,
    round_num: int,
    score: float,
    metrics: dict[str, float],
    delta_percent: float | None,
    best_score: float | None = None,
    best_round: int | None = None,
    overfitting: bool = False,
    underfitting: bool = False,
    optimize_for_class: str | None = None,
    optimize_class_id: int | None = None,
) -> None:
    """Print evaluation summary after a training round completes."""
    table = Table(
        title=f"Round {round_num} Evaluation",
        title_style="bold cyan",
        show_header=True,
    )
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Value", style="bright_white", width=18)

    table.add_row("Reward score", f"{score:.4f}")
    table.add_row("Δ vs historical best", format_delta(delta_percent))
    if best_score is not None and best_round is not None:
        table.add_row("Historical best", f"round #{best_round} — {best_score:.4f}")

    diagnostics: list[str] = []
    if overfitting:
        diagnostics.append("[yellow]overfitting[/yellow]")
    if underfitting:
        diagnostics.append("[yellow]underfitting[/yellow]")
    table.add_row(
        "Diagnostics",
        ", ".join(diagnostics) if diagnostics else "[dim]none[/dim]",
    )

    for key, label in (
        ("mAP50", "mAP50"),
        ("mAP50_95", "mAP50-95"),
        ("precision", "precision"),
        ("recall", "recall"),
    ):
        if key in metrics:
            table.add_row(label, f"{metrics[key]:.4f}")

    if optimize_for_class and optimize_class_id is not None:
        target_key = f"mAP50_class_{optimize_class_id}"
        if target_key in metrics:
            table.add_row(
                f"Target ({optimize_for_class})",
                f"mAP50={metrics[target_key]:.4f}",
            )

    per_class = sorted(
        (k, v) for k, v in metrics.items() if k.startswith("mAP50_class_")
    )
    if per_class and not optimize_for_class:
        preview = ", ".join(f"{k.split('_')[-1]}={v:.3f}" for k, v in per_class[:6])
        if len(per_class) > 6:
            preview += f", … (+{len(per_class) - 6} classes)"
        table.add_row("Per-class mAP50", preview)

    console.print()
    console.print(table)
    console.print()


def print_decision_timeline(
    decision_log: list[dict[str, Any]],
    *,
    round_scores: list[tuple[int, float]] | None = None,
) -> None:
    """Print a compact timeline of Green/Yellow/Red decisions across rounds."""
    if not decision_log:
        return

    scores_by_round = dict(round_scores or [])
    table = Table(title="Decision Timeline", title_style="bold cyan")
    table.add_column("Round", style="bold", width=6)
    table.add_column("Color", width=8)
    table.add_column("Score", width=10)
    table.add_column("Action", width=28)
    table.add_column("Reason", overflow="fold")

    color_style = {"green": "green", "yellow": "yellow", "red": "red"}

    for idx, entry in enumerate(decision_log, start=1):
        color = entry.get("color", "?")
        style = color_style.get(color, "white")
        round_num = idx
        score = scores_by_round.get(round_num)
        score_str = f"{score:.4f}" if score is not None else "-"
        table.add_row(
            str(round_num),
            f"[{style}]{color.upper()}[/{style}]",
            score_str,
            str(entry.get("action", "?")),
            str(entry.get("reason", "")),
        )

    console.print()
    console.print(table)


def print_metrics_table(metrics: dict[str, float], title: str = "Current Metrics") -> None:
    """Print a table of evaluation metrics."""
    table = Table(title=title, title_style="bold cyan")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Value", style="bright_white", width=12)
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Value", style="bright_white", width=12)

    items = list(metrics.items())
    for i in range(0, len(items), 2):
        row = []
        for j in range(2):
            if i + j < len(items):
                k, v = items[i + j]
                row.append(k)
                row.append(f"{v:.4f}")
            else:
                row.extend(["", ""])
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def print_validation_issues(issues: list[dict[str, Any]]) -> None:
    """Print a tree of dataset validation issues."""
    tree = Tree("[bold red]Dataset Validation Issues[/bold red]")

    for issue in issues:
        severity_style = "red" if issue.get("severity") == "error" else "yellow"
        node = tree.add(f"[{severity_style}][{issue.get('severity', '?').upper()}][/{severity_style}] {issue.get('category', 'unknown')}")
        node.add(f"[dim]Detail:[/dim] {issue.get('detail', '')}")
        if issue.get("suggestion"):
            node.add(f"[dim]Suggestion:[/dim] [green]{issue['suggestion']}[/green]")

    console.print()
    console.print(tree)
    console.print()


def print_diff(old_text: str, new_text: str, title: str = "Configuration Changes") -> None:
    """Print a unified diff view using Rich Panel."""
    # Render old and new side-by-side in panels
    panel = Panel(
        Text.from_markup(f"[red]- Old: {old_text[:100]}...[/red]\n[green]+ New: {new_text[:100]}...[/green]"),
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow",
    )
    console.print()
    console.print(panel)
    console.print()