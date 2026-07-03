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

from cv_agent.decision.strategy import StrategyPatch
from cv_agent.ui.terminal_charts import format_delta, sparkline

# Keys excluded from bulk metric tables (shown via per-class summary instead).
_PER_CLASS_METRIC_PREFIXES = ("mAP50_class_", "precision_class_", "recall_class_")

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

    metadata = decision.get("metadata") or {}
    if isinstance(metadata, dict):
        phase = metadata.get("decision_phase")
        thresholds = metadata.get("effective_thresholds")
        if phase and isinstance(thresholds, dict):
            green = thresholds.get("green_threshold_pct")
            soft_red = thresholds.get("soft_red_threshold_pct")
            red = thresholds.get("red_threshold_pct")
            table.add_row(
                "Decision phase",
                f"{phase} (green >= {green:+.2f}%, soft red <= {soft_red:+.2f}%, red <= {red:+.2f}%)",
            )
        recent_median = metadata.get("recent_median_score")
        delta_recent = metadata.get("delta_vs_recent_median_pct")
        if recent_median is not None and delta_recent is not None:
            table.add_row(
                "Recent median",
                f"{float(recent_median):.4f} ({float(delta_recent):+.2f}% vs recent)",
            )
        volatility = metadata.get("recent_volatility")
        if volatility is not None:
            relaxed = " relaxed" if metadata.get("volatility_relaxed") else ""
            table.add_row("Volatility", f"{float(volatility):.4f}{relaxed}")
        if metadata.get("recent_median_guard"):
            table.add_row("Guard", "Recent median prevented hard RED")

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


def summarize_per_class_map50(
    metrics: dict[str, float],
    class_names: dict[int, str] | None = None,
    *,
    worst_n: int = 3,
    best_n: int = 3,
) -> str | None:
    """Compact one-line per-class mAP50 summary (count, avg, worst/best)."""
    entries: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if not key.startswith("mAP50_class_"):
            continue
        cid = int(key.rsplit("_", 1)[-1])
        label = class_names.get(cid, str(cid)) if class_names else str(cid)
        entries.append((label, float(value)))

    if not entries:
        return None

    entries.sort(key=lambda item: item[1])
    values = [v for _, v in entries]
    avg = sum(values) / len(values)
    parts = [
        f"{len(entries)} classes",
        f"avg {avg:.3f}",
        f"min {values[0]:.3f}",
        f"max {values[-1]:.3f}",
    ]
    if worst_n > 0:
        worst = entries[:worst_n]
        parts.append("low: " + ", ".join(f"{n} {v:.3f}" for n, v in worst))
    if best_n > 0 and len(entries) > worst_n:
        best = list(reversed(entries[-best_n:]))
        parts.append("high: " + ", ".join(f"{n} {v:.3f}" for n, v in best))
    return " | ".join(parts)


def print_guidance_applied(
    *,
    before_params: dict[str, Any],
    after_params: dict[str, Any],
    source: str = "regex",
    reason: str = "",
    raw_text: str = "",
    constraints_meta: dict[str, Any] | None = None,
    pause: bool = False,
) -> None:
    """Show how user/LLM guidance changed hyperparameters before the next round."""
    from cv_agent.interaction.diff_renderer import render_diff
    from cv_agent.ui.prompts import press_enter_to_continue

    changed = {
        key: (before_params.get(key), after_params.get(key))
        for key in sorted(set(before_params) | set(after_params))
        if before_params.get(key) != after_params.get(key)
    }

    table = Table(
        title="[bold cyan]Guidance Applied[/bold cyan]",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value", style="bright_white")

    table.add_row("Parser", f"[bold]{source.upper()}[/bold]")
    if raw_text:
        table.add_row("Your input", raw_text)
    if reason:
        table.add_row("Interpretation", reason)
    if constraints_meta:
        frozen = constraints_meta.get("frozen_fields") or []
        if frozen:
            table.add_row("Frozen", ", ".join(frozen))
        mults = constraints_meta.get("multipliers") or {}
        if mults:
            table.add_row("Multipliers", ", ".join(f"{k}×{v}" for k, v in mults.items()))
        sets = constraints_meta.get("adjustments") or {}
        if sets:
            table.add_row("Set values", ", ".join(f"{k}={v}" for k, v in sets.items()))

    console.print()
    if changed:
        console.print(Panel(table, border_style="cyan"))
        console.print(render_diff(before_params, after_params, title="Changes from guidance"))
        log_success(f"Guidance applied — {len(changed)} parameter(s) updated ({source}).")
    else:
        table.add_row("Result", "[yellow]No parameter changes[/yellow] (check phrasing or frozen fields)")
        console.print(Panel(table, border_style="yellow"))
        log_warning("Guidance parsed but did not change any hyperparameters.")

    if pause:
        press_enter_to_continue("Press Enter to continue…")
    console.print()


def print_strategy_patch(patch: StrategyPatch) -> None:
    """Render the active LLM/controller strategy without verbose metadata."""
    table = Table(
        title="[bold cyan]LLM Strategy[/bold cyan]",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value", style="bright_white", overflow="fold")

    table.add_row("Phase", patch.phase.value)
    table.add_row("Confidence", f"{patch.confidence:.2f}")
    if patch.max_trials_for_phase is not None:
        table.add_row("Phase trials", str(patch.max_trials_for_phase))
    if patch.reason:
        table.add_row("Reason", patch.reason)
    if patch.search_space_patch:
        bounds = ", ".join(
            f"{key}=[{low:g}, {high:g}]"
            for key, (low, high) in sorted(patch.search_space_patch.items())
        )
        table.add_row("Search bounds", bounds)
    if patch.freeze:
        table.add_row("Frozen", ", ".join(sorted(patch.freeze)))
    if patch.objective_weights is not None:
        weights = patch.objective_weights.normalized().model_dump()
        active = {key: value for key, value in weights.items() if value > 0}
        table.add_row(
            "Objective",
            ", ".join(f"{key}={value:.2f}" for key, value in active.items()),
        )

    console.print()
    console.print(Panel(table, border_style="cyan"))
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
    class_names: dict[int, str] | None = None,
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

    per_class_summary = summarize_per_class_map50(metrics, class_names)
    if per_class_summary and not optimize_for_class:
        table.add_row("Per-class mAP50", f"[dim]{per_class_summary}[/dim]")

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

    items = [
        (k, v) for k, v in metrics.items()
        if not any(k.startswith(p) for p in _PER_CLASS_METRIC_PREFIXES)
    ]
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
