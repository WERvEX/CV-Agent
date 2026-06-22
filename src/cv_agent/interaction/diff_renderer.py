"""Configuration diff rendering for Ask-before-edit mode.

Uses diff_match_patch for word-level diffs and Rich Syntax for
color-coded display (+ green, - red).
"""

from __future__ import annotations

from typing import Any

import diff_match_patch as dmp_module
import yaml
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from cv_agent.ui.console import console


def yaml_to_lines(data: dict[str, Any]) -> str:
    """Convert a dict to a YAML string for diff comparison."""
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def compute_diff(old_data: dict[str, Any], new_data: dict[str, Any]) -> list[tuple[int, str]]:
    """Compute a word-level diff between two config dicts.

    Args:
        old_data: Previous configuration dict.
        new_data: Proposed configuration dict.

    Returns:
        List of (operation, text) tuples. -1=delete, 0=equal, 1=insert.
    """
    old_text = yaml_to_lines(old_data)
    new_text = yaml_to_lines(new_data)

    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(old_text, new_text)
    dmp.diff_cleanupSemantic(diffs)

    return diffs


def render_diff(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    title: str = "Configuration Changes",
    context_lines: int = 3,
) -> Panel:
    """Render a color-coded Rich Panel showing the diff between old and new config.

    Args:
        old_data: Previous configuration.
        new_data: Proposed new configuration.
        title: Panel title.
        context_lines: Number of context lines to show around changes.

    Returns:
        Rich Panel containing the colored diff.
    """
    diffs = compute_diff(old_data, new_data)

    text = Text()
    for op, chunk in diffs:
        if op == 0:  # equal
            text.append(chunk, style="dim")
        elif op == -1:  # delete
            for line in chunk.split("\n"):
                if line.strip():
                    text.append(f"- {line}\n", style="bold red")
        elif op == 1:  # insert
            for line in chunk.split("\n"):
                if line.strip():
                    text.append(f"+ {line}\n", style="bold green")

    return Panel(
        text,
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    )


def render_single_value_diff(
    key: str,
    old_value: Any,
    new_value: Any,
) -> Panel:
    """Render a diff for a single parameter change.

    Args:
        key: Parameter name.
        old_value: Previous value.
        new_value: Proposed new value.

    Returns:
        Rich Panel with the comparison.
    """
    text = Text()
    text.append(f"  {key}: ", style="bold white")
    text.append(f"{old_value}", style="bold red")
    text.append(" → ", style="dim")
    text.append(f"{new_value}\n", style="bold green")

    return Panel(
        text,
        title="[bold yellow]Parameter Change[/bold yellow]",
        border_style="yellow",
    )