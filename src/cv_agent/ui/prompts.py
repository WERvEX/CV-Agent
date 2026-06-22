"""questionary wrapper functions for interactive prompts.

Thin wrappers around questionary that provide Rich-styled prompt text
and graceful keyboard interrupt handling.
"""

from __future__ import annotations

import sys
from typing import Any

import questionary

# Custom style for questionary — uses cyan for questions, green for selections
Q_STYLE = questionary.Style([
    ("qmark", "fg:ansicyan bold"),
    ("question", "fg:ansicyan bold"),
    ("answer", "fg:ansigreen bold"),
    ("pointer", "fg:ansicyan bold"),
    ("highlighted", "fg:ansicyan bold"),
    ("selected", "fg:ansigreen"),
    ("separator", "fg:ansigray"),
    ("instruction", "fg:ansigray italic"),
    ("text", ""),
    ("disabled", "fg:ansigray italic"),
])


def _handle_interrupt(func):
    """Decorator to catch KeyboardInterrupt and exit gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n")
            print("Interrupted by user.")
            sys.exit(0)
    return wrapper


@_handle_interrupt
def confirm(message: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns True/False."""
    return questionary.confirm(
        message,
        default=default,
        style=Q_STYLE,
    ).unsafe_ask()


@_handle_interrupt
def text(message: str, default: str = "", multiline: bool = False) -> str:
    """Ask for free-text input. Returns the entered string."""
    return questionary.text(
        message,
        default=default,
        multiline=multiline,
        style=Q_STYLE,
    ).unsafe_ask()


@_handle_interrupt
def select(message: str, choices: list[str | questionary.Choice], default: str | None = None) -> str:
    """Ask to pick from a list of choices. Returns the selected value."""
    return questionary.select(
        message,
        choices=choices,
        default=default,
        style=Q_STYLE,
    ).unsafe_ask()


@_handle_interrupt
def path_input(message: str, only_directories: bool = False, default: str = "") -> str:
    """Ask for a filesystem path with validation. Returns the path string."""
    return questionary.path(
        message,
        only_directories=only_directories,
        default=default,
        style=Q_STYLE,
    ).unsafe_ask()


@_handle_interrupt
def press_enter_to_continue(message: str = "Press Enter to continue...") -> None:
    """Block until the user presses Enter."""
    questionary.text(
        message,
        default="",
        style=Q_STYLE,
    ).unsafe_ask()