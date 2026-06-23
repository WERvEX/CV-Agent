"""questionary wrapper functions for interactive prompts.

Thin wrappers around questionary that provide Rich-styled prompt text
and graceful keyboard interrupt handling.
"""

from __future__ import annotations

import questionary

from cv_agent.interaction.types import SessionQuit

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

QUIT_LABEL = "Quit training session"


def _handle_interrupt(func):
    """Decorator to catch KeyboardInterrupt and SessionQuit."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SessionQuit:
            raise
        except KeyboardInterrupt:
            print("\n")
            raise SessionQuit("Interrupted by user (Ctrl+C).")
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
def select_action(
    message: str,
    choices: list[tuple[str, str]],
    default_key: str | None = None,
    include_quit: bool = True,
) -> str:
    """Select from labeled choices; raises SessionQuit if user picks quit."""
    q_choices: list[questionary.Choice] = [
        questionary.Choice(title=label, value=key) for key, label in choices
    ]
    if include_quit:
        q_choices.append(questionary.Choice(title=QUIT_LABEL, value="__quit__"))

    default = default_key if default_key is not None else (choices[0][0] if choices else None)

    result = questionary.select(
        message,
        choices=q_choices,
        default=default,
        style=Q_STYLE,
    ).unsafe_ask()

    if result == "__quit__":
        raise SessionQuit("User chose to quit from prompt.")
    return result


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
    """Block until the user continues or chooses quit."""
    action = select_action(
        message,
        [("continue", "Continue")],
        default_key="continue",
        include_quit=True,
    )
    if action != "continue":
        raise SessionQuit("User chose to quit.")
