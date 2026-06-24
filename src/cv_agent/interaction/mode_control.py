"""Runtime Ask / Auto mode switching between training rounds."""

from __future__ import annotations

import math
import sys
import time
from typing import Literal

from collections.abc import Callable

from cv_agent.interaction.types import SessionQuit
from cv_agent.ui.console import console, log_info
from cv_agent.ui.prompts import select_action

ModeChoice = Literal["ask", "auto"]

DEFAULT_AUTO_PROMPT_SECONDS = 10.0


def offer_mode_control(
    current_mode: ModeChoice,
    round_num: int,
    auto_timeout_seconds: float = DEFAULT_AUTO_PROMPT_SECONDS,
    on_save_checkpoint: Callable[[], None] | None = None,
) -> ModeChoice:
    """Let the user pick Ask or Auto for reviewing **this round** before decision review runs.

    Ask mode: interactive menu (no timeout).
    Auto mode: countdown; no keypress within ``auto_timeout_seconds`` auto-approves this round.

    The chosen mode is persisted for subsequent rounds until changed again.

    Returns:
        The interaction mode to use for this round's review ('ask' or 'auto').

    Raises:
        SessionQuit: If the user chooses to quit training.
    """
    if not sys.stdin.isatty():
        return current_mode

    if current_mode == "ask":
        while True:
            choices: list[tuple[str, str]] = [
                ("ask", "Review in Ask mode (decision + hyperparameters)"),
                ("auto", "Switch to Auto (auto-approve this round)"),
            ]
            if on_save_checkpoint is not None:
                choices.append(("save", "Save current model and hyperparameters"))
            choice = select_action(
                f"Round {round_num} training finished. How should this round be reviewed?",
                choices,
                default_key="ask",
            )
            if choice == "save" and on_save_checkpoint is not None:
                on_save_checkpoint()
                continue
            if choice == "auto":
                log_info("Switching to Auto mode for this round's review.")
            return choice

    return _auto_mode_checkpoint(round_num=round_num, timeout=auto_timeout_seconds)


def _auto_mode_checkpoint(round_num: int, timeout: float) -> ModeChoice:
    """Auto-mode prompt before reviewing this round; timeout defaults to auto-approve."""
    console.print()
    console.print(
        f"[bold yellow]Round {round_num} training finished — Auto mode[/bold yellow]\n"
        "[dim]Press [cyan]A[/cyan] to review this round in Ask, [cyan]Q[/cyan] to quit. "
        "Keys only work during the countdown below — not while training is running "
        "(use Ctrl+C to stop a running round).[/dim]"
    )

    key = _wait_for_key_or_timeout(timeout)
    if key is None:
        log_info(f"No input — auto-approving this round ({timeout:.0f}s timeout).")
        return "auto"

    lowered = key.lower()
    if lowered == "a":
        log_info("Switching to Ask mode for this round's review.")
        return "ask"
    if lowered == "q":
        raise SessionQuit("User quit from Auto mode checkpoint.")
    if lowered in ("\r", "\n", " "):
        log_info("Auto-approving this round.")
        return "auto"

    log_info(f"Unrecognized key '{key}' — auto-approving this round.")
    return "auto"


def _drain_input_buffer() -> None:
    """Discard buffered keystrokes so stale input does not skip the countdown."""
    if sys.platform == "win32":
        import msvcrt

        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
    else:
        import select

        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)


def _poll_key_once() -> str | None:
    """Return one key if available right now, else None."""
    if sys.platform == "win32":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            return None
        return ch

    import select

    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _wait_for_key_or_timeout(timeout: float) -> str | None:
    """Wait up to ``timeout`` seconds, showing a live countdown; return a key or None."""
    _drain_input_buffer()
    deadline = time.monotonic() + timeout
    last_display: int | None = None

    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return None

        display = max(1, int(math.ceil(remaining)))
        if display != last_display:
            sys.stdout.write(
                f"\r  Auto continues in {display:2d}s — [A] Ask  [Q] Quit"
            )
            sys.stdout.flush()
            last_display = display

        key = _poll_key_once()
        if key is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return key

        time.sleep(0.05)
