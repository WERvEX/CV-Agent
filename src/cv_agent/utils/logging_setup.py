"""Structured logging setup with Rich handler and file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

# Module-level logger — configured once by setup_logging()
logger = logging.getLogger("cv_agent")


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    rich_console: bool = True,
) -> None:
    """Configure the cv_agent root logger.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path for file-based log output.
        rich_console: Use Rich handler for console output.
    """
    root = logging.getLogger("cv_agent")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rich_console:
        rh = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        rh.setLevel(level)
        rh.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(rh)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "openai", "mlflow", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = "cv_agent") -> logging.Logger:
    """Return a child logger of the cv_agent root."""
    return logging.getLogger(name)