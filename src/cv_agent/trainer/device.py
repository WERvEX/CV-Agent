"""Resolve Ultralytics ``device`` and DataLoader ``workers`` from config / env."""

from __future__ import annotations

import os
import sys

from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


def resolve_device(spec: str | None = None) -> str | int | list[int]:
    """Map config/env to Ultralytics ``device`` (DDP when multiple IDs).

    Args:
        spec: ``auto`` (all visible CUDA GPUs), ``cpu``, ``0``, or ``0,1,2,3``.
              Falls back to ``CV_AGENT_DEVICE`` env, then ``auto``.

    Returns:
        Value suitable for ``YOLO.train(device=...)``.
    """
    raw = (spec or os.environ.get("CV_AGENT_DEVICE") or "auto").strip().lower()
    if raw in ("", "auto", "all"):
        try:
            import torch

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 1:
                    ids = list(range(count))
                    logger.info("Using %d visible GPU(s) for training (DDP): %s", count, ids)
                    return ids
                logger.info("Using single visible GPU: 0")
                return 0
        except ImportError:
            pass
        logger.info("CUDA not available — using CPU")
        return "cpu"

    if raw == "cpu":
        return "cpu"

    if "," in raw:
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not ids:
            raise ValueError(f"Invalid device spec: {spec!r}")
        logger.info("Using configured GPU(s): %s", ids)
        return ids if len(ids) > 1 else ids[0]

    return int(raw)


def resolve_workers(spec: int | None = None) -> int:
    """DataLoader workers — 0 on Windows by default, 8 on Linux."""
    if spec is not None:
        return max(0, spec)
    env = os.environ.get("CV_AGENT_WORKERS")
    if env is not None and env.strip() != "":
        return max(0, int(env))
    return 0 if sys.platform == "win32" else 8
