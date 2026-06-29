"""Ultralytics AMP check weights — not used for training.

``check_amp()`` hardcodes ``yolo26n.pt`` to compare FP32 vs AMP inference.
Training still uses the configured model (e.g. yolo26m.pt).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

AMP_CHECK_WEIGHT = "yolo26n.pt"
# Ultralytics assets release used by ultralytics>=8.4
AMP_CHECK_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def _amp_check_destinations(work_dir: Path) -> list[Path]:
    destinations = [work_dir / AMP_CHECK_WEIGHT]
    try:
        from ultralytics.utils import USER_CONFIG_DIR

        weights_dir = Path(USER_CONFIG_DIR) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        destinations.append(weights_dir / AMP_CHECK_WEIGHT)
    except Exception:
        pass
    return destinations


def _is_valid_weight(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 1_000_000
    except OSError:
        return False


def _find_local_amp_weight(work_dir: Path) -> Path | None:
    candidates = [
        work_dir / AMP_CHECK_WEIGHT,
        work_dir / "weights" / AMP_CHECK_WEIGHT,
        work_dir.parent / "weights" / AMP_CHECK_WEIGHT,
    ]
    try:
        from ultralytics.utils import USER_CONFIG_DIR

        candidates.append(Path(USER_CONFIG_DIR) / "weights" / AMP_CHECK_WEIGHT)
    except Exception:
        pass
    for path in candidates:
        if _is_valid_weight(path):
            return path
    return None


def ensure_amp_check_weights(work_dir: Path | None = None) -> None:
    """Ensure ``yolo26n.pt`` exists for Ultralytics AMP checks (avoids slow re-download)."""
    work_dir = work_dir or Path.cwd()
    destinations = _amp_check_destinations(work_dir)

    if any(_is_valid_weight(d) for d in destinations):
        return

    source = _find_local_amp_weight(work_dir)
    if source is not None:
        for dest in destinations:
            if not _is_valid_weight(dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        logger.info(
            "AMP check weights ready (%s copied — used only for AMP probe, not training).",
            AMP_CHECK_WEIGHT,
        )
        return

    try:
        from ultralytics.utils.downloads import safe_download

        target = destinations[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s for Ultralytics AMP compatibility check (not your training model)...",
            AMP_CHECK_WEIGHT,
        )
        safe_download(url=AMP_CHECK_URL, file=str(target))
        for dest in destinations[1:]:
            if not _is_valid_weight(dest):
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, dest)
    except Exception as e:
        logger.warning(
            "Could not prepare %s for AMP checks: %s. "
            "Prefetch with: bash scripts/prefetch_weights.sh",
            AMP_CHECK_WEIGHT,
            e,
        )
