"""Ultralytics AMP check weights — not used for training.

``check_amp()`` hardcodes ``yolo26n.pt`` to compare FP32 vs AMP inference.
Training still uses the configured model (e.g. yolo26s.pt).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

AMP_CHECK_WEIGHT = "yolo26n.pt"
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


def _is_shadow_mount(path: Path) -> bool:
    """Docker creates an empty directory when mounting a missing host file."""
    return path.exists() and path.is_dir()


def _warn_shadow_mount(path: Path) -> None:
    if _is_shadow_mount(path):
        logger.error(
            "%s is a directory, not a weight file — Docker created this because the host "
            "file was missing when you used -v .../%s:/app/%s. "
            "Fix: bash scripts/prefetch_weights.sh, then mount "
            '-v "$(pwd)/weights:/app/weights:ro".',
            path,
            AMP_CHECK_WEIGHT,
            AMP_CHECK_WEIGHT,
        )


def _find_local_amp_weight(work_dir: Path) -> Path | None:
    candidates = [
        work_dir / "weights" / AMP_CHECK_WEIGHT,
        work_dir.parent / "weights" / AMP_CHECK_WEIGHT,
        work_dir / AMP_CHECK_WEIGHT,
    ]
    try:
        from ultralytics.utils import USER_CONFIG_DIR

        candidates.append(Path(USER_CONFIG_DIR) / "weights" / AMP_CHECK_WEIGHT)
    except Exception:
        pass
    for path in candidates:
        if _is_shadow_mount(path):
            continue
        if _is_valid_weight(path):
            return path
    return None


def _install_weight(source: Path, dest: Path) -> bool:
    """Copy ``source`` to ``dest`` if ``dest`` is not a shadow directory."""
    if _is_shadow_mount(dest):
        _warn_shadow_mount(dest)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def _require_cwd_amp_weight(work_dir: Path) -> None:
    """Ultralytics ``check_amp`` loads ``yolo26n.pt`` from the process cwd."""
    cwd_weight = work_dir / AMP_CHECK_WEIGHT
    if _is_valid_weight(cwd_weight):
        return
    if _is_shadow_mount(cwd_weight):
        raise RuntimeError(
            f"{cwd_weight} is a directory (bad Docker volume mount). "
            "On the host: rm -rf yolo26n.pt; bash scripts/prefetch_weights.sh; "
            'docker run with -v "$(pwd)/weights:/app/weights:ro". '
            "Or set use_amp: false in cv_agent.yaml."
        )
    raise RuntimeError(
        f"{AMP_CHECK_WEIGHT} not found under {work_dir}. "
        "Run bash scripts/prefetch_weights.sh and mount weights/ into the container."
    )


def ensure_amp_check_weights(work_dir: Path | None = None) -> None:
    """Ensure ``yolo26n.pt`` exists at cwd for Ultralytics AMP checks."""
    work_dir = work_dir or Path.cwd()
    destinations = _amp_check_destinations(work_dir)

    for dest in destinations:
        _warn_shadow_mount(dest)

    if _is_valid_weight(work_dir / AMP_CHECK_WEIGHT):
        return

    source = _find_local_amp_weight(work_dir)
    if source is not None:
        for dest in destinations:
            if not _is_valid_weight(dest):
                _install_weight(source, dest)
        _require_cwd_amp_weight(work_dir)
        logger.info(
            "AMP check weights ready (%s from %s — probe only, not training).",
            AMP_CHECK_WEIGHT,
            source,
        )
        return

    if _is_shadow_mount(work_dir / AMP_CHECK_WEIGHT):
        raise RuntimeError(
            f"{work_dir / AMP_CHECK_WEIGHT} is a directory (bad Docker mount). "
            'Use -v "$(pwd)/weights:/app/weights:ro" or use_amp: false.'
        )

    try:
        from ultralytics.utils.downloads import safe_download

        target = work_dir / AMP_CHECK_WEIGHT
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s for Ultralytics AMP check (not your training model)...",
            AMP_CHECK_WEIGHT,
        )
        safe_download(url=AMP_CHECK_URL, file=str(target))
        for dest in destinations[1:]:
            if not _is_valid_weight(dest):
                _install_weight(target, dest)
        _require_cwd_amp_weight(work_dir)
    except Exception as e:
        logger.warning(
            "Could not prepare %s: %s. Prefetch: bash scripts/prefetch_weights.sh",
            AMP_CHECK_WEIGHT,
            e,
        )
        _require_cwd_amp_weight(work_dir)
