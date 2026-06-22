"""Dataset bootstrap — download COCO128 for first-run convenience.

When the user runs `cv_agent run` without specifying a dataset (or points at a
nonexistent path), this module downloads COCO128 via Ultralytics' dataset
utilities so the full training loop can execute out of the box. COCO128 is a
small (128-image) subset of COCO — ideal for smoke-testing the closed loop.
"""

from __future__ import annotations

from pathlib import Path

from cv_agent.ui.console import log_info, log_success, log_warning
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Built-in dataset name recognized by Ultralytics' registry.
COCO128_NAME = "coco128.yaml"

# Where cv_agent stores bootstrapped datasets (relative to project root).
DEFAULT_DATASETS_DIR = Path("datasets")


def _resolve_via_ultralytics(datasets_dir: Path) -> Path | None:
    """Use Ultralytics' check_det_dataset to download & resolve coco128.

    Returns the absolute path to the resolved coco128.yaml, or None on failure.
    """
    try:
        from ultralytics import settings as ul_settings  # noqa: F401
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        logger.warning("ultralytics not available — cannot bootstrap dataset.")
        return None

    try:
        # Point Ultralytics at our datasets dir so downloads land predictably.
        datasets_dir.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics import settings as ul_settings_mod
            ul_settings_mod.update({"datasets_dir": str(datasets_dir.resolve())})
        except Exception:  # settings update is best-effort
            pass

        log_info(f"Downloading COCO128 into {datasets_dir.resolve()} ...")
        data = check_det_dataset(COCO128_NAME)
        yaml_path = data.get("yaml_file") if isinstance(data, dict) else None
        if yaml_path and Path(yaml_path).exists():
            return Path(yaml_path)
        logger.warning("check_det_dataset returned no usable yaml_file: %s", data)
        return None
    except Exception as e:  # network failure, corrupt download, etc.
        logger.warning("Ultralytics dataset download failed: %s", e)
        return None


def ensure_dataset(data_yaml: Path | None, datasets_dir: Path | None = None) -> Path:
    """Return a valid dataset yaml path, downloading COCO128 if needed.

    Args:
        data_yaml: The user-supplied dataset yaml path (may be None or missing).
        datasets_dir: Where to place bootstrapped datasets. Defaults to
            ``datasets/`` under the current working directory.

    Returns:
        A Path to an existing dataset YAML.

    Raises:
        FileNotFoundError: If the user explicitly pointed at a path that does
            not exist AND we could not fall back to COCO128.
    """
    if data_yaml is not None and data_yaml.exists():
        return data_yaml

    if data_yaml is not None:
        log_warning(f"Dataset YAML not found at {data_yaml} — falling back to COCO128.")
    else:
        log_info("No dataset specified — bootstrapping COCO128 for first run.")

    target_dir = datasets_dir or DEFAULT_DATASETS_DIR
    yaml_path = _resolve_via_ultralytics(target_dir)

    if yaml_path is None:
        raise FileNotFoundError(
            "Could not bootstrap COCO128. Either provide a valid dataset with "
            "--data-yaml, or ensure internet access so COCO128 can be downloaded."
        )

    log_success(f"COCO128 ready: {yaml_path}")
    return yaml_path
