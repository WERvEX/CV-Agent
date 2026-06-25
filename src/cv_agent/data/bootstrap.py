"""Dataset bootstrap — download Ultralytics registry datasets (COCO, COCO128, …).

When the configured dataset YAML is missing or has no images on disk, downloads
via ``ultralytics.data.utils.check_det_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cv_agent.ui.console import log_info, log_success, log_warning
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Ultralytics built-in detection datasets cv_agent knows how to bootstrap.
COCO_DATASET = "coco.yaml"
COCO128_DATASET = "coco128.yaml"
DEFAULT_BOOTSTRAP_DATASET = COCO_DATASET

ULTRALYTICS_REGISTRY_NAMES = frozenset(
    {
        "coco.yaml",
        "coco128.yaml",
        "coco8.yaml",
        "coco128-seg.yaml",
        "coco8-seg.yaml",
    }
)

DEFAULT_DATASETS_DIR = Path("datasets")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _set_ultralytics_datasets_dir(datasets_dir: Path) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import settings as ul_settings

        ul_settings.update({"datasets_dir": str(datasets_dir.resolve())})
    except Exception:
        pass


def _resolve_dataset_root(yaml_path: Path, dataset_info: dict) -> Path:
    path = dataset_info.get("path", "")
    if not path:
        return yaml_path.parent
    p = Path(path)
    if p.is_absolute():
        return p
    cand = yaml_path.parent / p
    if cand.exists():
        return cand
    try:
        from ultralytics.utils import SETTINGS

        cand2 = Path(SETTINGS["datasets_dir"]) / p
        if cand2.exists():
            return cand2
    except Exception:
        pass
    return cand


def _split_image_count(dataset_info: dict, root: Path, split: str) -> int:
    split_path = dataset_info.get(split, "")
    if not split_path:
        return 0
    sp = Path(split_path)
    full = sp if sp.is_absolute() else root / sp
    if not full.exists():
        return 0
    if full.is_file():
        try:
            text = full.read_text(encoding="utf-8")
            return sum(1 for line in text.splitlines() if line.strip())
        except OSError:
            return 0
    return sum(1 for f in full.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)


def yaml_has_images(yaml_path: Path) -> bool:
    """Return True if train or val split appears to contain images."""
    if not yaml_path.exists():
        return False
    try:
        with open(yaml_path, encoding="utf-8") as fh:
            info = yaml.safe_load(fh) or {}
    except OSError:
        return False
    root = _resolve_dataset_root(yaml_path, info)
    return _split_image_count(info, root, "train") > 0 or _split_image_count(info, root, "val") > 0


def _registry_name_for(path: Path) -> str | None:
    name = path.name
    if name in ULTRALYTICS_REGISTRY_NAMES:
        return name
    if path.suffix == ".yaml" and not path.is_absolute():
        return name
    return None


def _download_registry_dataset(datasets_dir: Path, dataset_name: str) -> Path | None:
    """Download a dataset via Ultralytics ``check_det_dataset``."""
    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        logger.warning("ultralytics not available — cannot bootstrap dataset.")
        return None

    _set_ultralytics_datasets_dir(datasets_dir)
    name = dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"

    try:
        log_info(f"Downloading {name} into {datasets_dir.resolve()} (may take a while for COCO) ...")
        data = check_det_dataset(name)
        yaml_path = data.get("yaml_file") if isinstance(data, dict) else None
        if yaml_path and Path(yaml_path).exists():
            return Path(yaml_path)
        logger.warning("check_det_dataset returned no usable yaml_file: %s", data)
        return None
    except Exception as e:
        logger.warning("Ultralytics dataset download failed: %s", e)
        return None


def ensure_dataset(data_yaml: Path | None, datasets_dir: Path | None = None) -> Path:
    """Return a dataset YAML path, downloading registry datasets when needed."""
    target_dir = datasets_dir or DEFAULT_DATASETS_DIR

    if data_yaml is None:
        log_info(f"No dataset specified — bootstrapping {DEFAULT_BOOTSTRAP_DATASET}.")
        resolved = _download_registry_dataset(target_dir, DEFAULT_BOOTSTRAP_DATASET)
        if resolved is None:
            raise FileNotFoundError(
                f"Could not download {DEFAULT_BOOTSTRAP_DATASET}. Check network access."
            )
        log_success(f"Dataset ready: {resolved}")
        return resolved

    registry = _registry_name_for(data_yaml)

    if data_yaml.exists() and yaml_has_images(data_yaml):
        return data_yaml

    if data_yaml.exists() and not yaml_has_images(data_yaml):
        log_warning(f"Dataset YAML found but no images on disk: {data_yaml}")

    download_name = registry or data_yaml.name
    if not download_name.endswith(".yaml"):
        download_name = f"{download_name}.yaml"

    if registry or not data_yaml.exists():
        if not data_yaml.exists():
            log_warning(f"Dataset YAML not found at {data_yaml} — downloading {download_name}.")
        resolved = _download_registry_dataset(target_dir, download_name)
        if resolved is not None:
            log_success(f"Dataset ready: {resolved}")
            return resolved

    if data_yaml.exists():
        return data_yaml

    raise FileNotFoundError(
        f"Could not resolve dataset {data_yaml}. Provide a valid --data-yaml or use "
        f"a built-in name such as {COCO_DATASET} or {COCO128_DATASET}."
    )
