"""Dataset bootstrap — download Ultralytics registry datasets (COCO, COCO128, …).

When the configured dataset YAML is missing or has no images on disk, downloads
via ``ultralytics.data.utils.check_det_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cv_agent.data.paths import candidate_datasets_dirs, resolve_dataset_root, resolve_datasets_dir
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
_GENERATED_DATASET_DIR = Path(".cv_agent_datasets")

_COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def _set_ultralytics_datasets_dir(datasets_dir: Path) -> None:
    datasets_dir = datasets_dir.resolve()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import settings as ul_settings

        ul_settings.update({"datasets_dir": str(datasets_dir)})
    except Exception:
        pass


def _find_registry_yaml(dataset_name: str, datasets_dir: Path) -> Path | None:
    """Locate a registry dataset YAML after Ultralytics download."""
    name = dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"
    stem = Path(name).stem
    for root in candidate_datasets_dirs(datasets_dir):
        for candidate in (root / name, root / stem / name):
            if candidate.is_file() and yaml_has_images(candidate):
                return candidate.resolve()
    return None


def _registry_dataset_info(dataset_name: str, dataset_root: Path) -> dict | None:
    name = dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"
    if name == COCO128_DATASET:
        return {
            "path": str(dataset_root.resolve()),
            "train": "images/train2017",
            "val": "images/train2017",
            "names": _COCO_NAMES,
        }
    if name == COCO_DATASET:
        return {
            "path": str(dataset_root.resolve()),
            "train": "images/train2017",
            "val": "images/val2017",
            "test": "images/test2017",
            "names": _COCO_NAMES,
        }
    return None


def _find_local_registry_dataset(dataset_name: str, datasets_dir: Path) -> Path | None:
    """Generate a small YAML for an already-mounted COCO/COCO128 dataset."""
    name = dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"
    stem = Path(name).stem
    for root in candidate_datasets_dirs(datasets_dir):
        dataset_root = root / stem
        info = _registry_dataset_info(name, dataset_root)
        if info is None or not dataset_root.exists():
            continue
        if _split_image_count(info, dataset_root, "train") == 0 and _split_image_count(info, dataset_root, "val") == 0:
            continue
        _GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        generated = _GENERATED_DATASET_DIR / name
        generated.write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
        log_success(f"Dataset ready: {generated.resolve()} -> {dataset_root.resolve()}")
        return generated.resolve()
    return None


def _resolve_registry_yaml(dataset_name: str, datasets_dir: Path) -> Path | None:
    located = _find_registry_yaml(dataset_name, datasets_dir)
    if located is not None:
        return located
    return _find_local_registry_dataset(dataset_name, datasets_dir)


def _resolve_dataset_root(yaml_path: Path, dataset_info: dict) -> Path:
    return resolve_dataset_root(yaml_path, dataset_info)


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
        if yaml_path:
            resolved = Path(yaml_path).resolve()
            if resolved.is_file() and yaml_has_images(resolved):
                return resolved
        located = _find_registry_yaml(name, datasets_dir)
        if located is not None:
            return located
        logger.warning("check_det_dataset returned no usable yaml_file: %s", data)
        return None
    except Exception as e:
        logger.warning("Ultralytics dataset download failed: %s", e)
        return None


def ensure_dataset(data_yaml: Path | None, datasets_dir: Path | None = None) -> Path:
    """Return a dataset YAML path, downloading registry datasets when needed."""
    target_dir = resolve_datasets_dir(datasets_dir)

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
        if registry := _registry_name_for(data_yaml):
            located = _resolve_registry_yaml(registry, target_dir)
            if located is not None:
                return located
        return data_yaml.resolve()

    if data_yaml.exists() and not yaml_has_images(data_yaml):
        log_warning(f"Dataset YAML found but no images on disk: {data_yaml}")
        # Bundled registry YAML (e.g. /app/coco128.yaml) — prefer downloaded copy under datasets_dir.
        located = _resolve_registry_yaml(data_yaml.name, target_dir)
        if located is not None:
            log_success(f"Dataset ready: {located}")
            return located

    download_name = registry or data_yaml.name
    if not download_name.endswith(".yaml"):
        download_name = f"{download_name}.yaml"

    if registry or not data_yaml.exists():
        if not data_yaml.exists():
            log_warning(f"Dataset YAML not found at {data_yaml} — downloading {download_name}.")
        located = _resolve_registry_yaml(download_name, target_dir)
        if located is not None:
            return located
        resolved = _download_registry_dataset(target_dir, download_name)
        if resolved is not None:
            log_success(f"Dataset ready: {resolved}")
            return resolved

    if data_yaml.exists():
        return data_yaml.resolve()

    raise FileNotFoundError(
        f"Could not resolve dataset {data_yaml}. Provide a valid --data-yaml or use "
        f"a built-in name such as {COCO_DATASET} or {COCO128_DATASET}."
    )
