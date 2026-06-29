"""Shared Ultralytics dataset path resolution (Docker / dual-mount safe)."""

from __future__ import annotations

import os
from pathlib import Path


def candidate_datasets_dirs(preferred: Path | None = None) -> list[Path]:
    """Directories where Ultralytics may store auto-downloaded datasets."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    if preferred is not None:
        add(preferred)
    env = os.environ.get("CV_AGENT_DATASETS_DIR", "").strip()
    if env:
        add(Path(env))
    add(Path("datasets"))
    add(Path("/app/datasets"))
    add(Path("/datasets"))
    try:
        from ultralytics.utils import SETTINGS

        add(Path(SETTINGS["datasets_dir"]))
    except Exception:
        pass
    return ordered


def resolve_datasets_dir(explicit: Path | None = None) -> Path:
    """Preferred datasets root for bootstrap downloads."""
    if explicit is not None:
        return explicit.resolve()
    dirs = candidate_datasets_dirs()
    return dirs[0] if dirs else Path("datasets").resolve()


def resolve_dataset_root(yaml_path: Path, dataset_info: dict) -> Path:
    """Resolve the dataset root (YAML ``path`` field), Ultralytics-style."""
    path = dataset_info.get("path", "")
    if not path:
        return yaml_path.parent

    p = Path(path)
    if p.is_absolute():
        return p

    yaml_dir = yaml_path.parent
    candidates = [yaml_dir / p]
    for ds_dir in candidate_datasets_dirs():
        candidates.append(ds_dir / p)

    for cand in candidates:
        if cand.exists():
            return cand

    return yaml_dir / p
