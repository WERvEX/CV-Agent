from __future__ import annotations

from pathlib import Path

import yaml

from cv_agent.data.paths import resolve_dataset_root


def test_resolve_dataset_root_finds_ultralytics_download_under_datasets_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundled = tmp_path / "app" / "coco128.yaml"
    bundled.parent.mkdir(parents=True)
    bundled.write_text(
        "path: coco128\ntrain: images/train2017\nval: images/train2017\n",
        encoding="utf-8",
    )
    root = tmp_path / "datasets" / "coco128" / "images" / "train2017"
    root.mkdir(parents=True)
    (root / "a.jpg").write_bytes(b"x")

    resolved = resolve_dataset_root(bundled, yaml.safe_load(bundled.read_text(encoding="utf-8")))
    assert resolved == (tmp_path / "datasets" / "coco128").resolve()
