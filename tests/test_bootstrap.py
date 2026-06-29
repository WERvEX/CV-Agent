from __future__ import annotations

from pathlib import Path

import yaml

from cv_agent.data.bootstrap import ULTRALYTICS_REGISTRY_NAMES, ensure_dataset, yaml_has_images


def test_yaml_has_images_false_when_missing(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [a]\n", encoding="utf-8")
    assert yaml_has_images(yaml_path) is False


def test_yaml_has_images_true_when_train_has_images(tmp_path):
    root = tmp_path / "ds"
    train = root / "images" / "train"
    train.mkdir(parents=True)
    (train / "a.jpg").write_bytes(b"x")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "val": "images/val", "names": ["a"]}),
        encoding="utf-8",
    )
    assert yaml_has_images(yaml_path) is True


def test_registry_names_include_coco():
    assert "coco.yaml" in ULTRALYTICS_REGISTRY_NAMES
    assert "coco128.yaml" in ULTRALYTICS_REGISTRY_NAMES


def test_ensure_dataset_returns_existing_populated_yaml(tmp_path):
    root = tmp_path / "ds"
    train = root / "images" / "train"
    train.mkdir(parents=True)
    (train / "a.jpg").write_bytes(b"x")
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        yaml.dump({"path": str(root), "train": "images/train", "names": ["a"]}),
        encoding="utf-8",
    )
    assert ensure_dataset(yaml_path, datasets_dir=tmp_path / "datasets") == yaml_path.resolve()


def test_ensure_dataset_prefers_downloaded_registry_yaml_over_bundled_spec(tmp_path, monkeypatch):
    """Bundled coco128.yaml exists but images live under datasets_dir (Ultralytics layout)."""
    bundled = tmp_path / "coco128.yaml"
    bundled.write_text(
        "path: coco128\ntrain: images/train2017\nval: images/train2017\nnames: [a]\n",
        encoding="utf-8",
    )
    datasets_dir = tmp_path / "datasets"
    root = datasets_dir / "coco128" / "images" / "train2017"
    root.mkdir(parents=True)
    (root / "a.jpg").write_bytes(b"x")
    downloaded_yaml = datasets_dir / "coco128.yaml"
    downloaded_yaml.write_text(
        yaml.dump(
            {
                "path": "coco128",
                "train": "images/train2017",
                "val": "images/train2017",
                "names": ["a"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    resolved = ensure_dataset(bundled, datasets_dir=datasets_dir)
    assert resolved == downloaded_yaml.resolve()
