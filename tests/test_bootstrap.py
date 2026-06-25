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
    assert ensure_dataset(yaml_path, datasets_dir=tmp_path / "datasets") == yaml_path
