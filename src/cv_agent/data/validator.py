"""Dataset validator — checks dataset completeness and quality.

Runs on startup before any training round. Returns a list of
ValidationIssue objects that feed into the data supplement mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from cv_agent.core.config import DataConfig
from cv_agent.data.paths import resolve_dataset_root
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    """A single dataset validation finding."""

    severity: str = Field(default="warning")   # "error" | "warning"
    category: str                               # "count", "missing_annotations", "small_objects", ...
    detail: str
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validates a YOLO-format dataset against configured thresholds."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def validate(self) -> list[ValidationIssue]:
        """Run all validation checks.

        Returns:
            List of ValidationIssue objects (empty = all checks passed).
        """
        issues: list[ValidationIssue] = []

        data_yaml = self.config.data_yaml
        if not data_yaml.exists():
            issues.append(ValidationIssue(
                severity="error",
                category="missing_yaml",
                detail=f"Dataset YAML not found: {data_yaml}",
                suggestion="Create a YOLO-format dataset YAML file with 'path', 'train', 'val', and 'names' keys.",
            ))
            return issues

        # Parse the dataset YAML
        with open(data_yaml, "r", encoding="utf-8") as fh:
            dataset_info = yaml.safe_load(fh)

        if dataset_info is None:
            issues.append(ValidationIssue(
                severity="error",
                category="empty_yaml",
                detail=f"Dataset YAML is empty: {data_yaml}",
                suggestion="Populate the YAML with dataset paths and class names.",
            ))
            return issues

        # --- Check image counts ---
        issues.extend(self._check_image_counts(dataset_info))

        # --- Check annotation completeness ---
        issues.extend(self._check_annotations(dataset_info))

        # --- Check class distribution ---
        issues.extend(self._check_class_distribution(dataset_info))

        # --- Optional: check object sizes ---
        if self.config.min_pixel_area > 0:
            issues.extend(self._check_object_sizes(dataset_info))

        return issues

    # ------------------------------------------------------------------
    # Path resolution (Ultralytics-compatible)
    # ------------------------------------------------------------------

    def _dataset_root(self, dataset_info: dict) -> Path:
        """Resolve the dataset root (the YAML ``path`` field)."""
        return resolve_dataset_root(self.config.data_yaml, dataset_info)

    def _resolve_split_path(self, dataset_info: dict, split: str) -> Path | None:
        """Resolve the image directory for a train/val/test split.

        Absolute split paths are used directly; relative ones resolve against
        the dataset root (see :meth:`_dataset_root`).
        """
        split_path = dataset_info.get(split, "")
        if not split_path:
            return None
        sp = Path(split_path)
        if sp.is_absolute():
            return sp
        return self._dataset_root(dataset_info) / split_path

    def _resolve_labels_dir(self, images_path: Path) -> Path:
        """Resolve the labels directory for an images directory.

        Tries the YOLO convention of replacing ``images`` with ``labels`` in
        the path, then the sibling ``labels/<split>`` layout.
        """
        labels_dir = Path(str(images_path).replace("images", "labels"))
        if labels_dir.exists():
            return labels_dir
        return images_path.parent / "labels" / images_path.name

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_image_counts(self, dataset_info: dict) -> list[ValidationIssue]:
        """Count images in train/val splits."""
        issues: list[ValidationIssue] = []

        for split in ("train", "val"):
            full_path = self._resolve_split_path(dataset_info, split)
            if full_path is None:
                continue

            # The path might be "path/to/images" — count image files
            if full_path.exists():
                if full_path.is_dir():
                    images = self._count_images(full_path)
                else:
                    # Might be a txt file listing paths
                    images = self._count_from_list(full_path)
            else:
                images = 0

            if images < self.config.min_images:
                issues.append(ValidationIssue(
                    severity="error",
                    category="image_count",
                    detail=f"{split} split has {images} images (minimum: {self.config.min_images})",
                    suggestion=f"Add more images to the {split} split. Consider using Roboflow Universe or OpenImages.",
                ))
            else:
                logger.info(f"{split} split: {images} images ✓")

        return issues

    def _check_annotations(self, dataset_info: dict) -> list[ValidationIssue]:
        """Cross-reference images and label files."""
        issues: list[ValidationIssue] = []

        for split in ("train", "val"):
            full_path = self._resolve_split_path(dataset_info, split)
            if full_path is None or not full_path.exists() or not full_path.is_dir():
                continue

            images = set(
                p.stem for p in full_path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            )

            labels_dir = self._resolve_labels_dir(full_path)

            labels = set()
            if labels_dir.exists():
                labels = set(
                    p.stem for p in labels_dir.iterdir()
                    if p.suffix.lower() == ".txt"
                )

            missing_labels = images - labels
            missing_images = labels - images

            if missing_labels:
                # A few label-less images (e.g. background/negative samples) are
                # normal and YOLO trains fine on them; only flag as an error
                # when a meaningful fraction is missing.
                frac = len(missing_labels) / len(images) if images else 1.0
                severity = "error" if frac > 0.05 else "warning"
                issues.append(ValidationIssue(
                    severity=severity,
                    category="missing_annotations",
                    detail=(f"{split}: {len(missing_labels)} images have no label file "
                            f"({frac*100:.1f}% of {len(images)}). "
                            f"Examples: {list(missing_labels)[:5]}"),
                    suggestion="Annotate missing images using CVAT, labelImg, or Roboflow Annotate, "
                               "or confirm they are intentional background samples.",
                ))

            if missing_images and len(missing_images) > len(labels) * 0.1:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="orphan_labels",
                    detail=f"{split}: {len(missing_images)} label files have no corresponding image.",
                    suggestion="Remove orphan label files or add the corresponding images.",
                ))

            logger.info(f"{split}: {len(images)} images, {len(labels & images)} with annotations")

        return issues

    def _check_class_distribution(self, dataset_info: dict) -> list[ValidationIssue]:
        """Check per-class annotation counts."""
        issues: list[ValidationIssue] = []

        names = dataset_info.get("names", {})
        if isinstance(names, list):
            names = {i: name for i, name in enumerate(names)}

        class_counts: dict[int, int] = {}
        for split in ("train", "val"):
            full_path = self._resolve_split_path(dataset_info, split)
            if full_path is None:
                continue

            labels_dir = self._resolve_labels_dir(full_path)
            if not labels_dir.exists():
                continue

            for label_file in labels_dir.iterdir():
                if label_file.suffix.lower() != ".txt":
                    continue
                try:
                    with open(label_file, "r") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(float(parts[0]))
                                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                except Exception:
                    continue

        ok_count = 0
        for cls_id, count in sorted(class_counts.items()):
            cls_name = names.get(cls_id, f"class_{cls_id}")
            if count < self.config.min_ann_per_class:
                issues.append(ValidationIssue(
                    severity="error",
                    category="class_count",
                    detail=(f"Class '{cls_name}' (id={cls_id}): only {count} annotations "
                            f"(minimum: {self.config.min_ann_per_class})"),
                    suggestion=(f"Add more annotated samples for '{cls_name}'. "
                                f"Roboflow Universe has many '{cls_name}' datasets."),
                ))
            else:
                ok_count += 1

        # Single-line summary instead of one line per class (avoids dozens of
        # log lines that make startup look like a hang).
        total_classes = len(class_counts)
        total_anns = sum(class_counts.values())
        logger.info(
            f"Class distribution: {total_classes} classes, {total_anns} annotations "
            f"total — {ok_count} OK, {total_classes - ok_count} below min"
        )

        return issues

    def _check_object_sizes(self, dataset_info: dict) -> list[ValidationIssue]:
        """Check for excessively small objects using label box dimensions."""
        issues: list[ValidationIssue] = []

        small_count = 0
        total_count = 0
        min_area = self.config.min_pixel_area

        for split in ("train", "val"):
            full_path = self._resolve_split_path(dataset_info, split)
            if full_path is None:
                continue

            labels_dir = self._resolve_labels_dir(full_path)
            if not labels_dir.exists():
                continue

            for label_file in labels_dir.iterdir():
                if label_file.suffix.lower() != ".txt":
                    continue
                try:
                    with open(label_file, "r") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # YOLO format: cls x_center y_center w h (normalized)
                                w = float(parts[3])
                                h = float(parts[4])
                                # Assume 640×640 image for rough area estimate
                                area = w * h * 640 * 640
                                total_count += 1
                                if area < min_area:
                                    small_count += 1
                except Exception:
                    continue

        if total_count > 0 and small_count / total_count > 0.3:
            issues.append(ValidationIssue(
                severity="warning",
                category="small_objects",
                detail=(f"{small_count}/{total_count} ({small_count/total_count*100:.1f}%) "
                        f"annotations have area < {min_area}px²"),
                suggestion="Consider higher-resolution images or tiling strategy for small object detection.",
            ))

        # Optional brightness check — lightweight sampling
        if self.config.validate_brightness:
            issues.extend(self._check_brightness(dataset_info))

        return issues

    def _check_brightness(self, dataset_info: dict) -> list[ValidationIssue]:
        """Sample brightness distribution of images."""
        issues: list[ValidationIssue] = []

        try:
            import cv2
        except ImportError:
            return []

        brightnesses = []
        sample_limit = 50

        for split in ("train", "val"):
            full_path = self._resolve_split_path(dataset_info, split)
            if full_path is None or not full_path.exists() or not full_path.is_dir():
                continue

            for img_file in full_path.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        brightnesses.append(float(np.mean(img)))
                    if len(brightnesses) >= sample_limit:
                        break
            if len(brightnesses) >= sample_limit:
                break

        if len(brightnesses) >= 10:
            mean_b = np.mean(brightnesses)
            std_b = np.std(brightnesses)

            if std_b < 15:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="low_brightness_diversity",
                    detail=f"Image brightness has low diversity (mean={mean_b:.1f}, std={std_b:.1f})",
                    suggestion="Add images with varied lighting conditions (day/night, indoor/outdoor).",
                ))

        return issues

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_images(directory: Path) -> int:
        """Count image files in a directory."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)

    @staticmethod
    def _count_from_list(list_file: Path) -> int:
        """Count entries in a text file listing paths."""
        if not list_file.exists():
            return 0
        with open(list_file, "r") as fh:
            return sum(1 for line in fh if line.strip())