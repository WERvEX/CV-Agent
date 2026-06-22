"""YOLO training wrapper for Ultralytics models.

Wraps ultralytics.YOLO.train() with hyperparameter injection,
artifact path management, and graceful error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from cv_agent.core.config import HyperParams
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class TrainArtifacts:
    """Container for paths to training output artifacts."""

    def __init__(
        self,
        best_pt: Path,
        last_pt: Path,
        args_yaml: Path,
        results_csv: Path,
        run_dir: Path,
    ) -> None:
        self.best_pt = best_pt
        self.last_pt = last_pt
        self.args_yaml = args_yaml
        self.results_csv = results_csv
        self.run_dir = run_dir

    def __repr__(self) -> str:
        return (
            f"TrainArtifacts(best_pt={self.best_pt}, last_pt={self.last_pt}, "
            f"results_csv={self.results_csv})"
        )


class YOLOTrainer:
    """Wraps Ultralytics YOLO training with cv_agent's parameter model."""

    def train(
        self,
        model_variant: str,
        data_yaml: Path,
        hyperparams: HyperParams,
        epochs: int,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> TrainArtifacts:
        """Run a single YOLO training session.

        Args:
            model_variant: e.g. "yolov8n", "yolov8s".
            data_yaml: Path to dataset YAML.
            hyperparams: HyperParams model instance.
            epochs: Number of epochs for this round.
            run_dir: Target directory for this experiment (runs/exp_xxx/).
            resume_from: Optional checkpoint to resume from.

        Returns:
            TrainArtifacts with paths to output files.

        Raises:
            RuntimeError: If training fails.
        """
        model_path = f"{model_variant}.pt"
        logger.info(f"Loading model: {model_path}")

        model = YOLO(model_path)

        # Build training arguments from HyperParams
        train_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "project": str(run_dir.parent),  # e.g., "runs"
            "name": run_dir.name,            # e.g., "exp_20260122_143052"
            "exist_ok": True,
            # Core hyperparams
            "lr0": hyperparams.lr0,
            "lrf": hyperparams.lrf,
            "batch": hyperparams.batch,
            "momentum": hyperparams.momentum,
            "weight_decay": hyperparams.weight_decay,
            "warmup_epochs": hyperparams.warmup_epochs,
            "warmup_momentum": hyperparams.warmup_momentum,
            "box": hyperparams.box,
            "cls": hyperparams.cls,
            "dfl": hyperparams.dfl,
            # Augmentation hyperparams
            "mosaic": hyperparams.mosaic,
            "mixup": hyperparams.mixup,
            "copy_paste": hyperparams.copy_paste,
            "hsv_h": hyperparams.hsv_h,
            "hsv_s": hyperparams.hsv_s,
            "hsv_v": hyperparams.hsv_v,
            "degrees": hyperparams.degrees,
            "translate": hyperparams.translate,
            "scale": hyperparams.scale,
            "shear": hyperparams.shear,
            "perspective": hyperparams.perspective,
            "flipud": hyperparams.flipud,
            "fliplr": hyperparams.fliplr,
        }

        if resume_from is not None and resume_from.exists():
            train_args["resume"] = True
            model = YOLO(str(resume_from))
            logger.info(f"Resuming from checkpoint: {resume_from}")

        logger.info(f"Starting training with {epochs} epochs, batch={hyperparams.batch}, lr0={hyperparams.lr0}")
        logger.info(f"Augmentations: mosaic={hyperparams.mosaic}, mixup={hyperparams.mixup}")

        try:
            results = model.train(**train_args)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"YOLO training failed: {e}") from e

        # Resolve artifact paths
        # Ultralytics saves to: project/name/weights/best.pt, etc.
        weights_dir = run_dir / "weights"

        artifacts = TrainArtifacts(
            best_pt=weights_dir / "best.pt",
            last_pt=weights_dir / "last.pt",
            args_yaml=run_dir / "args.yaml",
            results_csv=run_dir / "results.csv",
            run_dir=run_dir,
        )

        # Validate that expected outputs exist
        missing = []
        for attr_name in ("best_pt", "last_pt", "results_csv"):
            path = getattr(artifacts, attr_name)
            if not path.exists():
                missing.append(str(path))

        if missing:
            logger.warning(f"Some expected artifacts not found: {missing}")

        logger.info(f"Training round complete. Best weights: {artifacts.best_pt}")
        return artifacts

    def validate(
        self,
        weights_path: Path,
        data_yaml: Path,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Run standalone validation on a trained model.

        Args:
            weights_path: Path to model weights (.pt file).
            data_yaml: Path to dataset YAML.
            run_dir: Output directory for validation results.

        Returns:
            Validation metrics dict from Ultralytics.
        """
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_yaml),
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
        )
        return metrics.results_dict if hasattr(metrics, "results_dict") else {}