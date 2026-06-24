"""Metrics extraction, reward scoring, and round comparison.

Reads Ultralytics results.csv, computes per-class and global mAP,
detects overfitting/underfitting, and compares against historical best.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Metrics from a single training round."""

    round_num: int
    run_dir: Path
    metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    train_loss_final: float | None = None
    val_loss_final: float | None = None
    overfitting: bool = False
    underfitting: bool = False
    score: float = 0.0


@dataclass
class EvaluationComparison:
    """Comparison between current round and historical best."""

    current: RoundResult
    best_historical: RoundResult | None
    delta_percent: float = 0.0
    overfitting: bool = False
    underfitting: bool = False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Extracts metrics from YOLO training output and computes comparisons."""

    # Thresholds for overfitting / underfitting detection
    OVERFIT_TRAIN_DECREASING_PCT: float = 0.05   # 5% train loss decrease
    OVERFIT_VAL_INCREASING_PCT: float = 0.05      # 5% val loss increase
    UNDERFIT_LOSS_PLATEAU: float = 0.02            # <2% change in both losses

    def __init__(self, optimize_for_class_id: int | None = None) -> None:
        """Initialize the evaluator.

        Args:
            optimize_for_class_id: If set, reward scoring weights this class's
                                   mAP@0.5 3× higher than other classes.
        """
        self.optimize_for_class_id = optimize_for_class_id

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    def extract_metrics(
        self,
        results_csv: Path,
        data_yaml: Path,
        round_num: int,
        run_dir: Path,
    ) -> RoundResult:
        """Extract metrics from a YOLO results.csv file.

        Args:
            results_csv: Path to results.csv from Ultralytics training.
            data_yaml: Path to dataset YAML (for class name mapping).
            round_num: Current round number.
            run_dir: Experiment directory.

        Returns:
            RoundResult with extracted metrics.
        """
        result = RoundResult(round_num=round_num, run_dir=run_dir)

        if not results_csv.exists():
            logger.warning(f"results.csv not found at {results_csv}")
            return result

        df = pd.read_csv(results_csv)
        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # --- Global metrics from final epoch ---
        last_row = df.iloc[-1]

        result.metrics["mAP50"] = self._safe_float(last_row, "metrics/mAP50(B)")
        result.metrics["mAP50_95"] = self._safe_float(last_row, "metrics/mAP50-95(B)")
        result.metrics["precision"] = self._safe_float(last_row, "metrics/precision(B)")
        result.metrics["recall"] = self._safe_float(last_row, "metrics/recall(B)")

        # --- Loss values ---
        result.train_loss_final = self._safe_float(last_row, "train/box_loss") + \
                                  self._safe_float(last_row, "train/cls_loss") + \
                                  self._safe_float(last_row, "train/dfl_loss")

        result.val_loss_final = self._safe_float(last_row, "val/box_loss") + \
                                self._safe_float(last_row, "val/cls_loss") + \
                                self._safe_float(last_row, "val/dfl_loss")

        # --- Detect overfitting / underfitting ---
        if len(df) >= 3:
            result.overfitting = self._detect_overfitting(df)
            result.underfitting = self._detect_underfitting(df)

        # --- Compute reward score (CSV-only; may be updated after val enrichment) ---
        result.score = self.compute_reward(result)

        logger.info(
            f"Round {round_num} metrics: mAP50={result.metrics['mAP50']:.4f}, "
            f"mAP50-95={result.metrics['mAP50_95']:.4f}, "
            f"score={result.score:.4f}, "
            f"overfit={result.overfitting}, underfit={result.underfitting}"
        )

        return result

    def enrich_from_validation(
        self,
        result: RoundResult,
        weights_path: Path,
        data_yaml: Path,
    ) -> RoundResult:
        """Run model.val() to attach per-class metrics and a confusion matrix.

        Updates ``result.metrics``, ``per_class_metrics``, ``confusion_matrix``,
        and recomputes ``score`` (needed for ``--optimize-for`` weighting).
        """
        if not weights_path.exists():
            logger.warning(f"Cannot run validation enrichment — weights missing: {weights_path}")
            return result

        try:
            from ultralytics import YOLO

            logger.info(f"Running validation enrichment on {weights_path.name} ...")
            val_result = YOLO(str(weights_path)).val(
                data=str(data_yaml),
                plots=False,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Validation enrichment failed: {e}")
            return result

        box = getattr(val_result, "box", None)
        if box is not None:
            self._merge_per_class_array(result, "mAP50", getattr(box, "ap50", None))
            self._merge_per_class_array(result, "precision", getattr(box, "p", None))
            self._merge_per_class_array(result, "recall", getattr(box, "r", None))

            map50 = getattr(box, "map50", None)
            if map50 is not None:
                result.metrics["mAP50"] = float(map50)

        cm = getattr(val_result, "confusion_matrix", None)
        if cm is not None and getattr(cm, "matrix", None) is not None:
            result.confusion_matrix = np.asarray(cm.matrix)

        result.score = self.compute_reward(result)
        logger.info(
            f"Validation enrichment complete: score={result.score:.4f}, "
            f"per_class_keys={sum(1 for k in result.metrics if k.startswith('mAP50_class_'))}"
        )
        return result

    @staticmethod
    def _merge_per_class_array(
        result: RoundResult,
        metric_name: str,
        values: np.ndarray | list | None,
    ) -> None:
        """Store per-class metric values on ``result`` using cv_agent key conventions."""
        if values is None:
            return
        arr = np.asarray(values).flatten()
        for class_id, value in enumerate(arr):
            if np.isnan(value):
                continue
            key = f"{metric_name}_class_{class_id}"
            result.metrics[key] = float(value)
            bucket = result.per_class_metrics.setdefault(class_id, {})
            bucket[metric_name] = float(value)

    # ------------------------------------------------------------------
    # Overfitting / underfitting detection
    # ------------------------------------------------------------------

    def _detect_overfitting(self, df: pd.DataFrame) -> bool:
        """Detect overfitting: train loss decreasing while val loss increasing."""
        train_start = self._sum_losses(df.iloc[0], "train")
        train_end = self._sum_losses(df.iloc[-1], "train")
        val_start = self._sum_losses(df.iloc[0], "val")
        val_end = self._sum_losses(df.iloc[-1], "val")

        if train_start == 0 or val_start == 0:
            return False

        train_delta = (train_start - train_end) / train_start
        val_delta = (val_end - val_start) / val_start

        return train_delta > self.OVERFIT_TRAIN_DECREASING_PCT and val_delta > self.OVERFIT_VAL_INCREASING_PCT

    def _detect_underfitting(self, df: pd.DataFrame) -> bool:
        """Detect underfitting: both losses plateau (minimal change)."""
        train_start = self._sum_losses(df.iloc[0], "train")
        train_end = self._sum_losses(df.iloc[-1], "train")
        val_start = self._sum_losses(df.iloc[0], "val")
        val_end = self._sum_losses(df.iloc[-1], "val")

        if train_start == 0 or val_start == 0:
            return False

        train_delta = abs(train_start - train_end) / train_start
        val_delta = abs(val_start - val_end) / val_start

        return train_delta < self.UNDERFIT_LOSS_PLATEAU and val_delta < self.UNDERFIT_LOSS_PLATEAU

    @staticmethod
    def _sum_losses(row: pd.Series, prefix: str) -> float:
        """Sum box_loss + cls_loss + dfl_loss for a given prefix (train/val)."""
        total = 0.0
        for loss_type in ("box_loss", "cls_loss", "dfl_loss"):
            col = f"{prefix}/{loss_type}"
            val = row.get(col, 0.0)
            total += float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0
        return total

    # ------------------------------------------------------------------
    # Reward scoring
    # ------------------------------------------------------------------

    def compute_reward(self, result: RoundResult) -> float:
        """Compute the reward score from metrics.

        When optimize_for_class_id is set, the target class's mAP@0.5
        gets 3× the weight of other metrics.

        Args:
            result: RoundResult with populated metrics.

        Returns:
            Reward score (higher = better).
        """
        base_score = result.metrics.get("mAP50", 0.0)

        if self.optimize_for_class_id is not None:
            target_key = f"mAP50_class_{self.optimize_for_class_id}"
            target_map50 = result.metrics.get(target_key, 0.0)
            # Weighted: 30% global + 170% target = ~3× effective weight on target
            return 0.3 * base_score + 1.7 * target_map50

        return base_score

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        current: RoundResult,
        history: list[RoundResult],
    ) -> EvaluationComparison:
        """Compare current round against historical best.

        Args:
            current: This round's metrics.
            history: All previous round results.

        Returns:
            EvaluationComparison with deltas and diagnostic flags.
        """
        # Find best historical round by score
        best = max(history, key=lambda r: r.score) if history else None

        if best is None:
            return EvaluationComparison(
                current=current,
                best_historical=None,
                delta_percent=0.0,
                overfitting=current.overfitting,
                underfitting=current.underfitting,
            )

        if best.score > 0:
            delta_pct = ((current.score - best.score) / best.score) * 100.0
        else:
            delta_pct = 0.0 if current.score == 0 else 100.0

        return EvaluationComparison(
            current=current,
            best_historical=best,
            delta_percent=delta_pct,
            overfitting=current.overfitting,
            underfitting=current.underfitting,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(series: pd.Series, col: str) -> float:
        """Safely extract a float value from a DataFrame row."""
        if col not in series.index:
            return 0.0
        val = series[col]
        if isinstance(val, float) and np.isnan(val):
            return 0.0
        return float(val)

    @staticmethod
    def load_class_names(data_yaml: Path) -> dict[int, str]:
        """Load class id -> name mapping from a YOLO dataset YAML.

        Handles both formats:
            names: [list]
            names: {dict}

        Returns:
            Dict mapping class_id (int) to class_name (str).
        """
        with open(data_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        names = data.get("names", {})
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        if isinstance(names, dict):
            # Keys might be int or str
            return {int(k): v for k, v in names.items()}
        return {}

    @staticmethod
    def resolve_class_id(class_name: str, data_yaml: Path) -> int | None:
        """Resolve a class name to its numeric ID from the dataset YAML.

        Args:
            class_name: Human-readable class name (e.g., "vehicle").
            data_yaml: Path to the dataset YAML.

        Returns:
            Integer class ID, or None if not found.
        """
        names = Evaluator.load_class_names(data_yaml)
        for cid, cname in names.items():
            if cname.lower() == class_name.lower():
                return cid
        logger.warning(f"Class '{class_name}' not found in {data_yaml}. Available: {list(names.values())}")
        return None