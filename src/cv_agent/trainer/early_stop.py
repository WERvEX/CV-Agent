"""Early-stop metric resolution for closed-loop training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cv_agent.core.config import EarlyStopConfig
from cv_agent.trainer.evaluator import Evaluator, RoundResult


@dataclass
class EarlyStopDecision:
    """Result of evaluating the configured early-stop target."""

    reached: bool
    metric: str
    metric_value: float | None
    target: float
    reason: str = ""


_METRIC_ALIASES = {
    "map50": "mAP50",
    "map50_95": "mAP50_95",
    "map50-95": "mAP50_95",
    "precision": "precision",
    "recall": "recall",
}


def resolve_early_stop_metric(
    metric: str,
    result: RoundResult,
    data_yaml: Path,
) -> tuple[str, float | None]:
    """Resolve an early-stop metric name to the canonical key and value."""
    raw_metric = metric.strip()
    normalized = raw_metric.lower()
    if normalized == "score":
        return "score", float(result.score)

    if normalized.startswith("map50_class:"):
        class_ref = raw_metric.split(":", 1)[1].strip()
        class_id = _resolve_class_ref(class_ref, data_yaml)
        key = f"mAP50_class_{class_id}" if class_id is not None else f"mAP50_class:{class_ref}"
        value = result.metrics.get(key)
        return key, float(value) if value is not None else None

    key = _METRIC_ALIASES.get(normalized, raw_metric)
    value = result.metrics.get(key)
    return key, float(value) if value is not None else None


def evaluate_early_stop(
    config: EarlyStopConfig,
    result: RoundResult,
    data_yaml: Path,
) -> EarlyStopDecision:
    """Return whether this round reached the configured early-stop target."""
    metric, value = resolve_early_stop_metric(config.metric, result, data_yaml)
    if value is None:
        return EarlyStopDecision(
            reached=False,
            metric=metric,
            metric_value=None,
            target=config.target,
            reason=f"metric '{config.metric}' was not available",
        )
    reached = value >= config.target
    reason = (
        f"{metric}={value:.4f} reached target {config.target:.4f}"
        if reached
        else f"{metric}={value:.4f} below target {config.target:.4f}"
    )
    return EarlyStopDecision(
        reached=reached,
        metric=metric,
        metric_value=value,
        target=config.target,
        reason=reason,
    )


def _resolve_class_ref(class_ref: str, data_yaml: Path) -> int | None:
    if class_ref.isdigit():
        return int(class_ref)
    try:
        names = Evaluator.load_class_names(data_yaml)
    except Exception:
        return None
    for class_id, class_name in names.items():
        if str(class_name).lower() == class_ref.lower():
            return class_id
    return None
