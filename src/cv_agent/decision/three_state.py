"""Three-state decision classifier: Green / Yellow / Red.

Classifies each training round based on metric comparison with historical
best, then determines the appropriate action and next hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cv_agent.core.config import DecisionConfig, DecisionPhaseThresholds, HyperParams
from cv_agent.core.state_machine import DecisionAction, DecisionColor
from cv_agent.trainer.evaluator import EvaluationComparison
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Decision:
    """Output of the three-state classifier for one training round."""

    color: str
    action: str
    reason: str
    next_hyperparams: HyperParams = field(default_factory=HyperParams)
    proposed_hyperparams: HyperParams | None = None
    should_rollback: bool = False
    rollback_checkpoint: str | None = None
    data_gap_report_needed: bool = False
    llm_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        display_params = self.proposed_hyperparams or self.next_hyperparams
        return {
            "color": self.color,
            "action": self.action,
            "reason": self.reason,
            "next_hyperparams": display_params.model_dump(),
            "proposed_hyperparams": (
                self.proposed_hyperparams.model_dump() if self.proposed_hyperparams else None
            ),
            "should_rollback": self.should_rollback,
            "rollback_checkpoint": self.rollback_checkpoint,
            "data_gap_report_needed": self.data_gap_report_needed,
            "llm_context": self.llm_context,
            "metadata": self.metadata,
        }


class ThreeStateDecisionEngine:
    """Classifies each training round and determines next actions."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        cfg = config or DecisionConfig()
        self._cfg = cfg
        self.green_threshold = cfg.green_threshold_pct
        self.red_threshold = cfg.red_threshold_pct
        self.soft_red_threshold = cfg.soft_red_threshold_pct

    def decide(
        self,
        comparison: EvaluationComparison,
        current_params: HyperParams,
        round_num: int | None = None,
        max_rounds: int | None = None,
        history: list[Any] | None = None,
    ) -> Decision:
        """Classify the round and produce a Decision."""
        if comparison.best_historical is None:
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason="First round 鈥?accepting as baseline.",
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={"green_tier": "hard"},
            )

        delta_pct = comparison.delta_percent
        delta_abs = comparison.delta_abs
        threshold_ctx = self._effective_threshold_context(
            round_num=round_num,
            max_rounds=max_rounds,
            history=history or [],
            current_score=comparison.current.score,
        )
        thresholds: DecisionPhaseThresholds = threshold_ctx["thresholds"]
        meta_base: dict[str, Any] = {
            "delta_percent": delta_pct,
            "delta_abs": delta_abs,
            "decision_phase": threshold_ctx["phase"],
            "effective_thresholds": threshold_ctx["threshold_values"],
            "recent_median_score": threshold_ctx["recent_median_score"],
            "delta_vs_recent_median_pct": threshold_ctx["delta_vs_recent_median_pct"],
            "recent_volatility": threshold_ctx["recent_volatility"],
            "volatility_relaxed": threshold_ctx["volatility_relaxed"],
        }

        if self._is_hard_green(delta_pct, delta_abs, thresholds):
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason=f"Score improved by {delta_pct:+.2f}% 鈥?committing checkpoint.",
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={**meta_base, "green_tier": "hard"},
            )

        if self._is_marginal_green(delta_pct):
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason=(
                    f"Marginal improvement {delta_pct:+.2f}% "
                    f"(below {thresholds.green_threshold_pct}% threshold) - accepting checkpoint."
                ),
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={**meta_base, "green_tier": "marginal"},
            )

        if self._is_hard_red(delta_pct, delta_abs, thresholds):
            if self._recent_median_guard_applies(threshold_ctx):
                return self._yellow_decision(
                    comparison,
                    current_params,
                    delta_pct,
                    {**meta_base, "recent_median_guard": True},
                    thresholds,
                )
            return self._hard_red_decision(comparison, current_params, delta_pct, meta_base)

        if self._is_soft_red(delta_pct, delta_abs, thresholds):
            return self._soft_red_decision(comparison, current_params, delta_pct, meta_base, thresholds)

        return self._yellow_decision(comparison, current_params, delta_pct, meta_base, thresholds)

    def _effective_threshold_context(
        self,
        *,
        round_num: int | None,
        max_rounds: int | None,
        history: list[Any],
        current_score: float,
    ) -> dict[str, Any]:
        phase = self._decision_phase(round_num, max_rounds)
        thresholds = self._phase_thresholds(phase)
        recent_scores = [
            float(item.score)
            for item in history[-self._cfg.recent_window:]
            if hasattr(item, "score")
        ]
        recent_median = self._median(recent_scores) if recent_scores else None
        delta_vs_recent_median = (
            ((current_score - recent_median) / recent_median) * 100.0
            if recent_median and recent_median > 0
            else None
        )
        recent_volatility = self._stddev(recent_scores) if len(recent_scores) >= 2 else 0.0
        volatility_relaxed = (
            self._cfg.dynamic_thresholds
            and self._cfg.volatility_relaxation_enabled
            and recent_volatility >= self._cfg.high_volatility_abs
        )
        if volatility_relaxed:
            thresholds = DecisionPhaseThresholds(
                green_threshold_pct=thresholds.green_threshold_pct,
                green_threshold_abs=thresholds.green_threshold_abs,
                soft_red_threshold_pct=thresholds.soft_red_threshold_pct,
                red_threshold_pct=thresholds.red_threshold_pct - self._cfg.high_volatility_red_relax_pct,
                red_threshold_abs=(
                    thresholds.red_threshold_abs - self._cfg.high_volatility_abs
                    if thresholds.red_threshold_abs is not None
                    else None
                ),
            )
        return {
            "phase": phase,
            "thresholds": thresholds,
            "threshold_values": thresholds.model_dump(),
            "recent_median_score": recent_median,
            "delta_vs_recent_median_pct": delta_vs_recent_median,
            "recent_volatility": recent_volatility,
            "volatility_relaxed": volatility_relaxed,
        }

    def _decision_phase(self, round_num: int | None, max_rounds: int | None) -> str:
        if not self._cfg.dynamic_thresholds or not round_num or not max_rounds or max_rounds <= 0:
            return "static"
        progress = round_num / max_rounds
        schedule = self._cfg.phase_schedule
        if progress <= schedule.exploration_until_pct:
            return "exploration"
        if progress <= schedule.exploitation_until_pct:
            return "exploitation"
        return "convergence"

    def _phase_thresholds(self, phase: str) -> DecisionPhaseThresholds:
        if not self._cfg.dynamic_thresholds or phase == "static":
            return DecisionPhaseThresholds(
                green_threshold_pct=self._cfg.green_threshold_pct,
                green_threshold_abs=self._cfg.green_threshold_abs,
                soft_red_threshold_pct=self._cfg.soft_red_threshold_pct,
                red_threshold_pct=self._cfg.red_threshold_pct,
                red_threshold_abs=self._cfg.red_threshold_abs,
            )
        return getattr(self._cfg.dynamic, phase)

    @staticmethod
    def _median(values: list[float]) -> float:
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return sorted_values[mid]
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

    @staticmethod
    def _stddev(values: list[float]) -> float:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance ** 0.5

    def _recent_median_guard_applies(self, threshold_ctx: dict[str, Any]) -> bool:
        return (
            self._cfg.dynamic_thresholds
            and self._cfg.use_recent_median
            and threshold_ctx["delta_vs_recent_median_pct"] is not None
            and threshold_ctx["delta_vs_recent_median_pct"] >= 0
        )

    def _is_hard_green(
        self,
        delta_pct: float,
        delta_abs: float,
        thresholds: DecisionPhaseThresholds,
    ) -> bool:
        if delta_pct >= thresholds.green_threshold_pct:
            return True
        threshold_abs = thresholds.green_threshold_abs
        return threshold_abs is not None and delta_abs >= threshold_abs

    def _is_marginal_green(self, delta_pct: float) -> bool:
        return self._cfg.accept_marginal_improvement and delta_pct > 0

    def _is_hard_red(
        self,
        delta_pct: float,
        delta_abs: float,
        thresholds: DecisionPhaseThresholds,
    ) -> bool:
        if delta_pct <= thresholds.red_threshold_pct:
            return True
        threshold_abs = thresholds.red_threshold_abs
        return threshold_abs is not None and delta_abs <= threshold_abs

    def _is_soft_red(
        self,
        delta_pct: float,
        delta_abs: float,
        thresholds: DecisionPhaseThresholds,
    ) -> bool:
        if self._is_hard_red(delta_pct, delta_abs, thresholds):
            return False
        return delta_pct <= thresholds.soft_red_threshold_pct

    def _hard_red_decision(
        self,
        comparison: EvaluationComparison,
        current_params: HyperParams,
        delta_pct: float,
        meta_base: dict[str, Any],
    ) -> Decision:
        if comparison.overfitting:
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.ROLLBACK_REGULARIZE.value,
                reason=(
                    f"Hard RED 鈥?overfitting (train鈫?val鈫?. Score dropped {delta_pct:+.2f}%. "
                    "Rolling back and increasing regularization."
                ),
                next_hyperparams=self._regularize_params(current_params),
                should_rollback=True,
                metadata={**meta_base, "red_tier": "hard", "diagnosis": "overfitting"},
            )
        if comparison.underfitting:
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.AGGRESSIVE_LR_ADJUST.value,
                reason=(
                    f"Hard RED 鈥?underfitting (loss plateau). Score dropped {delta_pct:+.2f}%. "
                    "Rolling back and aggressively adjusting learning rate."
                ),
                next_hyperparams=self._aggressive_lr_params(current_params),
                should_rollback=True,
                metadata={**meta_base, "red_tier": "hard", "diagnosis": "underfitting"},
            )
        return Decision(
            color=DecisionColor.RED.value,
            action=DecisionAction.ROLLBACK.value,
            reason=(
                f"Hard RED 鈥?score dropped {delta_pct:+.2f}%. "
                "Rolling back to best checkpoint and applying perturbation."
            ),
            next_hyperparams=self._perturb_params(current_params),
            should_rollback=True,
            metadata={**meta_base, "red_tier": "hard", "diagnosis": "general"},
        )

    def _soft_red_decision(
        self,
        comparison: EvaluationComparison,
        current_params: HyperParams,
        delta_pct: float,
        meta_base: dict[str, Any],
        thresholds: DecisionPhaseThresholds,
    ) -> Decision:
        if comparison.overfitting:
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.MILD_REGULARIZE.value,
                reason=(
                    f"Soft RED 鈥?overfitting with score {delta_pct:+.2f}% "
                    f"(between {thresholds.soft_red_threshold_pct}% and {thresholds.red_threshold_pct}%). "
                    "Applying mild regularization without rollback."
                ),
                next_hyperparams=self._mild_regularize_params(current_params),
                should_rollback=False,
                metadata={**meta_base, "red_tier": "soft", "diagnosis": "overfitting"},
            )
        if comparison.underfitting:
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.MILD_LR_ADJUST.value,
                reason=(
                    f"Soft RED 鈥?underfitting with score {delta_pct:+.2f}%. "
                    "Mildly increasing learning rate without rollback."
                ),
                next_hyperparams=self._mild_lr_params(current_params),
                should_rollback=False,
                metadata={**meta_base, "red_tier": "soft", "diagnosis": "underfitting"},
            )
        return Decision(
            color=DecisionColor.RED.value,
            action=DecisionAction.MILD_REGULARIZE.value,
            reason=(
                f"Soft RED - score {delta_pct:+.2f}% in [{thresholds.soft_red_threshold_pct}%, "
                f"{thresholds.red_threshold_pct}%]. Mild regularization without rollback."
            ),
            next_hyperparams=self._mild_regularize_params(current_params),
            should_rollback=False,
            metadata={**meta_base, "red_tier": "soft", "diagnosis": "general"},
        )

    def _yellow_decision(
        self,
        comparison: EvaluationComparison,
        current_params: HyperParams,
        delta_pct: float,
        meta_base: dict[str, Any],
        thresholds: DecisionPhaseThresholds,
    ) -> Decision:
        band_lo = thresholds.soft_red_threshold_pct
        band_hi = thresholds.green_threshold_pct
        if comparison.overfitting:
            return Decision(
                color=DecisionColor.YELLOW.value,
                action=DecisionAction.MILD_REGULARIZE.value,
                reason=(
                    f"YELLOW 鈥?overfitting detected, score {delta_pct:+.2f}% "
                    f"in band [{band_lo}%, {band_hi}%). Mild regularization."
                ),
                next_hyperparams=self._mild_regularize_params(current_params),
                should_rollback=False,
                metadata={**meta_base, "diagnosis": "overfitting"},
            )
        if comparison.underfitting:
            return Decision(
                color=DecisionColor.YELLOW.value,
                action=DecisionAction.MILD_LR_ADJUST.value,
                reason=(
                    f"YELLOW 鈥?underfitting detected, score {delta_pct:+.2f}% "
                    f"in band [{band_lo}%, {band_hi}%). Mild LR adjustment."
                ),
                next_hyperparams=self._mild_lr_params(current_params),
                should_rollback=False,
                metadata={**meta_base, "diagnosis": "underfitting"},
            )
        return Decision(
            color=DecisionColor.YELLOW.value,
            action=DecisionAction.ESCAPE_LOCAL_OPTIMUM.value,
            reason=(
                f"YELLOW 鈥?score change {delta_pct:+.2f}% in band "
                f"[{band_lo}%, {band_hi}%). Triggering local optimum escape."
            ),
            next_hyperparams=current_params,
            should_rollback=False,
            metadata=meta_base,
        )

    @staticmethod
    def _regularize_params(params: HyperParams) -> HyperParams:
        return HyperParams(
            **{
                **params.model_dump(),
                "weight_decay": min(params.weight_decay * 2.0, 0.01),
                "mosaic": max(params.mosaic * 0.5, 0.0),
                "mixup": max(params.mixup * 0.5, 0.0),
                "copy_paste": max(params.copy_paste * 0.5, 0.0),
                "degrees": max(params.degrees * 0.5, 0.0),
            }
        )

    @staticmethod
    def _mild_regularize_params(params: HyperParams) -> HyperParams:
        return HyperParams(
            **{
                **params.model_dump(),
                "weight_decay": min(params.weight_decay * 1.5, 0.01),
                "mosaic": max(params.mosaic * 0.7, 0.0),
                "mixup": max(params.mixup * 0.7, 0.0),
                "copy_paste": max(params.copy_paste * 0.7, 0.0),
                "degrees": max(params.degrees * 0.7, 0.0),
            }
        )

    @staticmethod
    def _mild_lr_params(params: HyperParams) -> HyperParams:
        return HyperParams(
            **{
                **params.model_dump(),
                "lr0": min(params.lr0 * 1.5, 0.5),
                "lrf": min(params.lrf * 1.5, 0.5),
            }
        )

    @staticmethod
    def _aggressive_lr_params(params: HyperParams) -> HyperParams:
        import random

        multiplier = random.uniform(2.0, 5.0)
        return HyperParams(
            **{
                **params.model_dump(),
                "lr0": min(params.lr0 * multiplier, 0.5),
                "lrf": min(params.lrf * multiplier, 0.5),
            }
        )

    @staticmethod
    def _perturb_params(params: HyperParams) -> HyperParams:
        import random

        perturbed = params.model_dump()
        perturbed["lr0"] = params.lr0 * random.uniform(0.7, 1.3)
        perturbed["weight_decay"] = params.weight_decay * random.uniform(0.5, 1.5)
        perturbed["mosaic"] = min(max(params.mosaic + random.uniform(-0.15, 0.15), 0.0), 1.0)
        return HyperParams(**perturbed)
