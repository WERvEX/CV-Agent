"""Three-state decision classifier: Green / Yellow / Red.

Classifies each training round based on metric comparison with historical
best, then determines the appropriate action and next hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cv_agent.core.config import DecisionConfig, HyperParams
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
    ) -> Decision:
        """Classify the round and produce a Decision."""
        if comparison.best_historical is None:
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason="First round — accepting as baseline.",
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={"green_tier": "hard"},
            )

        delta_pct = comparison.delta_percent
        delta_abs = comparison.delta_abs
        meta_base: dict[str, Any] = {
            "delta_percent": delta_pct,
            "delta_abs": delta_abs,
        }

        if self._is_hard_green(delta_pct, delta_abs):
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason=f"Score improved by {delta_pct:+.2f}% — committing checkpoint.",
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
                    f"(below {self.green_threshold}% threshold) — accepting checkpoint."
                ),
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={**meta_base, "green_tier": "marginal"},
            )

        if self._is_hard_red(delta_pct, delta_abs):
            return self._hard_red_decision(comparison, current_params, delta_pct, meta_base)

        if self._is_soft_red(delta_pct, delta_abs):
            return self._soft_red_decision(comparison, current_params, delta_pct, meta_base)

        return self._yellow_decision(comparison, current_params, delta_pct, meta_base)

    def _is_hard_green(self, delta_pct: float, delta_abs: float) -> bool:
        if delta_pct >= self.green_threshold:
            return True
        threshold_abs = self._cfg.green_threshold_abs
        return threshold_abs is not None and delta_abs >= threshold_abs

    def _is_marginal_green(self, delta_pct: float) -> bool:
        return self._cfg.accept_marginal_improvement and delta_pct > 0

    def _is_hard_red(self, delta_pct: float, delta_abs: float) -> bool:
        if delta_pct <= self.red_threshold:
            return True
        threshold_abs = self._cfg.red_threshold_abs
        return threshold_abs is not None and delta_abs <= threshold_abs

    def _is_soft_red(self, delta_pct: float, delta_abs: float) -> bool:
        if self._is_hard_red(delta_pct, delta_abs):
            return False
        return delta_pct <= self.soft_red_threshold

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
                    f"Hard RED — overfitting (train↓ val↑). Score dropped {delta_pct:+.2f}%. "
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
                    f"Hard RED — underfitting (loss plateau). Score dropped {delta_pct:+.2f}%. "
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
                f"Hard RED — score dropped {delta_pct:+.2f}%. "
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
    ) -> Decision:
        if comparison.overfitting:
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.MILD_REGULARIZE.value,
                reason=(
                    f"Soft RED — overfitting with score {delta_pct:+.2f}% "
                    f"(between {self.soft_red_threshold}% and {self.red_threshold}%). "
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
                    f"Soft RED — underfitting with score {delta_pct:+.2f}%. "
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
                f"Soft RED — score {delta_pct:+.2f}% in [{self.soft_red_threshold}%, "
                f"{self.red_threshold}%]. Mild regularization without rollback."
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
    ) -> Decision:
        band_lo = self.soft_red_threshold
        band_hi = self.green_threshold
        if comparison.overfitting:
            return Decision(
                color=DecisionColor.YELLOW.value,
                action=DecisionAction.MILD_REGULARIZE.value,
                reason=(
                    f"YELLOW — overfitting detected, score {delta_pct:+.2f}% "
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
                    f"YELLOW — underfitting detected, score {delta_pct:+.2f}% "
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
                f"YELLOW — score change {delta_pct:+.2f}% in band "
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
