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

# Module-level defaults (used when no DecisionConfig is provided)
GREEN_THRESHOLD = 3.0
RED_THRESHOLD = -5.0
RED_ESCALATION_COUNT = 3


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
        self.green_threshold = cfg.green_threshold_pct
        self.red_threshold = cfg.red_threshold_pct
        self.red_escalation_count = cfg.red_escalation_count

    def decide(
        self,
        comparison: EvaluationComparison,
        red_count: int,
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
            )

        delta = comparison.delta_percent

        if delta >= self.green_threshold:
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason=f"mAP improved by {delta:+.2f}% — committing checkpoint.",
                next_hyperparams=current_params,
                should_rollback=False,
                metadata={"delta_percent": delta},
            )

        if delta <= self.red_threshold:
            if comparison.overfitting:
                return Decision(
                    color=DecisionColor.RED.value,
                    action=DecisionAction.ROLLBACK_REGULARIZE.value,
                    reason=(
                        f"Overfitting detected (train↓ val↑). Score dropped {delta:+.2f}%. "
                        "Rolling back to best checkpoint and increasing regularization."
                    ),
                    next_hyperparams=self._regularize_params(current_params),
                    should_rollback=True,
                    metadata={"delta_percent": delta, "diagnosis": "overfitting"},
                )
            if comparison.underfitting:
                return Decision(
                    color=DecisionColor.RED.value,
                    action=DecisionAction.AGGRESSIVE_LR_ADJUST.value,
                    reason=(
                        f"Underfitting detected (both losses plateau). Score dropped {delta:+.2f}%. "
                        "Aggressively adjusting learning rate."
                    ),
                    next_hyperparams=self._aggressive_lr_params(current_params),
                    should_rollback=False,
                    metadata={"delta_percent": delta, "diagnosis": "underfitting"},
                )
            return Decision(
                color=DecisionColor.RED.value,
                action=DecisionAction.ROLLBACK.value,
                reason=(
                    f"Score dropped {delta:+.2f}% — rolling back to best checkpoint "
                    "and applying random perturbation."
                ),
                next_hyperparams=self._perturb_params(current_params),
                should_rollback=True,
                metadata={"delta_percent": delta, "diagnosis": "general"},
            )

        return Decision(
            color=DecisionColor.YELLOW.value,
            action=DecisionAction.ESCAPE_LOCAL_OPTIMUM.value,
            reason=(
                f"Score change {delta:+.2f}% within oscillation band "
                f"(±{self.green_threshold}%). Triggering local optimum escape."
            ),
            next_hyperparams=current_params,
            should_rollback=False,
            metadata={"delta_percent": delta},
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
