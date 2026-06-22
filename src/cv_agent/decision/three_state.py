"""Three-state decision classifier: Green / Yellow / Red.

Classifies each training round based on metric comparison with historical
best, then determines the appropriate action and next hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cv_agent.core.config import HyperParams
from cv_agent.core.state_machine import DecisionAction, DecisionColor
from cv_agent.trainer.evaluator import EvaluationComparison
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

GREEN_THRESHOLD = 3.0    # +3% relative improvement = Green
RED_THRESHOLD = -5.0      # -5% relative degradation = Red
RED_ESCALATION_COUNT = 3   # 3 consecutive Reds = force data gap research


# ---------------------------------------------------------------------------
# Decision data model
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """Output of the three-state classifier for one training round."""

    color: str                                    # "green" | "yellow" | "red"
    action: str                                   # one of DecisionAction values
    reason: str                                   # human-readable explanation
    next_hyperparams: HyperParams = field(default_factory=HyperParams)
    should_rollback: bool = False
    rollback_checkpoint: str | None = None        # path to best.pt to restore
    data_gap_report_needed: bool = False
    llm_context: str | None = None                # user NL input from ask mode
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "action": self.action,
            "reason": self.reason,
            "next_hyperparams": self.next_hyperparams.model_dump(),
            "should_rollback": self.should_rollback,
            "rollback_checkpoint": self.rollback_checkpoint,
            "data_gap_report_needed": self.data_gap_report_needed,
            "llm_context": self.llm_context,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


class ThreeStateDecisionEngine:
    """Classifies each training round and determines next actions."""

    def decide(
        self,
        comparison: EvaluationComparison,
        red_count: int,
        current_params: HyperParams,
    ) -> Decision:
        """Classify the round and produce a Decision.

        Args:
            comparison: EvaluationComparison from the evaluator.
            red_count: Current consecutive Red count.
            current_params: The HyperParams used for this round.

        Returns:
            Decision with color, action, reason, and next hyperparameters.
        """
        # First round: always Green (no baseline to compare)
        if comparison.best_historical is None:
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason="First round — accepting as baseline.",
                next_hyperparams=current_params,
                should_rollback=False,
            )

        delta = comparison.delta_percent

        # ---- GREEN: significant improvement ----
        if delta >= GREEN_THRESHOLD:
            return Decision(
                color=DecisionColor.GREEN.value,
                action=DecisionAction.ACCEPT.value,
                reason=f"mAP improved by {delta:+.2f}% — committing checkpoint.",
                next_hyperparams=current_params,  # Optuna will mutate
                should_rollback=False,
                metadata={"delta_percent": delta},
            )

        # ---- RED: significant degradation ----
        if delta <= RED_THRESHOLD:
            # Diagnose overfitting vs underfitting
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
            elif comparison.underfitting:
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
            else:
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

        # ---- YELLOW: oscillation within thresholds ----
        return Decision(
            color=DecisionColor.YELLOW.value,
            action=DecisionAction.ESCAPE_LOCAL_OPTIMUM.value,
            reason=(
                f"Score change {delta:+.2f}% within oscillation band (±{GREEN_THRESHOLD}%). "
                "Triggering local optimum escape."
            ),
            next_hyperparams=current_params,  # Optuna random walk / SA will handle
            should_rollback=False,
            metadata={"delta_percent": delta},
        )

    # ------------------------------------------------------------------
    # Parameter mutation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regularize_params(params: HyperParams) -> HyperParams:
        """Increase regularization: reduce augmentation, increase weight_decay."""
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
        """Aggressively increase learning rate to escape underfitting plateau."""
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
        """Apply small random perturbations to break out of local minima."""
        import random

        perturbed = params.model_dump()
        perturbed["lr0"] = params.lr0 * random.uniform(0.7, 1.3)
        perturbed["weight_decay"] = params.weight_decay * random.uniform(0.5, 1.5)
        perturbed["mosaic"] = min(max(params.mosaic + random.uniform(-0.15, 0.15), 0.0), 1.0)
        return HyperParams(**perturbed)