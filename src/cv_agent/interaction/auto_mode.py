"""Auto mode interaction handler — print-only, no blocking.

All confirmations return True. All decisions are logged with Rich-styled
color-coded output. Writes decision_log.json to run directory.
"""

from __future__ import annotations

from typing import Any

from cv_agent.interaction.types import ConfigChangeReview, DecisionReview
from cv_agent.ui.console import log_decision, log_info, print_decision_recommendation
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AutoModeHandler:
    """Interaction handler for fully autonomous Auto mode."""

    def propagate_decision(self, decision: dict[str, Any], round_num: int) -> None:
        """Log a decision with color-coded Rich output."""
        color = decision.get("color", "white")
        action = decision.get("action", "?")
        reason = decision.get("reason", "")

        log_decision(color, f"Round {round_num}: [{action}] {reason}")
        print_decision_recommendation(decision, round_num)

    def review_decision(
        self,
        decision: dict[str, Any],
        round_num: int,
        current_params: dict[str, Any] | None,
        checkpoint_path: str | None,
    ) -> DecisionReview:
        """Auto-approve the controller recommendation without blocking."""
        self.propagate_decision(decision, round_num)
        should_rollback = bool(decision.get("should_rollback"))
        log_info(
            f"[auto] Decision auto-approved "
            f"(rollback={'yes' if should_rollback else 'no'})."
        )
        return DecisionReview(
            apply_recommendation=True,
            rollback_approved=should_rollback,
        )

    def confirm(self, message: str, default: bool = True) -> bool:
        """Auto-approve all confirmations."""
        log_info(f"[auto] {message} → Auto-approved (Y)")
        return True

    def text(self, message: str, default: str = "") -> str:
        """Return default text without blocking."""
        log_info(f"[auto] {message} → Using default: '{default}'")
        return default

    def press_enter(self, message: str = "Press Enter to continue...") -> None:
        """Skip wait in auto mode."""
        log_info(f"[auto] Skipping wait: {message}")

    def confirm_config_change(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        context: str = "",
    ) -> ConfigChangeReview:
        """Auto-approve config changes."""
        log_info(f"[auto] Config change auto-approved. {context}")
        return ConfigChangeReview(approved=True)

    def confirm_supplement(self, issues: list[dict[str, Any]]) -> bool:
        """Auto mode cannot self-heal data — caller treats False as abort."""
        log_info(f"[auto] Data supplement not interactive ({len(issues)} issues).")
        return False
