"""Auto mode interaction handler — print-only, no blocking.

All confirmations return True. All decisions are logged with Rich-styled
color-coded output. Writes decision_log.json to run directory.
"""

from __future__ import annotations

from typing import Any

from cv_agent.ui.console import log_decision, log_info, log_success, log_warning, print_decision_table
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AutoModeHandler:
    """Interaction handler for fully autonomous Auto mode."""

    def propagate_decision(self, decision: dict[str, Any], round_num: int) -> None:
        """Log a decision with color-coded Rich output.

        Args:
            decision: Decision dict (from Decision.to_dict()).
            round_num: Current round number.
        """
        color = decision.get("color", "white")
        action = decision.get("action", "?")
        reason = decision.get("reason", "")

        log_decision(color, f"Round {round_num}: [{action}] {reason}")
        print_decision_table(decision, round_num)

    def confirm(self, message: str, default: bool = True) -> bool:
        """Auto-approve all confirmations.

        Args:
            message: The confirmation prompt.
            default: Default value.

        Returns:
            Always True (auto-approved).
        """
        log_info(f"[auto] {message} → Auto-approved (Y)")
        return True

    def text(self, message: str, default: str = "") -> str:
        """Return default text without blocking.

        Args:
            message: Prompt message.
            default: Default value.

        Returns:
            Default value.
        """
        log_info(f"[auto] {message} → Using default: '{default}'")
        return default

    def press_enter(self, message: str = "Press Enter to continue...") -> None:
        """Skip wait in auto mode.

        Args:
            message: Prompt message.
        """
        log_info(f"[auto] Skipping wait: {message}")

    def confirm_config_change(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        context: str = "",
    ) -> tuple[bool, str]:
        """Auto-approve config changes and return empty NL context.

        Args:
            old_params: Previous HyperParams dict.
            new_params: Proposed next HyperParams dict.
            context: Additional context string.

        Returns:
            (True, "") — auto-approved with no user feedback.
        """
        log_info(f"[auto] Config change auto-approved. {context}")
        return True, ""

    def confirm_supplement(self, issues: list[dict[str, Any]]) -> bool:
        """Auto-approve data supplement scripts.

        Args:
            issues: Validation issue dicts.

        Returns:
            Always True.
        """
        log_info(f"[auto] Data supplement auto-approved ({len(issues)} issues).")
        return True