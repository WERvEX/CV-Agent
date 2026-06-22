"""Ask-before-edit mode interaction handler.

Before any config modification, weight replacement, data download, or
code-invasive change:
1. Render a change diff via Rich Panel
2. Block and wait for user input (Y/n confirmation)
3. Accept natural language explanations (e.g., "Don't change Mosaic, just adjust LR")
4. Store NL input in Decision.llm_context for next-round LLM prompting
"""

from __future__ import annotations

from typing import Any

from cv_agent.interaction.diff_renderer import (
    render_diff,
    render_single_value_diff,
)
from cv_agent.ui.console import console, log_decision, log_info, log_warning, print_decision_table
from cv_agent.ui.prompts import confirm, press_enter_to_continue, text
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AskModeHandler:
    """Interaction handler for the Ask-before-edit interactive mode."""

    def propagate_decision(self, decision: dict[str, Any], round_num: int) -> str:
        """Log a decision and capture user feedback.

        Args:
            decision: Decision dict (from Decision.to_dict()).
            round_num: Current round number.

        Returns:
            Raw user NL feedback string (may be empty).
        """
        color = decision.get("color", "white")
        action = decision.get("action", "?")
        reason = decision.get("reason", "")

        log_decision(color, f"Round {round_num}: [{action}] {reason}")
        print_decision_table(decision, round_num)

        # For Red or Yellow decisions, ask for user context
        user_feedback = ""
        if color in ("red", "yellow"):
            user_feedback = text(
                f"[yellow]Round {round_num} ({color.upper()}):[/yellow] {reason}\n"
                "Any guidance for the next round? (Enter to accept AI decision)",
                default="",
            )
            if user_feedback.strip():
                log_info(f"User feedback captured: '{user_feedback.strip()}'")

        return user_feedback

    def confirm(self, message: str, default: bool = True) -> bool:
        """Block for user yes/no confirmation.

        Args:
            message: The question to ask.
            default: Default answer.

        Returns:
            User's answer.
        """
        return confirm(message, default=default)

    def text_input(self, message: str, default: str = "") -> str:
        """Block for free-text user input.

        Args:
            message: The prompt to show.
            default: Default value.

        Returns:
            User's text input.
        """
        return text(message, default=default)

    def press_enter(self, message: str = "Press Enter to continue...") -> None:
        """Block until user presses Enter.

        Args:
            message: Prompt message.
        """
        press_enter_to_continue(message)

    def confirm_config_change(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        context: str = "",
    ) -> tuple[bool, str]:
        """Render config diff and block for user confirmation.

        Args:
            old_params: Previous HyperParams dict.
            new_params: Proposed next HyperParams dict.
            context: Additional context string (e.g., "Red round — rollback proposed").

        Returns:
            (approved: bool, user_feedback: str)
        """
        console.print()
        log_warning(f"Proposed hyperparameter change: {context}")

        # Show only the changed parameters
        changed = {}
        for key, new_val in new_params.items():
            old_val = old_params.get(key)
            if old_val != new_val:
                changed[key] = {"old": old_val, "new": new_val}

        if changed:
            # Build a simple diff view showing all changes
            console.print(render_diff(old_params, new_params, title="Hyperparameter Changes"))
        else:
            log_info("No parameter changes detected.")

        approved = confirm("Apply these hyperparameter changes?", default=True)

        if not approved:
            user_feedback = text(
                "Explain your reasoning (e.g., 'Don't change Mosaic, just adjust LR'):",
                default="",
            )
            log_info(f"Change rejected. User feedback: '{user_feedback}'")
            return False, user_feedback

        log_success = "Changes approved."
        console.print(f"[green]{log_success}[/green]")
        return True, ""

    def confirm_supplement(self, issues: list[dict[str, Any]]) -> bool:
        """Display dataset issues and confirm data supplement actions.

        Args:
            issues: Validation issue dicts.

        Returns:
            True to proceed with supplement, False to force-skip.
        """
        console.print()
        log_warning(f"Dataset validation found {len(issues)} issue(s).")
        return confirm("Enter data supplement mode? (Y=generate scripts + wait, n=skip)", default=True)

    def confirm_weight_rollback(self, checkpoint_path: str) -> bool:
        """Confirm before rolling back to a prior checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint to restore.

        Returns:
            User's confirmation.
        """
        log_warning(f"Rollback proposed: restoring weights from {checkpoint_path}")
        return confirm("Rollback to previous best checkpoint?", default=True)

    def confirm_download_execution(self, script_path: str) -> bool:
        """Confirm before executing a data download script.

        Args:
            script_path: Path to the download script.

        Returns:
            User's confirmation.
        """
        log_info(f"Data supplement script ready: {script_path}")
        return confirm("Execute this download script?", default=False)