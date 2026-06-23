"""Ask-before-edit mode interaction handler.

Before any config modification, weight replacement, or rollback:
1. Render a clear recommendation panel (action, reason, rollback, param deltas)
2. Block for an explicit choice (apply / skip rollback / reject / quit)
3. Accept natural-language guidance for the next round
"""

from __future__ import annotations

from typing import Any

from cv_agent.interaction.diff_renderer import render_diff
from cv_agent.interaction.types import ConfigChangeReview, DecisionReview, SessionQuit
from cv_agent.ui.console import (
    console,
    log_decision,
    log_info,
    log_warning,
    print_decision_recommendation,
)
from cv_agent.ui.prompts import select_action, text
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AskModeHandler:
    """Interaction handler for the Ask-before-edit interactive mode."""

    def propagate_decision(self, decision: dict[str, Any], round_num: int) -> str:
        """Legacy hook — prefer :meth:`review_decision`."""
        review = self.review_decision(
            decision=decision,
            round_num=round_num,
            current_params=None,
            checkpoint_path=None,
        )
        return review.feedback

    def review_decision(
        self,
        decision: dict[str, Any],
        round_num: int,
        current_params: dict[str, Any] | None,
        checkpoint_path: str | None,
    ) -> DecisionReview:
        """Show the AI recommendation and block for an explicit user choice."""
        color = decision.get("color", "white")
        action = decision.get("action", "?")
        reason = decision.get("reason", "")

        log_decision(color, f"Round {round_num}: [{action}] {reason}")
        print_decision_recommendation(
            decision,
            round_num,
            current_params=current_params,
            checkpoint_path=checkpoint_path,
        )

        should_rollback = bool(decision.get("should_rollback"))

        if color == "red":
            choices = [
                (
                    "apply_full",
                    "Apply recommendation (rollback + param changes)" if should_rollback
                    else "Apply recommendation (param changes)",
                ),
            ]
            if should_rollback:
                choices.append(
                    ("skip_rollback", "Apply param changes WITHOUT rollback"),
                )
            choices.extend([
                ("reject", "Reject recommendation — provide guidance"),
                ("continue", "Keep current weights — no AI param changes"),
            ])
            choice = select_action(
                "How do you want to handle this RED round?",
                choices,
                default_key="apply_full",
            )
        elif color == "yellow":
            choice = select_action(
                "How do you want to handle this YELLOW round?",
                [
                    ("apply_full", "Escape local optimum (AI recommendation)"),
                    ("guidance", "Add guidance for next round"),
                    ("continue", "Continue without AI escape strategy"),
                ],
                default_key="apply_full",
            )
        else:
            choice = select_action(
                "How do you want to proceed after this GREEN round?",
                [
                    ("apply_full", "Continue with AI plan (commit + Optuna search)"),
                    ("guidance", "Add guidance for next round"),
                    ("continue", "Continue without extra guidance"),
                ],
                default_key="apply_full",
            )

        feedback = ""
        if choice in ("guidance", "reject"):
            feedback = text(
                "Your guidance for the next round (e.g. 'Don't change Mosaic, just LR'):",
                default="",
            ).strip()
            if feedback:
                log_info(f"User feedback captured: '{feedback}'")

        if choice == "apply_full":
            return DecisionReview(
                apply_recommendation=True,
                rollback_approved=should_rollback,
                feedback=feedback,
            )
        if choice == "skip_rollback":
            return DecisionReview(
                apply_recommendation=True,
                rollback_approved=False,
                feedback=feedback,
            )
        if choice == "reject":
            return DecisionReview(
                apply_recommendation=False,
                rollback_approved=False,
                feedback=feedback,
            )
        # continue — green/yellow: proceed but skip AI-driven param mutation
        return DecisionReview(
            apply_recommendation=False,
            rollback_approved=False,
            feedback=feedback,
        )

    def confirm(self, message: str, default: bool = True) -> bool:
        """Block for user yes/no confirmation."""
        from cv_agent.ui.prompts import confirm as q_confirm

        try:
            action = select_action(
                message,
                [
                    ("yes", "Yes"),
                    ("no", "No"),
                ],
                default_key="yes" if default else "no",
            )
            return action == "yes"
        except SessionQuit:
            raise

    def text(self, message: str, default: str = "") -> str:
        """Block for free-text user input (protocol alias)."""
        return self.text_input(message, default=default)

    def text_input(self, message: str, default: str = "") -> str:
        """Block for free-text user input."""
        return text(message, default=default)

    def press_enter(self, message: str = "Press Enter to continue...") -> None:
        """Block until user continues or quits."""
        from cv_agent.ui.prompts import press_enter_to_continue

        press_enter_to_continue(message)

    def confirm_config_change(
        self,
        old_params: dict[str, Any],
        new_params: dict[str, Any],
        context: str = "",
    ) -> ConfigChangeReview:
        """Render config diff and block for user confirmation."""
        console.print()
        log_warning(f"Proposed hyperparameter change: {context}")

        changed = {
            key: {"old": old_params.get(key), "new": new_val}
            for key, new_val in new_params.items()
            if old_params.get(key) != new_val
        }

        if changed:
            console.print(render_diff(old_params, new_params, title="Hyperparameter Changes"))
        else:
            log_info("No parameter changes in this proposal.")

        choice = select_action(
            "Apply these hyperparameter changes for the next round?",
            [
                ("yes", "Yes, apply changes"),
                ("no", "No — explain what to change instead"),
                ("skip", "Skip — keep current hyperparameters"),
            ],
            default_key="yes",
        )

        if choice == "yes":
            console.print("[green]Changes approved.[/green]")
            return ConfigChangeReview(approved=True)

        feedback = ""
        if choice == "no":
            feedback = text(
                "Explain your reasoning (e.g., 'Don't change Mosaic, just adjust LR'):",
                default="",
            ).strip()
            log_info(f"Change rejected. User feedback: '{feedback}'")

        return ConfigChangeReview(approved=False, feedback=feedback)

    def confirm_supplement(self, issues: list[dict[str, Any]]) -> bool:
        """Display dataset issues and confirm data supplement actions."""
        console.print()
        log_warning(f"Dataset validation found {len(issues)} issue(s).")
        action = select_action(
            "Dataset validation failed. What should we do?",
            [
                ("supplement", "Enter data supplement mode (generate scripts + retry)"),
                ("retry", "Retry validation (I fixed the data manually)"),
                ("abort", "Abort this training session"),
            ],
            default_key="supplement",
        )
        if action == "abort":
            raise SessionQuit("User aborted after dataset validation failure.")
        return action in ("supplement", "retry")

    def confirm_weight_rollback(self, checkpoint_path: str) -> bool:
        """Confirm before rolling back to a prior checkpoint."""
        log_warning(f"Rollback proposed: restoring weights from {checkpoint_path}")
        action = select_action(
            "Rollback to previous best checkpoint?",
            [
                ("yes", "Yes, rollback"),
                ("no", "No, keep current weights"),
            ],
            default_key="yes",
        )
        return action == "yes"

    def confirm_download_execution(self, script_path: str) -> bool:
        """Confirm before executing a data download script."""
        log_info(f"Data supplement script ready: {script_path}")
        action = select_action(
            f"Execute download script {script_path}?",
            [
                ("no", "No — I'll run it manually"),
                ("yes", "Yes, execute now"),
            ],
            default_key="no",
        )
        return action == "yes"
