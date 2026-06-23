"""Shared types for ask/auto interaction handlers."""

from __future__ import annotations

from dataclasses import dataclass


class SessionQuit(Exception):
    """Raised when the user chooses to quit the training session from a prompt."""


@dataclass
class DecisionReview:
    """User response after reviewing a round decision."""

    apply_recommendation: bool = True
    rollback_approved: bool = True
    feedback: str = ""
    quit_session: bool = False


@dataclass
class ConfigChangeReview:
    """User response after reviewing a hyperparameter diff."""

    approved: bool = True
    feedback: str = ""
    quit_session: bool = False
