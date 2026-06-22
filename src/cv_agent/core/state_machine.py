"""State machine for the training loop.

Manages the high-level states of a cv_agent run:
    INIT -> VALIDATE_DATA -> TRAIN -> EVALUATE -> DECIDE -> TRAIN (loop)

With DATA_SUPPLEMENT as an off-ramp when validation fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class TrainingLoopState(Enum):
    """States of the automated training loop."""

    INIT = auto()
    VALIDATE_DATA = auto()
    DATA_SUPPLEMENT = auto()
    TRAIN = auto()
    EVALUATE = auto()
    DECIDE = auto()
    DONE = auto()


class DecisionColor(Enum):
    """Three-state decision outcome."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class DecisionAction(Enum):
    """Actions taken based on decision color and diagnostics."""

    ACCEPT = "accept"
    ROLLBACK = "rollback"
    ROLLBACK_REGULARIZE = "rollback_regularize"
    AGGRESSIVE_LR_ADJUST = "aggressive_lr_adjust"
    ESCAPE_LOCAL_OPTIMUM = "escape_local_optimum"
    DATA_GAP_RESEARCH = "data_gap_research"


@dataclass
class RedCountTracker:
    """Tracks consecutive Red decisions and triggers escalation."""

    max_consecutive: int = 3
    count: int = 0

    def increment(self) -> bool:
        """Return True if escalation threshold reached."""
        self.count += 1
        return self.count >= self.max_consecutive

    def reset(self) -> None:
        self.count = 0

    @property
    def is_escalated(self) -> bool:
        return self.count >= self.max_consecutive