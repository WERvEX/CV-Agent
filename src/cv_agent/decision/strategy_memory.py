from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cv_agent.decision.strategy import StrategyPatch


class StrategyMemory(BaseModel):
    max_items: int = Field(default=50, ge=1)
    effective_patterns: list[str] = Field(default_factory=list)
    avoid_patterns: list[str] = Field(default_factory=list)
    open_hypotheses: list[str] = Field(default_factory=list)

    def record_round(
        self,
        patch: StrategyPatch,
        before_score: float,
        after_score: float,
        params: dict[str, Any],
    ) -> None:
        delta = after_score - before_score
        summary = f"{patch.phase.value}: {patch.reason}; params={params}"
        if delta > 0:
            self.effective_patterns.append(summary)
        elif delta < 0:
            self.avoid_patterns.append(summary)

        self.effective_patterns = self.effective_patterns[-self.max_items :]
        self.avoid_patterns = self.avoid_patterns[-self.max_items :]
        self.open_hypotheses = self.open_hypotheses[-self.max_items :]
