from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cv_agent.core.config import ObjectiveWeights, OptunaSearchSpace


class StrategyPhase(str, Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    RECOVERY = "recovery"
    DATA_GAP = "data_gap"
    STABILITY_CHECK = "stability_check"


class StrategyPatch(BaseModel):
    phase: StrategyPhase = StrategyPhase.EXPLORATION
    reason: str = ""
    search_space_patch: dict[str, tuple[float, float]] = Field(default_factory=dict)
    freeze: set[str] = Field(default_factory=set)
    objective_weights: ObjectiveWeights | None = None
    max_trials_for_phase: int | None = Field(default=None, ge=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("freeze")
    @classmethod
    def validate_freeze_fields(cls, value: set[str]) -> set[str]:
        valid_fields = set(OptunaSearchSpace.model_fields)
        invalid_fields = sorted(field for field in value if field not in valid_fields)
        if invalid_fields:
            raise ValueError(f"Invalid freeze field(s): {', '.join(invalid_fields)}")
        return value

    @field_validator("search_space_patch")
    @classmethod
    def validate_search_space_patch(
        cls,
        value: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[float, float]]:
        base_fields = OptunaSearchSpace.model_fields
        invalid_fields: list[str] = []
        non_numeric_fields: list[str] = []
        inverted_fields: list[str] = []

        for key, bounds in value.items():
            if key not in base_fields:
                invalid_fields.append(key)
                continue

            default = base_fields[key].default
            if not isinstance(default, tuple) or len(default) != 2:
                non_numeric_fields.append(key)
                continue

            low, high = bounds
            if low > high:
                inverted_fields.append(key)

        if invalid_fields:
            raise ValueError(f"Invalid search_space_patch field(s): {', '.join(sorted(invalid_fields))}")
        if non_numeric_fields:
            raise ValueError(
                "search_space_patch only supports numeric interval field(s); "
                f"invalid: {', '.join(sorted(non_numeric_fields))}"
            )
        if inverted_fields:
            raise ValueError(
                "search_space_patch interval low must be <= high for field(s): "
                f"{', '.join(sorted(inverted_fields))}"
            )
        return value

    def apply_to_search_space(self, base: OptunaSearchSpace) -> OptunaSearchSpace:
        data = base.model_dump()
        for key, bounds in self.search_space_patch.items():
            if key not in data:
                continue
            base_bounds = data[key]
            if not isinstance(base_bounds, tuple) or len(base_bounds) != 2:
                continue
            low = max(float(bounds[0]), float(base_bounds[0]))
            high = min(float(bounds[1]), float(base_bounds[1]))
            if low <= high:
                data[key] = (low, high)
        return OptunaSearchSpace(**data)
