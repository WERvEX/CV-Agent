"""Lightweight parsing of user natural-language guidance into param constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cv_agent.core.config import HyperParams, OptunaSearchSpace

if TYPE_CHECKING:
    from cv_agent.decision.llm_advisor import LLMAdvisor

# Fields that Optuna / rules may change
TUNABLE_FIELDS = frozenset(HyperParams.model_fields.keys())


@dataclass
class GuidanceConstraints:
    """Structured constraints extracted from user feedback."""

    frozen_fields: set[str] = field(default_factory=set)
    only_lr: bool = False
    adjustments: dict[str, float] = field(default_factory=dict)
    multipliers: dict[str, float] = field(default_factory=dict)
    replace_proposal: bool = False
    reason: str = ""
    source: Literal["regex", "llm"] = "regex"
    raw_text: str = ""

    def to_metadata(self) -> dict:
        return {
            "frozen_fields": sorted(self.frozen_fields),
            "only_lr": self.only_lr,
            "adjustments": dict(self.adjustments),
            "multipliers": dict(self.multipliers),
            "replace_proposal": self.replace_proposal,
            "reason": self.reason,
            "source": self.source,
            "raw_text": self.raw_text,
        }


def parse_guidance(text: str) -> GuidanceConstraints:
    """Parse common English/Chinese guidance phrases into constraints."""
    if not text or not text.strip():
        return GuidanceConstraints()

    lowered = text.lower().strip()
    constraints = GuidanceConstraints(raw_text=text.strip())

    if re.search(r"\bonly\s+lr\b|\bjust\s+lr\b|只改\s*lr|只调\s*lr|仅.*学习率", lowered):
        constraints.only_lr = True

    field_aliases = {
        "mosaic": ("mosaic",),
        "mixup": ("mixup",),
        "batch": ("batch", "批次"),
        "lr0": ("lr0", "lr", "learning rate", "学习率"),
        "lrf": ("lrf",),
        "weight_decay": ("weight_decay", "weight decay", "权重衰减"),
        "degrees": ("degrees", "旋转"),
        "box": ("box",),
        "cls": ("cls",),
        "dfl": ("dfl",),
    }

    freeze_patterns = [
        r"don'?t\s+change\s+(\w+)",
        r"do\s+not\s+change\s+(\w+)",
        r"keep\s+(\w+)",
        r"不要改\s*(\w+)",
        r"别改\s*(\w+)",
        r"保持\s*(\w+)",
    ]

    for pattern in freeze_patterns:
        for match in re.finditer(pattern, lowered):
            token = match.group(1).strip()
            for field_name, aliases in field_aliases.items():
                if token in aliases or token == field_name:
                    constraints.frozen_fields.add(field_name)

    return constraints


def _clamp_to_search_space(
    params: dict[str, Any],
    search_space: OptunaSearchSpace | None,
) -> dict[str, Any]:
    if search_space is None:
        return params

    result = dict(params)
    for key, value in result.items():
        if key == "batch":
            choices = sorted(search_space.batch)
            if choices and int(value) not in choices:
                result[key] = min(choices, key=lambda c: abs(c - int(value)))
            continue

        bounds = getattr(search_space, key, None)
        if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
            continue

        low, high = bounds
        if isinstance(value, (int, float)):
            result[key] = max(low, min(high, float(value)))

    return result


def apply_guidance_constraints(
    old_params: HyperParams,
    proposed_params: HyperParams,
    constraints: GuidanceConstraints,
    search_space: OptunaSearchSpace | None = None,
) -> HyperParams:
    """Apply parsed constraints to a proposed hyperparameter set."""
    if not constraints.raw_text and not constraints.adjustments and not constraints.multipliers:
        return proposed_params

    old = old_params.model_dump()
    proposed = proposed_params.model_dump()
    result = dict(proposed)

    for key, mult in constraints.multipliers.items():
        if key in TUNABLE_FIELDS and key in old:
            result[key] = old[key] * mult

    for key, value in constraints.adjustments.items():
        if key in TUNABLE_FIELDS:
            result[key] = value

    if constraints.only_lr:
        for key in TUNABLE_FIELDS:
            if key not in ("lr0", "lrf"):
                result[key] = old[key]

    for field_name in constraints.frozen_fields:
        if field_name in old:
            result[field_name] = old[field_name]

    clamped = _clamp_to_search_space(result, search_space)
    return HyperParams(**clamped)


def apply_guidance_adjustments(
    old_params: HyperParams,
    proposed_params: HyperParams,
    constraints: GuidanceConstraints,
    search_space: OptunaSearchSpace | None = None,
) -> HyperParams:
    """Apply full guidance constraints including LLM adjustments."""
    base = old_params if constraints.replace_proposal else proposed_params
    return apply_guidance_constraints(old_params, base, constraints, search_space)


def parse_guidance_with_llm(
    text: str,
    *,
    advisor: LLMAdvisor | None,
    current_params: HyperParams,
    proposed_params: HyperParams,
    decision_summary: dict[str, Any],
    metrics: dict[str, float],
    search_space: OptunaSearchSpace,
    fallback_regex: bool = True,
) -> GuidanceConstraints:
    """Parse guidance via LLM when available; fall back to regex rules."""
    if not text or not text.strip():
        return GuidanceConstraints()

    if advisor is not None:
        llm_result = advisor.interpret_guidance(
            feedback=text,
            current_params=current_params,
            proposed_params=proposed_params,
            decision_summary=decision_summary,
            metrics=metrics,
            search_space=search_space,
        )
        if llm_result is not None:
            return llm_result

    if fallback_regex:
        return parse_guidance(text)

    return GuidanceConstraints(raw_text=text.strip())
