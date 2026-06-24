"""Lightweight parsing of user natural-language guidance into param constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cv_agent.core.config import HyperParams

# Fields that Optuna / rules may change
TUNABLE_FIELDS = frozenset(HyperParams.model_fields.keys())


@dataclass
class GuidanceConstraints:
    """Structured constraints extracted from user feedback."""

    frozen_fields: set[str] = field(default_factory=set)
    only_lr: bool = False
    raw_text: str = ""

    def to_metadata(self) -> dict:
        return {
            "frozen_fields": sorted(self.frozen_fields),
            "only_lr": self.only_lr,
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


def apply_guidance_constraints(
    old_params: HyperParams,
    proposed_params: HyperParams,
    constraints: GuidanceConstraints,
) -> HyperParams:
    """Apply parsed constraints to a proposed hyperparameter set."""
    if not constraints.raw_text:
        return proposed_params

    old = old_params.model_dump()
    proposed = proposed_params.model_dump()
    result = dict(proposed)

    if constraints.only_lr:
        for key in TUNABLE_FIELDS:
            if key not in ("lr0", "lrf"):
                result[key] = old[key]

    for field_name in constraints.frozen_fields:
        if field_name in old:
            result[field_name] = old[field_name]

    return HyperParams(**result)
