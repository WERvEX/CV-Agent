from __future__ import annotations

from cv_agent.core.config import HyperParams
from cv_agent.decision.guidance import apply_guidance_constraints, parse_guidance


def test_parse_only_lr():
    c = parse_guidance("only lr please")
    assert c.only_lr is True


def test_parse_freeze_mosaic():
    c = parse_guidance("Don't change mosaic, just tweak lr")
    assert "mosaic" in c.frozen_fields


def test_apply_only_lr_freezes_other_fields():
    old = HyperParams(lr0=0.01, lrf=0.01, batch=8, mosaic=0.5)
    proposed = HyperParams(lr0=0.05, lrf=0.02, batch=32, mosaic=0.1)
    constraints = parse_guidance("only lr")
    result = apply_guidance_constraints(old, proposed, constraints)
    assert result.lr0 == 0.05
    assert result.lrf == 0.02
    assert result.batch == 8
    assert result.mosaic == 0.5


def test_apply_freeze_field():
    old = HyperParams(mosaic=0.8, lr0=0.01)
    proposed = HyperParams(mosaic=0.2, lr0=0.05)
    constraints = parse_guidance("keep mosaic")
    result = apply_guidance_constraints(old, proposed, constraints)
    assert result.mosaic == 0.8
    assert result.lr0 == 0.05
