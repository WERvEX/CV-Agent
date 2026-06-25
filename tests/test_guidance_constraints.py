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


def test_apply_multiplier_clamped():
    from cv_agent.core.config import OptunaSearchSpace

    old = HyperParams(lr0=0.01, mosaic=0.5)
    proposed = HyperParams(lr0=0.02, mosaic=0.3)
    constraints = parse_guidance("tweak")
    constraints.multipliers = {"lr0": 0.5}
    result = apply_guidance_constraints(
        old, proposed, constraints, search_space=OptunaSearchSpace()
    )
    assert result.lr0 == 0.005


def test_interpret_guidance_handles_null_json_fields():
    """LLM may return null for multipliers/set_values instead of omitting keys."""
    import json

    from cv_agent.core.config import LLMConfig, OptunaSearchSpace
    from cv_agent.decision.llm_advisor import LLMAdvisor

    advisor = LLMAdvisor(LLMConfig(api_key="test-key"))
    advisor._client = object()  # skip heuristic path
    advisor._call_llm = lambda prompt: json.dumps({
        "frozen_fields": None,
        "multipliers": None,
        "set_values": {"lr0": 0.015},
        "replace_proposal": False,
        "reason": "Slightly increase lr0",
    })

    result = advisor.interpret_guidance(
        feedback="lr 稍微大一点",
        current_params=HyperParams(lr0=0.01),
        proposed_params=HyperParams(lr0=0.01),
        decision_summary={"color": "green"},
        metrics={"mAP50": 0.5},
        search_space=OptunaSearchSpace(),
    )

    assert result is not None
    assert result.adjustments == {"lr0": 0.015}
    assert result.multipliers == {}
    assert result.frozen_fields == set()
