from __future__ import annotations

import json

from cv_agent.core.config import LLMConfig
from cv_agent.decision.llm_advisor import LLMAdvisor
from cv_agent.decision.strategy import StrategyPhase


def _plan(advisor: LLMAdvisor):
    return advisor.plan_strategy(
        round_num=4,
        decision_summary={"color": "yellow", "action": "continue"},
        metrics={"mAP50-95": 0.41, "recall": 0.5, "precision": 0.6},
        history=[{"round": 1, "score": 0.4}, {"round": 2, "score": 0.42}],
        memory={},
        base_search_space={"lr0": [0.001, 0.01], "mosaic": [0.0, 1.0]},
    )


def test_plan_strategy_without_api_key_returns_heuristic_recovery_patch():
    advisor = LLMAdvisor(LLMConfig(api_key=""))

    patch = advisor.plan_strategy(
        round_num=4,
        decision_summary={"color": "red", "action": "rollback", "reason": "score dropped"},
        metrics={"mAP50-95": 0.31, "recall": 0.24, "precision": 0.6},
        history=[{"round": 1, "score": 0.4}, {"round": 2, "score": 0.35}],
        memory={},
        base_search_space={},
    )

    assert patch.phase is StrategyPhase.RECOVERY
    assert "lr0" in patch.search_space_patch
    assert patch.confidence > 0


def test_plan_strategy_valid_llm_json_returns_llm_patch_with_normalized_weights(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x"))
    advisor._client = object()
    monkeypatch.setattr(
        advisor,
        "_call_llm",
        lambda prompt: json.dumps({
            "phase": "exploitation",
            "reason": "stable improvement",
            "search_space_patch": {"lr0": [0.002, 0.006], "mosaic": [0.1, 0.4]},
            "freeze": ["batch"],
            "objective_weights": {
                "map50_95": 2.0,
                "map50": 1.0,
                "recall": 1.0,
                "precision": 1.0,
                "overfit_penalty": 0.0,
                "cost_penalty": 0.0,
            },
            "max_trials_for_phase": 3,
            "confidence": 0.8,
        }),
    )

    patch = _plan(advisor)

    assert patch.metadata["source"] == "llm"
    assert patch.phase is StrategyPhase.EXPLOITATION
    assert patch.search_space_patch == {"lr0": (0.002, 0.006), "mosaic": (0.1, 0.4)}
    assert patch.objective_weights is not None
    total = sum(patch.objective_weights.model_dump().values())
    assert round(total, 6) == 1.0
    assert round(patch.objective_weights.map50_95, 6) == 0.4


def test_plan_strategy_invalid_json_falls_back_to_heuristic(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x"))
    advisor._client = object()
    monkeypatch.setattr(advisor, "_call_llm", lambda prompt: "{not-json")

    patch = _plan(advisor)

    assert patch.metadata["source"] == "heuristic"


def test_plan_strategy_invalid_phase_falls_back_to_heuristic(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x"))
    advisor._client = object()
    monkeypatch.setattr(
        advisor,
        "_call_llm",
        lambda prompt: json.dumps({"phase": "panic", "confidence": 0.8}),
    )

    patch = _plan(advisor)

    assert patch.metadata["source"] == "heuristic"


def test_plan_strategy_invalid_freeze_field_falls_back_to_heuristic(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x"))
    advisor._client = object()
    monkeypatch.setattr(
        advisor,
        "_call_llm",
        lambda prompt: json.dumps({"phase": "exploration", "freeze": ["mosiac"]}),
    )

    patch = _plan(advisor)

    assert patch.metadata["source"] == "heuristic"


def test_plan_strategy_invalid_search_space_field_falls_back_to_heuristic(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x"))
    advisor._client = object()
    monkeypatch.setattr(
        advisor,
        "_call_llm",
        lambda prompt: json.dumps({
            "phase": "exploration",
            "search_space_patch": {"not_a_field": [0.1, 0.2]},
        }),
    )

    patch = _plan(advisor)

    assert patch.metadata["source"] == "heuristic"


def test_plan_strategy_call_limit_exhausted_skips_llm_and_returns_heuristic(monkeypatch):
    advisor = LLMAdvisor(LLMConfig(api_key="x", max_calls_per_session=1))
    advisor._client = object()
    advisor._call_count = 1
    monkeypatch.setattr(
        advisor,
        "_call_llm",
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    patch = _plan(advisor)

    assert patch.metadata["source"] == "heuristic"


def test_call_llm_stops_retrying_when_call_budget_is_exhausted(monkeypatch):
    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("temporary failure")

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FailingCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    advisor = LLMAdvisor(LLMConfig(api_key="x", max_calls_per_session=1))
    fake_client = FakeClient()
    advisor._client = fake_client
    monkeypatch.setattr("cv_agent.decision.llm_advisor.time.sleep", lambda seconds: None)

    response = advisor._call_llm("prompt")

    assert response is None
    assert fake_client.chat.completions.calls == 1
    assert advisor._call_count == 1
