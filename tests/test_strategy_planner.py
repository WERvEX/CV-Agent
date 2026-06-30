from __future__ import annotations

from cv_agent.core.config import LLMConfig
from cv_agent.decision.llm_advisor import LLMAdvisor
from cv_agent.decision.strategy import StrategyPhase


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
