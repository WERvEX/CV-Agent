from __future__ import annotations

from cv_agent.decision.strategy import StrategyPatch, StrategyPhase
from cv_agent.decision.strategy_memory import StrategyMemory
from cv_agent.tracking.run_dir import load_strategy_memory, save_strategy_memory


def test_strategy_memory_records_effective_and_avoid_patterns():
    memory = StrategyMemory(max_items=3)

    memory.record_round(
        patch=StrategyPatch(phase=StrategyPhase.RECOVERY, reason="lower mosaic"),
        before_score=0.4,
        after_score=0.45,
        params={"mosaic": 0.2},
    )
    memory.record_round(
        patch=StrategyPatch(phase=StrategyPhase.EXPLORATION, reason="high lr"),
        before_score=0.45,
        after_score=0.30,
        params={"lr0": 0.01},
    )

    data = memory.model_dump()

    assert data["effective_patterns"]
    assert data["avoid_patterns"]


def test_strategy_memory_trims_patterns_to_max_items():
    memory = StrategyMemory(max_items=2)

    for idx in range(3):
        memory.record_round(
            patch=StrategyPatch(phase=StrategyPhase.EXPLORATION, reason=f"round {idx}"),
            before_score=0.1,
            after_score=0.2,
            params={"lr0": idx},
        )

    assert len(memory.effective_patterns) == 2
    assert "round 0" not in memory.effective_patterns[0]
    assert "round 1" in memory.effective_patterns[0]


def test_strategy_memory_persistence_save_load_and_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    memory = {
        "max_items": 3,
        "effective_patterns": ["recovery helped"],
        "avoid_patterns": ["high lr hurt"],
        "open_hypotheses": ["try lower mosaic"],
    }

    assert load_strategy_memory(run_dir) == {}

    save_strategy_memory(run_dir, memory)

    assert load_strategy_memory(run_dir) == memory
