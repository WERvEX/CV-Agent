from __future__ import annotations

from cv_agent.core.config import OptunaSearchSpace
from cv_agent.decision.strategy import ObjectiveWeights, StrategyPatch, StrategyPhase


def test_strategy_patch_clamps_search_space_to_base_bounds():
    base = OptunaSearchSpace(lr0=(0.001, 0.01), mosaic=(0.0, 1.0))
    patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="recover from red rounds",
        search_space_patch={"lr0": (0.00001, 0.02), "mosaic": (0.2, 0.8)},
    )

    effective = patch.apply_to_search_space(base)

    assert effective.lr0 == (0.001, 0.01)
    assert effective.mosaic == (0.2, 0.8)


def test_objective_weights_normalize_positive_values():
    weights = ObjectiveWeights(map50_95=2.0, recall=1.0, precision=1.0)
    normalized = weights.normalized()

    assert round(normalized.map50_95, 6) == 0.5
    assert round(normalized.recall, 6) == 0.25
    assert round(normalized.precision, 6) == 0.25
