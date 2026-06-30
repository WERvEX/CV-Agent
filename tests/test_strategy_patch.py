from __future__ import annotations

import pytest

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

    assert round(normalized.map50_95, 6) == 0.470588
    assert round(normalized.map50, 6) == 0.035294
    assert round(normalized.recall, 6) == 0.235294
    assert round(normalized.precision, 6) == 0.235294
    assert round(normalized.overfit_penalty, 6) == 0.023529


def test_objective_weights_normalized_stable_after_model_dump_round_trip():
    weights = ObjectiveWeights(map50_95=2.0, recall=1.0, precision=1.0)

    normalized = weights.normalized()
    round_tripped = ObjectiveWeights.model_validate(weights.model_dump()).normalized()

    assert round_tripped == normalized


def test_objective_weights_normalized_stable_after_json_round_trip():
    weights = ObjectiveWeights(map50_95=2.0, recall=1.0, precision=1.0)

    normalized = weights.normalized()
    round_tripped = ObjectiveWeights.model_validate_json(weights.model_dump_json()).normalized()

    assert round_tripped == normalized


def test_strategy_patch_rejects_invalid_freeze_field():
    with pytest.raises(ValueError, match="mosiac"):
        StrategyPatch(freeze={"mosiac"})


def test_strategy_patch_accepts_valid_freeze_field():
    patch = StrategyPatch(freeze={"mosaic"})

    assert patch.freeze == {"mosaic"}


def test_strategy_patch_coerces_search_space_patch_list_to_tuple():
    patch = StrategyPatch(search_space_patch={"lr0": [0.001, 0.002]})

    assert patch.search_space_patch == {"lr0": (0.001, 0.002)}


def test_strategy_patch_rejects_invalid_search_space_patch_key():
    with pytest.raises(ValueError, match="not_a_field"):
        StrategyPatch(search_space_patch={"not_a_field": (0.1, 0.2)})


def test_strategy_patch_rejects_non_numeric_search_space_patch_field():
    with pytest.raises(ValueError, match="batch"):
        StrategyPatch(search_space_patch={"batch": (8, 16)})


def test_strategy_patch_rejects_inverted_search_space_patch_interval():
    with pytest.raises(ValueError, match="lr0"):
        StrategyPatch(search_space_patch={"lr0": (0.01, 0.001)})
