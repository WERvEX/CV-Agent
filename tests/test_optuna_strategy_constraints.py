from __future__ import annotations

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer
from cv_agent.decision.strategy import StrategyPatch, StrategyPhase


def test_strategy_patch_constrains_bayesian_proposal(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(n_trials=3), study_db=tmp_path / "study.db")
    optimizer.set_strategy_patch(
        StrategyPatch(
            phase=StrategyPhase.RECOVERY,
            search_space_patch={"lr0": (0.001, 0.002), "mosaic": (0.0, 0.1)},
            freeze={"batch"},
        )
    )

    params, from_optuna = optimizer.propose_next(
        HyperParams(batch=16),
        "green",
        current_score=0.5,
    )

    assert from_optuna is True
    assert 0.001 <= params.lr0 <= 0.002
    assert 0.0 <= params.mosaic <= 0.1
    assert params.batch == 16
