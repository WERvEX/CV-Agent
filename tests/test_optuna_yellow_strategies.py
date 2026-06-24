from __future__ import annotations

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer


def test_random_walk_batch_stays_neighbor(tmp_path):
    cfg = OptunaConfig(search_space=OptunaConfig().search_space)
    optimizer = OptunaOptimizer(cfg, study_db=tmp_path / "study.db")
    base = HyperParams(batch=16, lr0=0.01, lrf=0.01)

    batches = set()
    for _ in range(30):
        params = optimizer._propose_random_walk(base)
        batches.add(params.batch)

    assert batches.issubset({8, 16, 32})


def test_sa_improves_keeps_current_params(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    base = HyperParams(lr0=0.01, lrf=0.01)
    optimizer._sa_best_score = 0.3
    optimizer._sa_best_params = base

    result = optimizer._propose_simulated_annealing(base, current_score=0.5)

    assert result.lr0 == base.lr0
    assert result.batch == base.batch


def test_yellow_random_walk_propose_next_returns_hyperparams_not_nested_tuple(tmp_path):
    """Regression: propose_next must not double-wrap (params, bool) from random walk."""
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    base = HyperParams(lr0=0.01, lrf=0.01)

    params, from_optuna = optimizer.propose_next(base, "yellow", current_score=0.5)

    assert isinstance(params, HyperParams)
    assert from_optuna is False
    assert hasattr(params, "model_dump")
