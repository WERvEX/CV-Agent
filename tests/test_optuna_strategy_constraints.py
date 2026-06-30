from __future__ import annotations

import optuna

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer
from cv_agent.decision.strategy import StrategyPatch, StrategyPhase


class RecordingTrial:
    number = 11

    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = (low + high) / 2
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[int]) -> int:
        value = choices[0]
        self.params[name] = value
        return value


class RecordingStudy:
    trials: list[object] = []

    def __init__(self) -> None:
        self.trial = RecordingTrial()
        self.told: list[tuple] = []

    def ask(self) -> RecordingTrial:
        return self.trial

    def tell(self, trial_number: int, score=None, state=None) -> None:
        self.told.append((trial_number, score, state))


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


def test_frozen_batch_is_not_sampled_into_pending_trial_params(tmp_path, monkeypatch):
    optimizer = OptunaOptimizer(OptunaConfig(n_trials=3), study_db=tmp_path / "study.db")
    study = RecordingStudy()
    optimizer.set_strategy_patch(StrategyPatch(freeze={"batch"}))
    monkeypatch.setattr(optimizer, "_init_study", lambda: None)
    optimizer._study = study

    params, from_optuna = optimizer.propose_next(HyperParams(batch=32), "green")

    assert from_optuna is True
    assert params.batch == 32
    assert "batch" not in study.trial.params
    assert optimizer._pending_params == params


def test_report_result_succeeds_with_frozen_params(tmp_path, monkeypatch):
    optimizer = OptunaOptimizer(OptunaConfig(n_trials=3), study_db=tmp_path / "study.db")
    study = RecordingStudy()
    optimizer.set_strategy_patch(StrategyPatch(freeze={"batch"}))
    monkeypatch.setattr(optimizer, "_init_study", lambda: None)
    optimizer._study = study

    params, _ = optimizer.propose_next(HyperParams(batch=32), "green")
    optimizer.report_result(0.73, params)

    assert study.told == [(11, 0.73, None)]
    assert all(tell[2] is not optuna.trial.TrialState.FAIL for tell in study.told)


def test_set_strategy_patch_none_resets_constraints(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")

    optimizer.set_strategy_patch(
        StrategyPatch(search_space_patch={"lr0": (0.001, 0.002)}, freeze={"batch"})
    )
    optimizer.set_strategy_patch(None)

    assert optimizer._effective_search_space == optimizer.search_space
    assert optimizer._frozen_fields == set()
