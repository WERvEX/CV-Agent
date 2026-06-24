from __future__ import annotations

import optuna

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer


class FakeTrial:
    number = 7


class FakeStudy:
    trials = [object()] * 8

    def __init__(self) -> None:
        self.told: list[tuple] = []

    def tell(self, trial_number: int, score=None, state=None) -> None:
        self.told.append((trial_number, score, state))

    def ask(self):
        return FakeTrial()


def test_report_result_tells_matching_params(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study
    optimizer._pending_trial = FakeTrial()
    optimizer._pending_params = HyperParams(lr0=0.01, batch=8)

    params = HyperParams(lr0=0.01, batch=8)
    optimizer.report_result(0.42, params)

    assert study.told == [(7, 0.42, None)]
    assert optimizer._pending_trial is None


def test_report_result_fails_on_param_mismatch(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study
    optimizer._pending_trial = FakeTrial()
    optimizer._pending_params = HyperParams(lr0=0.01, batch=8)

    optimizer.report_result(0.42, HyperParams(lr0=0.02, batch=8))

    assert study.told == [(7, None, optuna.trial.TrialState.FAIL)]


def test_abandon_pending_marks_fail(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study
    optimizer._pending_trial = FakeTrial()
    optimizer._pending_params = HyperParams()

    optimizer.abandon_pending()

    assert study.told == [(7, None, optuna.trial.TrialState.FAIL)]


def test_trial_budget_exhausted_skips_new_ask(tmp_path, monkeypatch):
    optimizer = OptunaOptimizer(OptunaConfig(n_trials=0), study_db=tmp_path / "study.db")
    monkeypatch.setattr(optimizer, "_init_study", lambda: None)
    optimizer._study = FakeStudy()

    params, from_optuna = optimizer.propose_next(HyperParams(), "green")

    assert from_optuna is False
    assert optimizer._pending_trial is None
