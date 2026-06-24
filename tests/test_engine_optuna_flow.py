from __future__ import annotations

import optuna

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer


class FakeTrial:
    number = 1


class FakeStudy:
    trials = []

    def __init__(self) -> None:
        self.told: list = []

    def tell(self, trial_number: int, score=None, state=None) -> None:
        self.told.append((trial_number, score, state))

    def ask(self):
        return FakeTrial()


def test_red_path_abandon_clears_pending_before_next_green(tmp_path):
    """Simulates RED skipping Optuna: pending trial must be abandoned."""
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study
    optimizer._pending_trial = FakeTrial()
    optimizer._pending_params = HyperParams(lr0=0.02)

    optimizer.abandon_pending()

    assert optimizer._pending_trial is None
    assert study.told[0][2] == optuna.trial.TrialState.FAIL


def test_round_trip_matching_params_tell_score(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study

    proposed = HyperParams(lr0=0.03, batch=16)
    optimizer._pending_trial = FakeTrial()
    optimizer._pending_params = proposed

    optimizer.report_result(0.55, proposed)

    assert study.told == [(1, 0.55, None)]
