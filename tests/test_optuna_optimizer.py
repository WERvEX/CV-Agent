from __future__ import annotations

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer


class FakeTrial:
    number = 7


class FakeStudy:
    trials = [object()] * 8

    def __init__(self) -> None:
        self.told = []

    def tell(self, trial_number: int, score: float) -> None:
        self.told.append((trial_number, score))


def test_report_result_tells_the_last_asked_trial_number(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(), study_db=tmp_path / "study.db")
    study = FakeStudy()
    optimizer._study = study
    optimizer._last_trial = FakeTrial()

    optimizer._report_result(HyperParams(), 0.42)

    assert study.told == [(7, 0.42)]
