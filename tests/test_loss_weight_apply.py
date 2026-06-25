from __future__ import annotations

from cv_agent.core.config import HyperParams
from cv_agent.core.engine import TrainingEngine
from cv_agent.decision.three_state import Decision


def test_apply_loss_weight_suggestions_clamps():
    decision = Decision(color="red", action="rollback", reason="test")
    params = HyperParams(box=7.5, cls=0.5, dfl=1.5)

    updated = TrainingEngine._apply_loss_weight_suggestions(
        {"box": 25.0, "cls": 0.05, "dfl": 2.0},
        decision,
        params,
    )

    assert updated.box == 20.0
    assert updated.cls == 0.1
    assert updated.dfl == 2.0
    assert decision.metadata["loss_weight_adjustment"]["box"] == 20.0
