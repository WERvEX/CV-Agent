from __future__ import annotations

import logging

from cv_agent.tracking.mlflow_manager import MLflowManager


def test_mlflow_manager_disables_async_trace_noise(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", raising=False)
    monkeypatch.delenv("MLFLOW_TRACE_SAMPLING_RATIO", raising=False)

    MLflowManager()

    import os

    assert os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] == "false"
    assert os.environ["MLFLOW_TRACE_SAMPLING_RATIO"] == "0"
    assert logging.getLogger("mlflow.tracing").level == logging.ERROR
