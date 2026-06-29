from __future__ import annotations

import pytest

from cv_agent.trainer.device import resolve_device, resolve_workers


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_single_gpu() -> None:
    assert resolve_device("0") == 0


def test_resolve_device_multi_gpu() -> None:
    assert resolve_device("0,1,2,3") == [0, 1, 2, 3]


def test_resolve_workers_explicit() -> None:
    assert resolve_workers(4) == 4
