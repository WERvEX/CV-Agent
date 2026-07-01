from __future__ import annotations

from pathlib import Path

from cv_agent.core.config import HyperParams
from cv_agent.trainer.yolo_trainer import YOLOTrainer


def test_train_passes_quiet_verbose_flag_by_default(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            self.trainer = type("Trainer", (), {"save_dir": tmp_path / "actual"})()

        def train(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.YOLO", FakeYOLO)
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.ensure_amp_check_weights", lambda path: None)
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.resolve_device", lambda device: "cpu")
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.resolve_workers", lambda workers: 0)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")

    YOLOTrainer().train(
        model_variant="yolo26n",
        data_yaml=data_yaml,
        hyperparams=HyperParams(),
        epochs=1,
        run_dir=tmp_path / "run",
        device="cpu",
        use_amp=False,
    )

    assert captured["verbose"] is False


def test_train_allows_verbose_model_output(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            self.trainer = type("Trainer", (), {"save_dir": tmp_path / "actual"})()

        def train(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.YOLO", FakeYOLO)
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.ensure_amp_check_weights", lambda path: None)
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.resolve_device", lambda device: "cpu")
    monkeypatch.setattr("cv_agent.trainer.yolo_trainer.resolve_workers", lambda workers: 0)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [class0]\n", encoding="utf-8")

    YOLOTrainer().train(
        model_variant="yolo26n",
        data_yaml=data_yaml,
        hyperparams=HyperParams(),
        epochs=1,
        run_dir=tmp_path / "run",
        device="cpu",
        use_amp=False,
        model_verbose=True,
    )

    assert captured["verbose"] is True
