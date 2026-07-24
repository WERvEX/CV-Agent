from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cv_agent.cli.main import _build_data_yaml_override, _load_config, cli


def test_data_yaml_override_does_not_reintroduce_tracked_thresholds() -> None:
    override = _build_data_yaml_override(Path("coco128.yaml"))

    assert override == {"data_yaml": Path("coco128.yaml")}


def test_load_config_preserves_local_data_threshold_when_data_yaml_is_overridden(tmp_path) -> None:
    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text(
        """
data:
  data_yaml: tracked.yaml
  min_ann_per_class: 50
""",
        encoding="utf-8",
    )
    config_path.with_suffix(".local.yaml").write_text(
        """
data:
  min_ann_per_class: 1
""",
        encoding="utf-8",
    )

    config = _load_config(config_path, {"data": _build_data_yaml_override(Path("override.yaml"))})

    assert config.data.data_yaml == Path("override.yaml")
    assert config.data.min_ann_per_class == 1


def test_load_config_skips_local_override_when_path_is_directory(tmp_path) -> None:
    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text("max_rounds: 10\n", encoding="utf-8")
    (tmp_path / "cv_agent.local.yaml").mkdir()

    config = _load_config(config_path, {})

    assert config.max_rounds == 10


def test_load_config_works_without_local_override_file(tmp_path) -> None:
    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text("max_rounds: 7\nepochs_per_round: 3\n", encoding="utf-8")

    config = _load_config(config_path, {})

    assert config.max_rounds == 7
    assert config.epochs_per_round == 3


def test_load_config_preserves_early_stop_without_cli_override(tmp_path) -> None:
    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text(
        """
early_stop:
  enabled: true
  metric: precision
  target: 0.82
""",
        encoding="utf-8",
    )

    config = _load_config(config_path, {})

    assert config.early_stop.enabled is True
    assert config.early_stop.metric == "precision"
    assert config.early_stop.target == 0.82


def test_early_stop_target_override_enables_stop_and_preserves_metric(tmp_path) -> None:
    from cv_agent.cli.main import _build_early_stop_override

    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text("early_stop:\n  metric: recall\n  target: 0.9\n", encoding="utf-8")

    config = _load_config(
        config_path,
        _build_early_stop_override(enabled=None, metric=None, target=0.75),
    )

    assert config.early_stop.enabled is True
    assert config.early_stop.metric == "recall"
    assert config.early_stop.target == 0.75


def test_early_stop_cli_override_replaces_all_configured_values(tmp_path) -> None:
    from cv_agent.cli.main import _build_early_stop_override

    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text("early_stop:\n  enabled: false\n  metric: score\n  target: 1.0\n", encoding="utf-8")

    config = _load_config(
        config_path,
        _build_early_stop_override(enabled=True, metric="mAP50", target=0.75),
    )

    assert config.early_stop.enabled is True
    assert config.early_stop.metric == "mAP50"
    assert config.early_stop.target == 0.75


@pytest.mark.parametrize("metric", ["invalid", "mAP50_class:"])
def test_load_config_rejects_invalid_early_stop_metric(tmp_path, metric) -> None:
    config_path = tmp_path / "cv_agent.yaml"
    config_path.write_text(f"early_stop:\n  metric: '{metric}'\n", encoding="utf-8")

    with pytest.raises(Exception, match="mAP50_class"):
        _load_config(config_path, {})


def test_run_rejects_invalid_early_stop_metric_before_starting_training() -> None:
    result = CliRunner().invoke(cli, ["run", "--early-stop-metric", "invalid"])

    assert result.exit_code == 2
    assert "Invalid value for '--early-stop-metric'" in result.output


def test_run_help_lists_early_stop_options() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "--early-stop" in result.output
    assert "--early-stop-metric" in result.output
    assert "--early-stop-target" in result.output


def test_early_stop_prompt_returns_none_without_tty(monkeypatch) -> None:
    import sys

    from cv_agent.cli import main
    from cv_agent.core.config import EarlyStopConfig

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    assert main._prompt_early_stop_override(EarlyStopConfig(enabled=True, metric="mAP50", target=0.8)) is None


def test_early_stop_prompt_collects_enabled_metric_and_target(monkeypatch) -> None:
    import sys

    from cv_agent.cli import main
    from cv_agent.core.config import EarlyStopConfig

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("cv_agent.ui.prompts.confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr("cv_agent.ui.prompts.select_action", lambda *args, **kwargs: "mAP50")
    monkeypatch.setattr("cv_agent.ui.prompts.text", lambda *args, **kwargs: "0.75")

    result = main._prompt_early_stop_override(EarlyStopConfig())

    assert result == {"early_stop": {"enabled": True, "metric": "mAP50", "target": 0.75}}


def test_early_stop_prompt_can_disable_for_current_run(monkeypatch) -> None:
    import sys

    from cv_agent.cli import main
    from cv_agent.core.config import EarlyStopConfig

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("cv_agent.ui.prompts.confirm", lambda *args, **kwargs: False)

    assert main._prompt_early_stop_override(EarlyStopConfig(enabled=True)) == {"early_stop": {"enabled": False}}


def test_early_stop_prompt_retries_invalid_target(monkeypatch) -> None:
    import sys

    from cv_agent.cli import main
    from cv_agent.core.config import EarlyStopConfig

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("cv_agent.ui.prompts.confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr("cv_agent.ui.prompts.select_action", lambda *args, **kwargs: "mAP50")
    responses = iter(["-0.1", "1.1", "not-a-number", "0.75"])
    monkeypatch.setattr("cv_agent.ui.prompts.text", lambda *args, **kwargs: next(responses))
    warnings = []
    monkeypatch.setattr("cv_agent.cli.main.log_warning", lambda message: warnings.append(message))

    result = main._prompt_early_stop_override(EarlyStopConfig())

    assert result == {"early_stop": {"enabled": True, "metric": "mAP50", "target": 0.75}}
    assert len(warnings) == 3


def test_prompt_start_mode_resume_cancel_exits(monkeypatch, tmp_path):
    import sys

    from cv_agent.cli import main
    from cv_agent.core.config import TrainConfig
    from cv_agent.interaction.types import SessionQuit

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "cv_agent.tracking.checkpoint_manager.list_resumable_runs",
        lambda output_root: [tmp_path / "exp_1"],
    )
    monkeypatch.setattr(
        "cv_agent.ui.prompts.select_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(SessionQuit("cancel")),
    )

    with pytest.raises(SystemExit):
        main._prompt_start_mode(TrainConfig(output_root=tmp_path), "resume", None, None)
