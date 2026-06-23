from __future__ import annotations

from pathlib import Path

from cv_agent.cli.main import _build_data_yaml_override, _load_config


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
