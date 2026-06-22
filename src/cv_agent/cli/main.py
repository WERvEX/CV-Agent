"""CLI entry point for cv_agent.

Provides three subcommands:
    run       — Full closed-loop automated training
    validate  — Dataset validation dry run
    resume    — Resume from a prior experiment directory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Force unbuffered/line-buffered stdout BEFORE heavy imports, so Ultralytics'
# tqdm/print flush in real time during training (not just when a buffer fills).
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # reconfigure unavailable on this stream

import click
import yaml

from cv_agent import __version__
from cv_agent.core.config import TrainConfig
from cv_agent.ui.console import console, log_error, log_info, log_success, log_warning, print_banner


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins).

    Used to layer a git-ignored ``cv_agent.local.yaml`` (which may carry real
    secrets) on top of the tracked ``cv_agent.yaml`` template.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(config_path: Path, cli_overrides: dict) -> TrainConfig:
    """Load TrainConfig from YAML file and apply CLI overrides.

    A sibling ``<name>.local.yaml`` (e.g. ``cv_agent.local.yaml``) is merged on
    top of ``config_path`` if present. This local file is git-ignored and is the
    intended place to put secrets like the LLM API key — it can never be pushed.

    Args:
        config_path: Path to the YAML configuration file.
        cli_overrides: Dictionary of field paths to override values.

    Returns:
        Validated TrainConfig instance.

    Raises:
        click.ClickException: If the config file is missing or invalid.
    """
    if not config_path.exists():
        if config_path.name == "cv_agent.yaml":
            log_warning(f"No config file found at {config_path}, using defaults.")
            config_data = {}
        else:
            raise click.ClickException(f"Config file not found: {config_path}")
    else:
        with open(config_path, "r", encoding="utf-8") as fh:
            config_data = yaml.safe_load(fh) or {}

    # Layer a git-ignored local override file on top (secrets go here).
    local_override = config_path.with_suffix(".local.yaml")
    if local_override.exists():
        log_info(f"Loading local overrides from {local_override.name} (git-ignored).")
        with open(local_override, "r", encoding="utf-8") as fh:
            local_data = yaml.safe_load(fh) or {}
        config_data = _deep_merge(config_data, local_data)

    # Apply CLI overrides to top-level config dict
    for key, value in cli_overrides.items():
        if value is not None:
            config_data[key] = value

    try:
        return TrainConfig(**config_data)
    except Exception as e:
        raise click.ClickException(f"Invalid configuration: {e}")


def _check_env() -> None:
    """Verify required packages are importable (conda yolo env check)."""
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        log_error(
            f"Required dependency not found: {e}\n"
            "  Make sure the 'yolo' conda environment is active:\n"
            "  > conda activate yolo"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Click CLI group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.option("--config", "-c", type=click.Path(exists=False, path_type=Path), default=Path("cv_agent.yaml"),
              help="Path to YAML config file.")
@click.option("--interaction", type=click.Choice(["auto", "ask"]), default=None,
              help="Override interaction mode.")
@click.version_option(version=__version__, prog_name="cv_agent", message="%(prog)s v%(version)s")
@click.pass_context
def cli(ctx: click.Context, config: Path, interaction: Optional[str]) -> None:
    """cv_agent — Automated Closed-Loop YOLO Training CLI.

    Combines Ultralytics YOLO, Optuna hyperparameter optimization,
    MLflow experiment tracking, and LLM-based strategic reasoning.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["interaction_override"] = interaction


# ---------------------------------------------------------------------------
# cv_agent run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--optimize-for", type=str, default=None,
              help="Class name to prioritize in reward function (e.g., 'vehicle').")
@click.option("--max-rounds", type=int, default=None,
              help="Override max training rounds from config.")
@click.option("--data-yaml", type=click.Path(exists=False, path_type=Path), default=None,
              help="Override dataset YAML path. If omitted or missing, COCO128 is downloaded for first run.")
@click.option("--model", type=str, default=None,
              help="Override model variant (e.g., yolov8s).")
@click.pass_context
def run(
    ctx: click.Context,
    optimize_for: Optional[str],
    max_rounds: Optional[int],
    data_yaml: Optional[Path],
    model: Optional[str],
) -> None:
    """Start automated closed-loop training.

    Runs the full training loop: data validation, YOLO training,
    evaluation, decision, hyperparameter mutation, repeat.
    """
    _check_env()
    print_banner(__version__)

    # Bootstrap a dataset if none was provided (or the path is missing).
    # Falls back to downloading COCO128 so the loop can run out of the box.
    from cv_agent.data.bootstrap import ensure_dataset
    resolved_data_yaml = ensure_dataset(data_yaml)

    # Build CLI overrides dict
    overrides = {}
    if ctx.obj.get("interaction_override"):
        overrides["interaction_mode"] = ctx.obj["interaction_override"]
    if optimize_for:
        overrides["optimize_for_class"] = optimize_for
    if max_rounds:
        overrides["max_rounds"] = max_rounds
    # Merge into the existing `data` section so we don't clobber thresholds
    # (min_images, min_ann_per_class, ...) defined in cv_agent.yaml.
    existing_data = {}
    cfg_path = ctx.obj["config_path"]
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            existing_data = (yaml.safe_load(fh) or {}).get("data", {}) or {}
    overrides["data"] = {**existing_data, "data_yaml": resolved_data_yaml}
    if model:
        overrides["model_variant"] = model

    config = _load_config(ctx.obj["config_path"], overrides)

    log_info(f"Configuration loaded: model={config.model_variant}, "
             f"interaction={config.interaction_mode}, max_rounds={config.max_rounds}")
    log_info(f"Dataset: {config.data.data_yaml}")

    if config.optimize_for_class:
        log_info(f"Optimizing for class: [bold cyan]{config.optimize_for_class}[/bold cyan]")

    # Import here to avoid heavy deps on --help
    from cv_agent.core.engine import TrainingEngine

    engine = TrainingEngine()
    engine.run(config)


# ---------------------------------------------------------------------------
# cv_agent validate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data-yaml", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to dataset YAML to validate.")
@click.pass_context
def validate(ctx: click.Context, data_yaml: Path) -> None:
    """Run dataset validation only (dry run, no training).

    Checks image count, annotation completeness, class distribution,
    and optional quality metrics (brightness, angles, object sizes).
    """
    _check_env()
    print_banner(__version__)

    from cv_agent.core.config import DataConfig
    from cv_agent.data.validator import DatasetValidator

    data_config = DataConfig(data_yaml=data_yaml)

    # Partially load main config for validation thresholds
    config_path = ctx.obj["config_path"]
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config_data = yaml.safe_load(fh) or {}
        if "data" in config_data:
            data_config = DataConfig(**{**data_config.model_dump(), **config_data["data"]})

    validator = DatasetValidator(data_config)
    issues = validator.validate()

    if issues:
        from cv_agent.ui.console import print_validation_issues
        print_validation_issues([i.model_dump() for i in issues])
        log_warning(f"Found {len(issues)} validation issue(s).")
    else:
        log_success("Dataset validation passed — all checks OK.")


# ---------------------------------------------------------------------------
# cv_agent resume
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--run-dir", "-r", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Path to a prior experiment directory to resume from.")
@click.pass_context
def resume(ctx: click.Context, run_dir: Path) -> None:
    """Resume training from a prior experiment directory.

    Restores Optuna study state, reloads configuration from the run's
    args.yaml, and continues from the last logged round.
    """
    _check_env()
    print_banner(__version__)

    log_info(f"Resuming from: {run_dir}")

    from cv_agent.core.engine import TrainingEngine

    engine = TrainingEngine()
    engine.resume(run_dir)