"""CLI entry point for cv_agent.

Provides subcommands:
    run              — Full closed-loop automated training
    validate         — Dataset validation dry run
    resume           — Resume from a prior experiment directory (shortcut)
    list-checkpoints — List saved Top-N and manual checkpoints
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

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
from cv_agent.core.config import EarlyStopConfig, TrainConfig
from cv_agent.ui.console import log_error, log_info, log_success, log_warning, print_banner


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


def _build_data_yaml_override(resolved_data_yaml: Path) -> dict:
    """Build the run-command data override without clobbering local thresholds."""
    return {"data_yaml": resolved_data_yaml}


def _validate_early_stop_metric(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str | None,
) -> str | None:
    """Reject invalid early-stop metrics before dataset setup or training starts."""
    if value is None:
        return None
    try:
        return EarlyStopConfig(metric=value).metric
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


_EARLY_STOP_METRIC_CHOICES = [
    ("mAP50", "mAP@0.5 (recommended)"),
    ("mAP50_95", "mAP@0.5:0.95"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("score", "Composite training score"),
    ("mAP50_class", "mAP@0.5 for a specific class"),
]


def _build_early_stop_override(
    enabled: bool | None,
    metric: str | None,
    target: float | None,
) -> dict:
    """Build a nested early-stop override from optional CLI arguments."""
    if enabled is None and metric is None and target is None:
        return {}

    early_stop: dict[str, bool | str | float] = {}
    if enabled is not None:
        early_stop["enabled"] = enabled
    if metric is not None:
        early_stop["metric"] = metric
    if target is not None:
        early_stop["target"] = target
    if metric is not None or target is not None:
        early_stop["enabled"] = True
    return {"early_stop": early_stop}


def _prompt_early_stop_override(default: EarlyStopConfig) -> dict | None:
    """Collect a per-run early-stop override when stdin is interactive."""
    if not sys.stdin.isatty():
        return None

    from cv_agent.ui.prompts import confirm, select_action, text

    enabled = confirm(
        "Stop training and save the best model when a target metric is reached?",
        default=default.enabled,
    )
    if not enabled:
        return {"early_stop": {"enabled": False}}

    selected_metric = default.metric
    default_key = selected_metric if selected_metric in dict(_EARLY_STOP_METRIC_CHOICES) else "mAP50_class"
    metric = select_action(
        "Choose the early-stop metric:",
        _EARLY_STOP_METRIC_CHOICES,
        default_key=default_key,
    )
    if metric == "mAP50_class":
        class_ref = text(
            "Class name or ID for mAP@0.5:",
            default=selected_metric.split(":", 1)[1] if ":" in selected_metric else "",
        ).strip()
        if not class_ref:
            log_warning("A class name or ID is required for class-specific mAP50; using mAP50 instead.")
            metric = "mAP50"
        else:
            metric = f"mAP50_class:{class_ref}"

    while True:
        raw_target = text("Target value (0 to 1):", default=f"{default.target:g}").strip()
        try:
            target = float(raw_target)
        except ValueError:
            log_warning("Target must be a number from 0 to 1.")
            continue
        if 0.0 <= target <= 1.0:
            return {"early_stop": {"enabled": True, "metric": metric, "target": target}}
        log_warning("Target must be between 0 and 1.")


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
        with open(config_path, encoding="utf-8") as fh:
            config_data = yaml.safe_load(fh) or {}

    # Layer a git-ignored local override file on top (optional; mainly for secrets).
    local_override = config_path.with_suffix(".local.yaml")
    if local_override.is_dir():
        log_warning(
            f"Ignoring {local_override.name}: path is a directory, not a file. "
            "Remove it and create a YAML file, or omit the Docker volume mount."
        )
    elif local_override.is_file():
        log_info(f"Loading local overrides from {local_override.name} (git-ignored).")
        with open(local_override, encoding="utf-8") as fh:
            local_data = yaml.safe_load(fh) or {}
        config_data = _deep_merge(config_data, local_data)

    # Apply CLI overrides on top. Use deep-merge so a nested override like
    # {"data": {"data_yaml": ...}} augments rather than replaces the data
    # section (preserving local threshold overrides such as min_ann_per_class).
    for key, value in cli_overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(config_data.get(key), dict):
            config_data[key] = _deep_merge(config_data[key], value)
        else:
            config_data[key] = value

    try:
        return TrainConfig(**config_data)
    except Exception as e:
        raise click.ClickException(f"Invalid configuration: {e}")


def _check_env() -> None:
    """Verify required packages are importable (conda yolo env check)."""
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as e:
        log_error(
            f"Required dependency not found: {e}\n"
            "  Make sure the 'yolo' conda environment is active:\n"
            "  > conda activate yolo"
        )
        sys.exit(1)


def _prompt_interaction_mode(config_default: str, cli_override: str | None) -> str:
    """Prompt for ask vs auto when running interactively without --interaction."""
    if cli_override:
        return cli_override
    if not sys.stdin.isatty():
        return config_default

    from cv_agent.interaction.types import SessionQuit
    from cv_agent.ui.prompts import select_action

    try:
        return select_action(
            "Choose interaction mode:",
            [
                ("ask", "Ask before edit — confirm rollbacks, params, and changes"),
                ("auto", "Auto — fully autonomous (no prompts)"),
            ],
            default_key=config_default,
        )
    except SessionQuit:
        log_warning("Startup cancelled.")
        sys.exit(0)


StartMode = Literal["fresh", "resume", "from-checkpoint"]


def _prompt_start_mode(
    config: TrainConfig,
    start_override: StartMode | None,
    run_dir_override: Path | None,
    checkpoint_id_override: str | None,
) -> tuple[StartMode, Path | None, str | None]:
    """Prompt for fresh / resume / fork-from-checkpoint startup."""
    if start_override:
        if start_override == "resume" and run_dir_override is None and sys.stdin.isatty():
            from cv_agent.interaction.types import SessionQuit
            from cv_agent.tracking.checkpoint_manager import list_resumable_runs
            from cv_agent.ui.prompts import select_action

            runs = list_resumable_runs(config.output_root)
            if not runs:
                log_warning("No resumable experiments — starting fresh.")
                return "fresh", None, None
            choices = [(str(r), r.name) for r in runs]
            try:
                picked = select_action("Select experiment to resume:", choices, default_key=str(runs[-1]))
            except SessionQuit:
                sys.exit(0)
            return "resume", Path(picked), None
        if start_override == "from-checkpoint" and not checkpoint_id_override:
            raise click.ClickException("--checkpoint-id is required for --start from-checkpoint.")
        return start_override, run_dir_override, checkpoint_id_override

    if not sys.stdin.isatty():
        return "fresh", None, None

    from cv_agent.interaction.types import SessionQuit
    from cv_agent.tracking.checkpoint_manager import list_checkpoints, list_resumable_runs
    from cv_agent.ui.prompts import select_action

    try:
        mode = select_action(
            "Choose how to start training:",
            [
                ("fresh", "New experiment from pretrained weights"),
                ("resume", "Resume an existing experiment (same run directory)"),
                ("from-checkpoint", "New experiment from a saved checkpoint"),
            ],
            default_key="fresh",
        )
    except SessionQuit:
        log_warning("Startup cancelled.")
        sys.exit(0)

    if mode == "fresh":
        return "fresh", None, None

    if mode == "resume":
        runs = list_resumable_runs(config.output_root)
        if not runs:
            log_warning("No resumable experiments found — starting fresh.")
            return "fresh", None, None
        choices = [(str(r), f"{r.name}") for r in runs]
        try:
            picked = select_action("Select experiment to resume:", choices, default_key=str(runs[-1]))
        except SessionQuit:
            sys.exit(0)
        return "resume", Path(picked), None

    # from-checkpoint
    all_ckpt = list_checkpoints(config.output_root)
    forkable = [c for c in all_ckpt if c.kind in ("top", "manual")]
    if not forkable:
        log_warning("No saved checkpoints found — starting fresh from pretrained weights.")
        return "fresh", None, None
    choices = [(c.id, c.label) for c in forkable]
    try:
        picked_id = select_action("Select checkpoint to fine-tune from:", choices, default_key=forkable[0].id)
    except SessionQuit:
        sys.exit(0)
    return "from-checkpoint", None, picked_id


def _execute_training_start(
    config: TrainConfig,
    start_mode: StartMode,
    run_dir: Path | None,
    checkpoint_id: str | None,
) -> None:
    """Dispatch to fresh run, resume, or fork-from-checkpoint."""
    from cv_agent.core.engine import TrainingEngine
    from cv_agent.tracking.checkpoint_manager import find_checkpoint_by_id

    engine = TrainingEngine()

    if start_mode == "resume":
        if run_dir is None:
            raise click.ClickException("Resume requires --run-dir.")
        engine.resume(run_dir.resolve(), config)
        return

    if start_mode == "from-checkpoint":
        if not checkpoint_id:
            raise click.ClickException("from-checkpoint requires --checkpoint-id.")
        info = find_checkpoint_by_id(config.output_root, checkpoint_id)
        if info is None:
            raise click.ClickException(f"Checkpoint not found: {checkpoint_id}")
        engine.run_from_checkpoint(config, info)
        return

    engine.run(config)


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
def cli(ctx: click.Context, config: Path, interaction: str | None) -> None:
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
              help="Override model variant (e.g., yolo26s, yolov8n).")
@click.option(
    "--start",
    type=click.Choice(["fresh", "resume", "from-checkpoint"]),
    default=None,
    help="Startup mode (default: interactive prompt or fresh).",
)
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Run directory for --start resume.")
@click.option("--checkpoint-id", type=str, default=None,
              help="Checkpoint id for --start from-checkpoint (see list-checkpoints).")
@click.option("--device", type=str, default=None,
              help="CUDA device(s): auto, cpu, 0, or 0,1,2,3 (DDP). Overrides config.")
@click.option("--early-stop", is_flag=True, default=None,
              help="Stop after a target metric is reached and save the best model.")
@click.option("--early-stop-metric", type=str, default=None, callback=_validate_early_stop_metric,
              help="Target metric: score, mAP50, mAP50_95, precision, recall, or mAP50_class:<name-or-id>.")
@click.option("--early-stop-target", type=click.FloatRange(0.0, 1.0), default=None,
              help="Target metric value from 0 to 1; also enables early stopping.")
@click.pass_context
def run(
    ctx: click.Context,
    optimize_for: str | None,
    max_rounds: int | None,
    data_yaml: Path | None,
    model: str | None,
    start: str | None,
    run_dir: Path | None,
    checkpoint_id: str | None,
    device: str | None,
    early_stop: bool | None,
    early_stop_metric: str | None,
    early_stop_target: float | None,
) -> None:
    """Start automated closed-loop training.

    Runs the full training loop: data validation, YOLO training,
    evaluation, decision, hyperparameter mutation, repeat.
    """
    _check_env()
    print_banner(__version__)

    # Resolve dataset from CLI or config, then bootstrap/download if needed.
    from cv_agent.data.bootstrap import ensure_dataset

    pre_config = _load_config(ctx.obj["config_path"], {})
    requested_data_yaml = data_yaml if data_yaml is not None else pre_config.data.data_yaml
    resolved_data_yaml = ensure_dataset(requested_data_yaml)

    # Build CLI overrides dict (interaction mode prompted after base config load)
    overrides: dict = {}
    if optimize_for:
        overrides["optimize_for_class"] = optimize_for
    if max_rounds:
        overrides["max_rounds"] = max_rounds
    overrides["data"] = _build_data_yaml_override(resolved_data_yaml)
    if model:
        overrides["model_variant"] = model
    if device:
        overrides["device"] = device
    overrides = _deep_merge(
        overrides,
        _build_early_stop_override(early_stop, early_stop_metric, early_stop_target),
    )

    base_config = _load_config(ctx.obj["config_path"], overrides)
    interaction_mode = _prompt_interaction_mode(
        base_config.interaction_mode,
        ctx.obj.get("interaction_override"),
    )
    has_early_stop_cli_override = any(
        value is not None for value in (early_stop, early_stop_metric, early_stop_target)
    )
    if not has_early_stop_cli_override:
        interactive_override = _prompt_early_stop_override(base_config.early_stop)
        if interactive_override is not None:
            overrides = _deep_merge(overrides, interactive_override)
            base_config = _load_config(ctx.obj["config_path"], overrides)
    config = base_config.model_copy(update={"interaction_mode": interaction_mode})

    log_info(f"Configuration loaded: model={config.model_variant}, "
             f"interaction={config.interaction_mode}, max_rounds={config.max_rounds}")
    log_info(f"Dataset: {config.data.data_yaml}")

    if config.optimize_for_class:
        log_info(f"Optimizing for class: [bold cyan]{config.optimize_for_class}[/bold cyan]")

    if config.early_stop.enabled:
        log_info(
            f"Early stop: {config.early_stop.metric} >= {config.early_stop.target:g}; "
            "best model will be exported to <run_dir>/final/best.pt."
        )
    else:
        log_info("Early stop: disabled.")

    start_mode, resume_dir, ckpt_id = _prompt_start_mode(
        config,
        start_override=start,
        run_dir_override=run_dir,
        checkpoint_id_override=checkpoint_id,
    )

    if start_mode == "resume" and resume_dir is None and run_dir:
        resume_dir = run_dir

    _execute_training_start(config, start_mode, resume_dir, ckpt_id or checkpoint_id)


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

    config = _load_config(ctx.obj["config_path"], {"data": _build_data_yaml_override(data_yaml)})
    data_config: DataConfig = config.data

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

    base_config = _load_config(ctx.obj["config_path"], {})
    interaction_mode = _prompt_interaction_mode(
        base_config.interaction_mode,
        ctx.obj.get("interaction_override"),
    )
    config = base_config.model_copy(update={"interaction_mode": interaction_mode})

    _execute_training_start(config, "resume", run_dir.resolve(), None)


# ---------------------------------------------------------------------------
# cv_agent list-checkpoints
# ---------------------------------------------------------------------------

@cli.command("list-checkpoints")
@click.pass_context
def list_checkpoints_cmd(ctx: click.Context) -> None:
    """List Top-N, manual, and resumable experiment checkpoints."""
    config = _load_config(ctx.obj["config_path"], {})
    from cv_agent.tracking.checkpoint_manager import list_checkpoints
    from rich.table import Table
    from cv_agent.ui.console import console

    entries = list_checkpoints(config.output_root)
    if not entries:
        log_info("No checkpoints found under output_root.")
        return

    table = Table(title="Saved checkpoints", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Kind")
    table.add_column("Score")
    table.add_column("Round")
    table.add_column("Label")

    for e in entries:
        table.add_row(
            e.id,
            e.kind,
            f"{e.score:.4f}",
            str(e.round),
            e.label,
        )
    console.print(table)
