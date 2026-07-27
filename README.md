# cv_agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./pyproject.toml)

[中文文档](./README.zh-CN.md)

`cv_agent` is a command-line agent for iterative YOLO object-detection training. It validates a dataset, trains and evaluates a model, decides how the round performed, proposes the next hyperparameters, and repeats—all while preserving experiment state and artifacts.

It is intended for experiments where a single `yolo train` invocation is not enough: you want controlled iteration, reproducible decisions, checkpoints, and a way to resume or branch an experiment.

## What it does

```text
validate data → train a round → evaluate → decide → adjust → next round
                                      │
                         Green / Yellow / Red
```

- Runs YOLO training in rounds rather than as one opaque job.
- Scores each round using validation metrics, with optional emphasis on one class.
- Uses rule-based Green / Yellow / Red decisions for checkpointing, recovery, and local-optimum escape.
- Uses Optuna to propose bounded hyperparameter changes.
- Supports interactive review (`ask`) and unattended execution (`auto`).
- Keeps Top-N and manual checkpoints; supports resume and fork-from-checkpoint workflows.
- Validates datasets before training and produces a data-gap report after repeated failures.
- Logs locally and to MLflow when available; an unreachable HTTP MLflow server falls back to `./mlruns`.
- Optionally uses an OpenAI-compatible LLM endpoint for Ask-mode guidance and data-gap analysis. No API key is required for the core loop.

## Requirements

- Python 3.10 or later.
- PyTorch compatible with the target CPU/GPU and CUDA installation.
- An Ultralytics-compatible environment. A CUDA GPU is strongly recommended for practical training.
- Docker with NVIDIA Container Toolkit only if you choose the Docker workflow.

The project declares its Python dependencies in `pyproject.toml`, including Ultralytics, PyTorch, Optuna, MLflow, and Rich.

## Install

```bash
git clone <repository-url>
cd cv_agent

# Activate an environment that has the appropriate PyTorch build for your hardware.
python -m pip install -e .

# Optional developer dependencies
python -m pip install -e ".[dev]"
```

Both commands below invoke the same CLI:

```bash
cv_agent --help
cvagent --help
```

## Quick start

The tracked profiles serve different purposes:

| Profile | Purpose | Dataset | Default model |
| --- | --- | --- | --- |
| `cv_agent.quick.yaml` | Smoke-test the complete loop | COCO128 | `yolo26n` |
| `cv_agent.yaml` | Longer formal experiments | COCO | `yolo26s` |

Start with the quick profile. If the configured Ultralytics registry dataset is absent, `cv_agent` asks Ultralytics to bootstrap it; the initial download can take time and needs network access.

```bash
cv_agent --config cv_agent.quick.yaml run
```

To run against your own dataset, first create a YOLO dataset YAML. `dataset.yaml.example` is a starting point.

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

Then launch a bounded experiment:

```bash
cv_agent run \
  --data-yaml /absolute/path/to/dataset.yaml \
  --model yolo26n \
  --max-rounds 5
```

Validate a dataset without training:

```bash
cv_agent validate --data-yaml /absolute/path/to/dataset.yaml
```

## Choose an interaction mode

`ask` is the default in `cv_agent.yaml` and is best for an interactive terminal. It lets you review a proposed change, add guidance, reject a rollback, or save a named checkpoint at decision time.

`auto` is suitable for CI, Docker, `tmux`, or an unattended server. Decisions are accepted automatically after the configured countdown. Auto mode does not ask for per-round guidance.

```bash
# Interactive review
cv_agent --interaction ask run --data-yaml dataset.yaml

# Unattended run
cv_agent --interaction auto run --data-yaml dataset.yaml --max-rounds 20
```

## How decisions work

The first completed round establishes the run-local baseline. Later rounds are compared with the best historical score from the same experiment. The decision engine uses configured relative and optional absolute thresholds; with `dynamic_thresholds: true`, thresholds tighten as the run progresses.

| Outcome | Meaning | Typical result |
| --- | --- | --- |
| Green | Meaningful improvement, or an accepted marginal improvement | Commit the improved checkpoint; make an Optuna proposal when applicable. |
| Yellow | Near the baseline / possible local optimum | Make a conservative diagnostic adjustment or use the configured escape strategy. |
| Red | Performance degradation | Apply recovery; hard Red may roll back to the best checkpoint. |

After `red_escalation_count` consecutive Red outcomes (default: 3), the agent forces recovery, generates `data_gap_report.md` and `data_gap_report.json`, applies bounded loss-weight suggestions when available, and enters data-supplement handling.

This classification is deterministic and metric-driven. An LLM can suggest bounded strategy patches or interpret user guidance, but it does not replace evaluation metrics or make checkpoint decisions.

## Resume, checkpoints, and outputs

Each experiment is written under `runs/exp_<timestamp>/`. Important contents include:

| Path | Contents |
| --- | --- |
| `weights/` | Current YOLO weights and saved best snapshots. |
| `checkpoints/` | Top-N leaderboard entries and manual checkpoints. |
| `optuna_study.db` | Per-experiment Optuna study. |
| `session_state.json` | Round number, parameters, best score, Red streak, and trial count needed for resume. |
| `cv_agent.log` | Session log. |
| `decision_log.json` / strategy artifacts | Decision and strategy audit trail. |
| `final/best.pt` | Exported best model, with `final/summary.json`. |

List available checkpoints:

```bash
cv_agent list-checkpoints
```

Resume an interrupted experiment:

```bash
cv_agent resume --run-dir runs/exp_<timestamp>
# Equivalent explicit form
cv_agent run --start resume --run-dir runs/exp_<timestamp>
```

Create a new experiment from a saved checkpoint:

```bash
cv_agent list-checkpoints
cv_agent run --start from-checkpoint --checkpoint-id <checkpoint-id>
```

Early stopping exports the best model when the target is met:

```bash
cv_agent run \
  --data-yaml dataset.yaml \
  --early-stop \
  --early-stop-metric mAP50 \
  --early-stop-target 0.75
```

Supported early-stop metrics are `score`, `mAP50`, `mAP50_95`, `precision`, `recall`, and `mAP50_class:<class-name-or-id>`.

## Configuration

Configuration is YAML-based. Precedence is:

1. Selected profile (`--config`, default `cv_agent.yaml`)
2. Sibling local override (`<profile-name>.local.yaml`, if present)
3. CLI options

For the default profile, the local override is `cv_agent.local.yaml`. It is ignored by Git and should contain secrets or machine-specific overrides only. Do not put credentials in tracked configuration files.

```bash
cp cv_agent.local.yaml.example cv_agent.local.yaml
```

```yaml
# cv_agent.local.yaml
device: "0"
workers: 0
llm:
  api_key: ""
```

The most useful top-level settings are:

| Setting | Purpose |
| --- | --- |
| `model_variant`, `epochs_per_round`, `max_rounds` | Training scope. |
| `device`, `workers`, `model_verbose` | Hardware and runtime behavior. |
| `data` | Dataset YAML and validation thresholds. |
| `initial_hyperparams` | Starting YOLO training parameters. |
| `optuna` | Trial budget, Yellow escape strategy, and search ranges. |
| `decision` | Green / Yellow / Red and escalation thresholds. |
| `strategy` | Strategy-planner cadence, memory, and objective weights. |
| `checkpoints`, `output_root` | Artifact retention and destination. |
| `early_stop` | Stop-and-export target. |

Supported model identifiers are `yolo26{n,s,m,l,x}`, `yolov8{n,s,m,l,x}`, and `yolo11{n,s,m,l,x}`.

Use `--device auto`, `--device 0`, `--device 0,1,2,3`, or `--device cpu` to override device selection. On Windows, setting `workers: 0` is often the safest starting point.

### Prioritize one class

Use `--optimize-for` to weight one class's mAP@0.5 more heavily in the reward score.

```bash
cv_agent run --data-yaml dataset.yaml --optimize-for vehicle
```

The class must match a name in the dataset YAML. If it cannot be resolved, the agent warns and uses global metrics.

## LLM integration

LLM support is optional. The built-in adapter expects an OpenAI-compatible API and defaults to a DeepSeek endpoint in the provided profiles. Supply the key through an environment variable where possible:

```bash
export CV_AGENT_LLM_KEY="<your-key>"
# DEEPSEEK_API_KEY is also recognized
```

Ask mode can use the LLM to interpret natural-language constraints such as “only adjust lr” or “keep mosaic”. Regex and heuristic fallback handling remain available when no key is configured. API calls are bounded by `llm.max_calls_per_session`.

## MLflow

Set `mlflow_uri` and `experiment_name` in the selected profile to use a tracking server. For an HTTP(S) URI, the agent checks reachability before training. If it is unavailable, it uses a local file store at `./mlruns` instead; training continues and artifacts remain in the run directory.

To inspect local tracking data:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Docker

The supplied Dockerfile uses the Ultralytics image and exposes the CLI as its entrypoint.

```bash
docker build -t cv_agent:latest .

docker run --rm -it --gpus '"device=0"' \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/cv_agent.quick.yaml:/app/cv_agent.quick.yaml:ro" \
  cv_agent:latest --config cv_agent.quick.yaml run
```

For a custom dataset, mount it and use paths valid inside the container:

```bash
docker run --rm -it --gpus '"device=0"' \
  -v "$(pwd)/runs:/app/runs" \
  -v "/host/path/to/dataset:/data:ro" \
  -v "$(pwd)/dataset.yaml:/app/dataset.yaml:ro" \
  cv_agent:latest --interaction auto run --data-yaml /app/dataset.yaml --device 0
```

For multi-GPU DDP, expose the desired GPUs and provide shared memory, for example `--ipc=host` or `--shm-size=8g`. Do not mount `cv_agent.local.yaml` unless the file exists on the host; Docker otherwise creates a directory at that path.

## Development

```bash
python -m pytest
ruff check src tests
```

The test suite covers configuration, the decision engine, optimizer flow, state persistence, checkpoint handling, CLI behavior, and terminal presentation.

## Project layout

```text
src/cv_agent/
  cli/          Command-line entry points
  core/         Configuration, state machine, training orchestrator
  data/         Dataset bootstrap, validation, supplement and gap reports
  decision/     Decisions, Optuna, strategy and LLM guidance
  interaction/  Ask and auto workflows
  tracking/     Run directories, checkpoints and MLflow
  trainer/      YOLO execution, evaluation, devices and early stopping
tests/          Automated tests
```

## License

MIT. See the package metadata in [pyproject.toml](./pyproject.toml).
