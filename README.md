# cv_agent

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20search-green.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](pyproject.toml)

**Automated closed-loop YOLO object-detection training.**

`cv_agent` runs repeated training rounds: validate data → train → evaluate → classify the outcome (Green / Yellow / Red) → propose next hyperparameters → repeat. [Optuna](https://optuna.org/) drives hyperparameter search; a rule-based controller handles checkpoint commits, rollbacks, and local-optimum escape. An optional LLM parses natural-language guidance in **Ask mode**, and assists with data-gap analysis after repeated failures.

Run fully unattended (`auto`) on servers, or stay in the loop (`ask`) locally with diffs, confirmations, and guidance feedback.

Command alias: `cvagent` and `cv_agent` are equivalent console scripts. This README keeps both visible where it matters; use whichever is easier in your shell.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [Running on a Server](#running-on-a-server)
  - [Docker (recommended)](#docker-recommended)
- [CLI Reference](#cli-reference)
- [Interaction Modes](#interaction-modes)
- [Decision System](#decision-system)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [AI Strategy Planner](#ai-strategy-planner)
- [Checkpoints & Resume](#checkpoints--resume)
- [Model Weights & Export](#model-weights--export)
- [Dataset Validation & Supplement](#dataset-validation--supplement)
- [Single-Class Optimization](#single-class-optimization)
- [Configuration Reference](#configuration-reference)
- [LLM Integration](#llm-integration)
- [MLflow Tracking](#mlflow-tracking)
- [Project Layout](#project-layout)
- [Development](#development)
- [Security](#security)
- [License](#license)

---

## Features

- **Closed-loop training** — multi-round train → evaluate → decide without manual scripting
- **Tiered decisions** — hard/marginal Green, diagnostic Yellow, soft/hard Red with overfit/underfit-aware recovery
- **Optuna integration** — TPE Bayesian on Green; random walk / simulated annealing / Bayesian on Yellow
- **Ask-mode LLM guidance** — natural-language feedback (e.g. “lr 稍微大一点”) parsed into param constraints with a visible diff before the next round
- **Per-class metrics** — compact per-class mAP summary after validation enrichment
- **Checkpoint management** — Top-N leaderboard, per-round snapshots, manual named saves, resume & fork
- **Dataset validation** — image/label checks, class counts, object size, optional quality heuristics
- **Data supplement mode** — helper scripts when data is insufficient or after 3× Red escalation
- **Ask or Auto** — human-in-the-loop with param diffs, or hands-off overnight runs
- **MLflow logging** — remote server or transparent local fallback
- **Resume persistence** — `session_state.json` stores round, hyperparameters, `red_streak`, and Optuna trial budget

---

## How It Works

Each **round** trains for `epochs_per_round` epochs, then evaluates against the historical best score:

```
VALIDATE → TRAIN → EVALUATE → DECIDE → (next round)
                              ↓
                    Green / Yellow / Red
                              ↓
              Optuna proposal + rule recovery
                              ↓
              Ask review / Auto approve → apply params
```

| Component | Responsibility |
|-----------|----------------|
| `TrainingEngine` | Main loop, round lifecycle, artifact I/O |
| `ThreeStateDecisionEngine` | Tiered Green / Yellow / Red classification |
| `OptunaOptimizer` | Next-round hyperparameter proposals |
| `LLMAdvisor` | Ask-mode guidance parsing; Red×3 data-gap analysis |
| `CheckpointManager` | Top-N saves, manual saves, fork metadata |
| `Evaluator` | mAP scoring, overfit/underfit detection, per-class metrics |

**Reward score** defaults to global mAP@0.5; with `--optimize-for`, a weighted blend favors one class (see [Single-Class Optimization](#single-class-optimization)).

---

## Requirements

- **Python** 3.10+
- **PyTorch** + **Ultralytics YOLO** (GPU recommended)
- **MLflow server** — optional (`./mlruns` fallback)
- **LLM API key** — optional (regex guidance + heuristic fallback work without it)

The CLI checks that `torch` and `ultralytics` are importable at startup. Use a dedicated conda/venv with those packages installed.

---

## Installation

```bash
git clone <your-repo-url>
cd cv_agent

conda activate <your-yolo-env>   # e.g. a env with torch + ultralytics

pip install -e .

# Optional: dev tools
pip install -e ".[dev]"
```

**Server (Docker)** — see [Docker (recommended)](#docker-recommended) under [Running on a Server](#running-on-a-server).

---

## Quick Start

### First run (full COCO)

Defaults in `cv_agent.yaml` target **formal training**: `yolo26s` on full **COCO** (`coco.yaml`), 50 epochs/round × 6 rounds.

```bash
cvagent run
```

On first run, Ultralytics **auto-downloads** `yolo26s.pt` (~20 MB) and COCO (~20 GB) into `datasets/`. If the default Ultralytics CDN is slow, **prefetch on the host** first (GitHub mirror):

```bash
bash scripts/prefetch_coco.sh                    # default: GitHub mirror for labels
LABEL_MIRROR=ghfast bash scripts/prefetch_coco.sh   # China-friendly GitHub proxy
SKIP_IMAGES=1 bash scripts/prefetch_coco.sh      # labels only (~168 MB), then images later
```

Files land in `./datasets/`; Docker mount `-v "$(pwd)/datasets:/app/datasets"` reuses them.

**Model weights:** Ultralytics downloads pretrained `.pt` on first train. Prefetch on the host (includes `yolo26n.pt` for AMP checks — **not** your training model):

```bash
bash scripts/prefetch_weights.sh
# Docker: also mount -v "$(pwd)/weights/yolo26n.pt:/app/yolo26n.pt:ro" etc.
```

**Quick functional test** (~minutes, not hours) — use bundled `cv_agent.quick.yaml`:

| Profile | Dataset | Model | Epochs/round | Rounds | ~Train steps/epoch |
|---------|---------|-------|--------------|--------|-------------------|
| `cv_agent.quick.yaml` | COCO128 (128 img) | yolo26n | 5 | 3 | 8 |
| `cv_agent.yaml` (default) | COCO (~118k img) | yolo26s | 50 | 6 | ~1849 |

Full COCO is the main time cost; switching `yolo26s` → `yolo26n` only helps modestly.

```bash
cvagent --config cv_agent.quick.yaml run
```

Docker (single GPU is enough for smoke test):

```bash
docker run -dit --gpus '"device=0"' --name cv_agent_quick \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/datasets:/datasets" \
  -v "$(pwd)/weights:/app/weights:ro" \
  -v "$(pwd)/cv_agent.quick.yaml:/app/cv_agent.quick.yaml:ro" \
  cv_agent:latest --config cv_agent.quick.yaml run
docker attach cv_agent_quick
```

Or override flags on the default config:

```bash
cvagent run --data-yaml coco128.yaml --model yolo26n --max-rounds 3
```

Add `epochs_per_round: 5` in `cv_agent.local.yaml` for shorter smoke rounds (see example file).

### Train on your dataset

```bash
cv_agent run \
  --data-yaml dataset.yaml \
  --model yolo26s \
  --max-rounds 10
```

### Validate only

```bash
cv_agent validate --data-yaml dataset.yaml
```

---

## Running Locally

Use **Ask mode** (default) when you have an interactive terminal and want to review each round.

```bash
# Interactive startup wizard (fresh / resume / from-checkpoint)
cv_agent run

# Explicit options
cv_agent run --interaction ask --data-yaml dataset.yaml --model yolo26n --max-rounds 5

# Resume the same experiment
cv_agent resume --run-dir runs/exp_<timestamp>

# Fork a new experiment from a saved checkpoint
cv_agent list-checkpoints
cv_agent run --start from-checkpoint --checkpoint-id <ID>
```

**Local overrides** — copy the example and edit (git-ignored, never committed):

```bash
cp cv_agent.local.yaml.example cv_agent.local.yaml
```

Typical local `cv_agent.local.yaml` tweaks: shorter smoke runs, Ask mode, dataset path.

After each round in Ask mode you can:

- Approve or reject the controller / Optuna proposal
- Add **guidance** (natural language or regex phrases like `only lr`, `keep mosaic`)
- See a **Guidance Applied** panel with parameter diff before continuing
- Manually save a named checkpoint at the DECIDE prompt

---

## Running on a Server

Use **Auto mode** when there is no TTY (Docker, `nohup`, cron, SSH batch jobs).

### Docker (recommended)

On a GPU server with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed:

```bash
# Clone on the server (or copy the repo)
git clone <your-repo-url>
cd cv_agent

# Build image (bundles cv_agent.yaml; local overrides are optional)
docker build -t cv_agent:latest .
mkdir -p runs datasets
```

**Config:** defaults come from **`cv_agent.yaml`** (inside the image). Mount host config if you changed it: `-v "$(pwd)/cv_agent.yaml:/app/cv_agent.yaml:ro"`.

**Multi-GPU (recommended on 8-GPU servers)** — expose 4 GPUs; add **`--shm-size=8g`** (NCCL/DDP needs more than Docker’s default 64MB `/dev/shm`):

```bash
docker run -dit --gpus '"device=0,1,2,3"' --shm-size=8g --name cv_agent_train \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/datasets:/datasets" \
  -v "$(pwd)/weights:/app/weights:ro" \
  -v "$(pwd)/cv_agent.yaml:/app/cv_agent.yaml:ro" \
  -e CV_AGENT_LLM_KEY="${CV_AGENT_LLM_KEY:-}" \
  cv_agent:latest run --data-yaml /app/datasets/coco_runner.yaml
```

If NCCL still fails, try `--ipc=host` instead of/in addition to `--shm-size=8g`.

**Single GPU:**

```bash
docker run -dit --gpus '"device=0"' --name cv_agent_train \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/datasets:/datasets" \
  -v "$(pwd)/cv_agent.yaml:/app/cv_agent.yaml:ro" \
  cv_agent:latest run --data-yaml /app/datasets/coco_runner.yaml --device 0
```

`device` in yaml: `auto` | `0` | `0,1,2,3` | `cpu`. CLI: `run --device 0,1,2,3`.

| Mount / env | Purpose |
|-------------|---------|
| `-v .../runs:/app/runs` | Training artifacts, checkpoints, Optuna DB, `session_state.json` |
| `-v .../datasets:/app/datasets` | Persist COCO / other Ultralytics auto-downloads (~20 GB for COCO) |
| `-v .../cv_agent.local.yaml:...:ro` | **Optional** — only when the file exists; overrides selected keys |
| `-v /host/datasets:/data:ro` | **Custom** dataset only — paths in yaml must use `/data/...` |
| `-e CV_AGENT_LLM_KEY=...` | LLM key (preferred over putting secrets in local yaml) |
| `--gpus '"device=0"'` | Expose one GPU on multi-GPU hosts |
| `--shm-size=8g` | **Required for multi-GPU DDP** — avoids NCCL `No space left on device` on `/dev/shm` |

> **Tip:** Paths in `dataset.yaml` must match **inside the container** (e.g. `path: /data/dataset`). See `dataset.yaml.example`.

### 1. Prepare config (bare-metal / venv)

Uses **`cv_agent.yaml` by default**. Create **`cv_agent.local.yaml` only if** you need to override keys or store a secret:

```bash
cp cv_agent.local.yaml.example cv_agent.local.yaml   # optional
```

Example local override (secrets only):

```yaml
llm:
  api_key: "sk-..."   # or: export CV_AGENT_LLM_KEY="sk-..."
```

For non-interactive servers, pass `--interaction auto` or uncomment `interaction_mode: auto` in local yaml.

### 2. Set secrets via environment (recommended on servers)

```bash
export CV_AGENT_LLM_KEY="sk-..."    # optional
export DEEPSEEK_API_KEY="sk-..."   # alternative
```

### 3. Run

```bash
# Foreground (attach to tmux/screen first for long jobs)
cv_agent run --interaction auto --max-rounds 10

# Background with log file
nohup cv_agent run --interaction auto \
  --data-yaml dataset.yaml \
  --model yolo26s \
  --max-rounds 20 \
  > train.log 2>&1 &
```

### 4. Monitor

- Tail the log: `tail -f train.log` or `tail -f runs/exp_<timestamp>/cv_agent.log`
- MLflow UI (if configured): `mlflow ui --host 0.0.0.0 --port 5000`
- Artifacts under `runs/exp_<timestamp>/`

### 5. Resume after interruption

```bash
cv_agent resume --run-dir runs/exp_<timestamp>
# or
cv_agent run --start resume --run-dir runs/exp_<timestamp> --interaction auto
```

Resume restores round number, hyperparameters, Optuna study, `red_streak`, and trial budget from `session_state.json`.

> **Note:** Auto mode does not call the LLM for per-round guidance (no user input). LLM still runs on **3× Red escalation** if an API key is set.

---

## CLI Reference

```
cvagent [--config PATH] [--interaction auto|ask] [--version] <command>
# `cv_agent` works the same way for every command.

Commands:
  run               Start closed-loop training
  validate          Dataset validation only
  resume            Resume from a prior experiment directory
  list-checkpoints  List Top-N, manual, and resumable checkpoints
```

### Global flags

| Flag | Description |
|------|-------------|
| `-c, --config PATH` | Config file (default: `cv_agent.yaml`) |
| `--interaction auto\|ask` | Override interaction mode |
| `--version` | Print version |

### `cvagent run` / `cv_agent run`

| Option | Description |
|--------|-------------|
| `--data-yaml PATH` | Dataset YAML (COCO128 bootstrap if omitted) |
| `--model TEXT` | Model variant (`yolo26s`, `yolov8m`, …) |
| `--device TEXT` | CUDA devices: `auto`, `0`, `0,1,2,3`, `cpu` |
| `--max-rounds INT` | Override `max_rounds` |
| `--optimize-for TEXT` | Class name to prioritize in reward |
| `--start fresh\|resume\|from-checkpoint` | Startup mode (skips wizard when set) |
| `--run-dir PATH` | Experiment dir for `--start resume` |
| `--checkpoint-id TEXT` | ID from `list-checkpoints` for `--start from-checkpoint` |

### `cv_agent resume`

| Option | Description |
|--------|-------------|
| `-r, --run-dir PATH` | Path to prior `runs/exp_<timestamp>/` directory |

### `cv_agent list-checkpoints`

Lists Top-N, manual, and resumable experiment entries with IDs like `exp_<timestamp>:top:1`.

**Supported model variants:** `yolo26{n,s,m,l,x}`, `yolov8{n,s,m,l,x}`, `yolo11{n,s,m,l,x}`.

---

## Interaction Modes

### Ask mode (default)

| Capability | Behavior |
|------------|----------|
| Decision panel | Color, action, reason, proposed params, rollback hint |
| Choices | Apply proposal, add guidance, skip changes, reject (Red), quit |
| Guidance | Natural language (LLM if key set) or regex (`only lr`, `keep mosaic`, `不要改mosaic`) |
| Guidance Applied panel | Shows parser source, interpretation, and param diff; pauses for Enter |
| Config confirm | Second diff review before params are applied |
| Manual checkpoint | Optional named save at DECIDE prompt |

### Auto mode

| Capability | Behavior |
|------------|----------|
| Decisions | Auto-approved after countdown (`auto_prompt_seconds`, default 10s) |
| Countdown keys | `A` → switch to Ask for this round; `Q` → quit |
| Guidance | Not collected (no blocking prompts) |
| Data errors | Writes supplement scripts and exits |

Switch **Ask ↔ Auto per round** at the DECIDE checkpoint; choice persists in `session_state.json`.

---

## Decision System

Score is compared to the **historical best** each round (`delta_percent` and optional `delta_abs`).

Green / Yellow / Red is a deterministic numeric classification from the rule controller, not an LLM judgment. The LLM can influence later proposals through bounded strategy patches and Ask-mode guidance, but it does not overwrite the measured round score used for rollback, checkpointing, or the decision timeline.

| State | Condition (vs. best) | Typical action |
|-------|----------------------|----------------|
| **Green (hard)** | Δ% ≥ `green_threshold_pct` or Δ_abs ≥ `green_threshold_abs` | Commit checkpoint; Optuna Bayesian proposal |
| **Green (marginal)** | `0 < Δ% < green_threshold` when `accept_marginal_improvement: true` | Commit checkpoint; keep params unless `marginal_green_use_optuna: true` |
| **Yellow** | Between `soft_red_threshold_pct` and green thresholds | Mild regularize/LR if overfit/underfit; else Optuna escape |
| **Red (soft)** | `soft_red_threshold_pct` ≥ Δ% > `red_threshold_pct` | Mild recovery **without** rollback |
| **Red (hard)** | Δ% ≤ `red_threshold_pct` or Δ_abs ≤ `red_threshold_abs` | Rollback + rule recovery (overfit / underfit / general) |

The **first round** establishes the run-local baseline and is auto-accepted without an interactive review. Official or published model metrics are useful external references, but they are not used as historical best because this loop needs apples-to-apples scores from the same dataset, epochs, device, and evaluation path.

**Diagnostic routing:** overfitting or underfitting in Yellow / soft-Red triggers targeted mild adjustments instead of blind random-walk escape.

### Red escalation (3× consecutive Reds)

When `red_escalation_count` is reached:

1. Force rollback to best checkpoint
2. LLM or heuristic confusion-matrix analysis
3. Apply suggested `box` / `cls` / `dfl` loss weights once (clamped 0.1–20)
4. Write `data_gap_report.md` / `.json`
5. Enter **Data Supplement Mode**

`yellow_resets_red_count: true` resets the Red streak after Yellow.

---

## Hyperparameter Optimization

| Round color | Strategy |
|-------------|----------|
| Green (hard) | Optuna TPE via `study.ask()` / `study.tell()` |
| Green (marginal) | Keep current params by default (`marginal_green_use_optuna: false`) |
| Yellow (escape) | `yellow_strategy`: `random_walk` (default), `simulated_annealing`, `bayesian` |
| Yellow / soft-Red (diagnostic) | Rule-based mild param adjust |
| Red | Rule recovery; pending Optuna trials abandoned |

**Key behaviors:**

- Per-run `optuna_study.db` under the experiment directory
- `n_trials` caps `ask()` calls; counter syncs with DB on resume
- `pruner: none` recommended — one `tell()` per round, so Median/Hyperband pruners have no effect
- Mismatched executed params mark the trial `FAIL`
- Search space fully configurable in YAML

Legacy `search_strategy: random_walk|simulated_annealing` maps to Yellow escape when non-Bayesian.

---

## AI Strategy Planner

The LLM strategy planner does not emit exact training hyperparameters. It emits bounded strategy patches: search-space narrowing, frozen fields, objective weights, and phase selection. Optuna still proposes precise numeric values inside those validated bounds, keeping numeric optimization auditable while using the LLM for diagnosis and strategy selection.

Planner objective weights are logged as strategy context for Optuna and memory, but the decision score remains the raw training reward. This keeps rollback comparisons stable even when the strategy planner changes its weighting emphasis.

Strategy runs persist `strategy_memory.json` for useful and avoid patterns across rounds, `strategy_log.json` for planner decisions, and the current `active_strategy_patch` in `session_state.json` so resumed experiments continue with the same constraints.

---

## Checkpoints & Resume

### Experiment layout

```
runs/exp_<timestamp>/
├── weights/
│   ├── best.pt
│   └── last.pt
├── best_snapshots/          # immutable per-round copies for rollback
├── checkpoints/
│   ├── top/                 # score-ranked Top-N
│   └── manual/              # user-named saves
├── optuna_study.db
├── session_state.json       # resume: round, params, red_streak, optuna_trial_count
├── metrics.json
├── decision_log.json
├── results.csv
└── cv_agent.log
```

### Three ways to continue training

| Mode | Command | What it does |
|------|---------|--------------|
| **Fresh** | `cv_agent run` | New `exp_*` from Ultralytics pretrained weights (`model_variant.pt`) |
| **Resume** | `cv_agent resume --run-dir runs/exp_<timestamp>` | Same directory; restores full session state |
| **Fork** | `cv_agent run --start from-checkpoint --checkpoint-id <ID>` | New `exp_*` fine-tuning from Top-N or manual save |

```bash
cv_agent list-checkpoints
cv_agent run --start from-checkpoint --checkpoint-id exp_<timestamp>:top:1
```

Training stops when `round_num >= max_rounds`. Raise `max_rounds` or resume with updated config for more rounds.

---

## Model Weights & Export

| Artifact | Location |
|----------|----------|
| Best weights | `runs/exp_<timestamp>/weights/best.pt` |
| Last weights | `runs/exp_<timestamp>/weights/last.pt` |
| Top-N library | `runs/exp_<timestamp>/checkpoints/top/` |

There is **no** built-in `cv_agent export` command. To export ONNX / TensorRT / etc., use Ultralytics directly:

```python
from ultralytics import YOLO
YOLO("runs/exp_<timestamp>/weights/best.pt").export(format="onnx")
```

**From scratch training** (no pretrained weights) is not supported — fresh runs always load official `{model_variant}.pt` COCO pretrained weights. To start from custom weights, save them as a checkpoint and use `--start from-checkpoint`.

---

## Dataset Validation & Supplement

`DatasetValidator` checks at startup:

- `min_images`, `min_ann_per_class`, `min_pixel_area`
- Image ↔ label pairing
- Optional `validate_brightness`, `validate_angles`

| Severity | Behavior |
|----------|----------|
| **error** | Data Supplement Mode |
| **warning** | Log and continue |

**Data Supplement Mode** (validation errors or 3× Red) generates scripts under `supplement_scripts/`. Run them manually, fix data, then re-run or retry validation.

---

## Single-Class Optimization

```bash
cv_agent run --data-yaml dataset.yaml --optimize-for person
```

```
reward = 0.3 × global_mAP50 + 1.7 × target_class_mAP50   # when --optimize-for set
reward = global_mAP50                                       # otherwise
```

Evaluation prints a **compact per-class summary** (count, avg, min/max, worst/best classes with names) instead of listing every class.

---

## Configuration Reference

Settings load from **`cv_agent.yaml`** (tracked). If present, **`cv_agent.local.yaml`** (git-ignored) deep-merges on top — include only keys you want to override (typically `llm.api_key`). CLI flags override YAML. No local file is required.

### Training loop

| Key | Default | Description |
|-----|---------|-------------|
| `model_variant` | `yolo26s` | Ultralytics model slug |
| `epochs_per_round` | `50` | Epochs per closed-loop round |
| `max_rounds` | `6` | Total rounds before stop |
| `device` | `auto` | `auto` (all visible GPUs), `0`, `0,1,2,3`, or `cpu` |
| `workers` | `8` (Linux) | DataLoader workers (`0` on Windows if unset) |
| `model_verbose` | `false` | Set `true` only when you need full Ultralytics model/training detail output |
| `interaction_mode` | `ask` | `ask` or `auto` |
| `auto_prompt_seconds` | `10` | Auto mode countdown before approving a round |
| `optimize_for_class` | `null` | Class name for weighted reward |
| `output_root` | `runs` | Parent directory for experiments |
| `experiment_name` | `cv_agent` | MLflow experiment name |
| `mlflow_uri` | `http://localhost:5000` | MLflow tracking URI |

### `data`

| Key | Default | Description |
|-----|---------|-------------|
| `data_yaml` | `coco.yaml` | Dataset spec path (Ultralytics registry; auto-download) |
| `min_images` | `50` | Minimum images per split |
| `min_ann_per_class` | `1` | Minimum annotations per class |
| `min_pixel_area` | `64` | Minimum object area (pixels) |
| `validate_brightness` | `true` | Brightness diversity heuristic |
| `validate_angles` | `true` | Angle diversity heuristic |

### `initial_hyperparams`

Starting YOLO training args for round 1 (and fallback). Includes `lr0`, `lrf`, `batch`, `momentum`, `weight_decay`, warmup, loss weights (`box`, `cls`, `dfl`), and augmentation (`mosaic`, `mixup`, `hsv_*`, `degrees`, …). See `cv_agent.yaml` for the full list.

### `decision`

| Key | Default | Description |
|-----|---------|-------------|
| `green_threshold_pct` | `3.0` | Hard Green if Δ% ≥ this |
| `green_threshold_abs` | `null` | Hard Green if absolute Δ score ≥ this (optional) |
| `red_threshold_pct` | `-5.0` | Hard Red if Δ% ≤ this |
| `red_threshold_abs` | `null` | Hard Red if absolute Δ score ≤ this (optional) |
| `soft_red_threshold_pct` | `-3.0` | Soft Red lower bound |
| `accept_marginal_improvement` | `true` | Treat small positive Δ% as marginal Green |
| `marginal_green_use_optuna` | `false` | Run Optuna on marginal Green rounds |
| `red_escalation_count` | `3` | Consecutive Reds before data-gap escalation |
| `yellow_resets_red_count` | `true` | Yellow round resets Red streak |

### `optuna`

| Key | Default | Description |
|-----|---------|-------------|
| `n_trials` | `50` | Max `ask()` calls per session |
| `search_strategy` | `bayesian` | Legacy; non-Bayesian values map to Yellow escape |
| `yellow_strategy` | `random_walk` | Yellow escape: `random_walk`, `simulated_annealing`, `bayesian` |
| `n_startup_trials` | `10` | TPE random startup trials |
| `pruner` | `none` | `none` recommended; `median` / `hyperband` ineffective here |
| `random_walk_min_step_scale` | `0.02` | Floor for Yellow random-walk step size |
| `search_space.*` | see yaml | Per-param ranges; `batch` is a categorical list |

### `strategy`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Run the AI strategy planner before Optuna proposals |
| `planner_cadence` | `1` | Plan every N rounds |
| `min_confidence` | `0.35` | Ignore planner patches below this confidence |
| `memory_enabled` | `true` | Persist strategy memory between rounds and resumes |
| `max_memory_items` | `50` | Maximum remembered effective / avoid patterns |
| `objective_weights.*` | see yaml | Reward weights for mAP, recall, precision, overfit, and cost |

### `checkpoints`

| Key | Default | Description |
|-----|---------|-------------|
| `top_n` | `5` | Leaderboard size |
| `auto_save_top` | `true` | Auto-record Top-N after each round |
| `manual_save_dir` | `manual` | Subdir for named manual saves |

### `llm`

| Key | Default | Description |
|-----|---------|-------------|
| `api_base` | DeepSeek URL | OpenAI-compatible API base |
| `api_key` | `""` | Leave empty; use env var or `cv_agent.local.yaml` |
| `model` | `deepseek-v4-flash` | Chat model name |
| `max_tokens` | `4096` | Max response tokens |
| `temperature` | `0.3` | Sampling temperature |
| `max_calls_per_session` | `20` | LLM call budget per run |
| `guidance_enabled` | `true` | Parse Ask-mode feedback via LLM |
| `guidance_fallback_regex` | `true` | Fall back to regex if LLM fails |

### Minimal config example

```yaml
model_variant: yolo26s
epochs_per_round: 50
max_rounds: 10
interaction_mode: auto

data:
  data_yaml: dataset.yaml
  min_images: 100
  min_ann_per_class: 10

decision:
  accept_marginal_improvement: true
  marginal_green_use_optuna: false

optuna:
  n_trials: 50
  pruner: none
  yellow_strategy: random_walk

llm:
  guidance_enabled: true
```

> Ultralytics uses `optimizer=auto` so `lr0` scales appropriately for the chosen optimizer (e.g. AdamW vs SGD).

---

## LLM Integration

Default backend: **DeepSeek** (OpenAI-compatible). Change `api_base` and `model` for other providers.

| Trigger | What the LLM does |
|---------|-------------------|
| **Ask-mode guidance** | Parses user feedback into frozen fields, multipliers, and set-values; shows diff panel |
| **Red×3 escalation** | Confusion-matrix analysis, data-gap report, loss-weight suggestions |

Normal rounds use the rule controller + Optuna. Without an API key, guidance falls back to regex rules; Red×3 uses heuristic analysis.

---

## MLflow Tracking

Point `mlflow_uri` at a running server, or let `cv_agent` fall back to `./mlruns` if unreachable.

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

`mlruns/` is git-ignored.

---

## Project Layout

```
cv_agent/
├── cv_agent.yaml              # default config (tracked)
├── cv_agent.local.yaml.example
├── coco128.yaml               # demo dataset spec
├── dataset.yaml.example       # custom dataset template (container paths)
├── Dockerfile                 # GPU image for server deployment
├── pyproject.toml
├── src/cv_agent/
│   ├── cli/                   # Click CLI
│   ├── core/                  # engine, config, state machine
│   ├── data/                  # validation, supplement, bootstrap
│   ├── decision/              # three-state, Optuna, LLM, guidance
│   ├── interaction/           # ask/auto handlers
│   ├── tracking/              # checkpoints, MLflow, run dirs
│   ├── trainer/               # YOLO trainer, evaluator
│   └── ui/                    # Rich console, live panel
└── tests/
```

---

## Development

```bash
pytest tests/ --basetemp=.tmp_pytest -p no:cacheprovider
ruff check src tests
```

Enable the pre-commit secret scanner:

```bash
git config core.hooksPath .githooks
```

---

## Security

**Never commit real API keys.**

| Priority | Method |
|----------|--------|
| 1 | `export CV_AGENT_LLM_KEY="sk-..."` or `export DEEPSEEK_API_KEY="sk-..."` |
| 2 | `cv_agent.local.yaml` (git-ignored) |
| 3 | Empty `api_key` → no LLM calls |

The `.githooks/pre-commit` hook blocks commits that match common secret patterns.

---

## License

MIT — see `pyproject.toml`.
