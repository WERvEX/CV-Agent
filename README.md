# cv_agent

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20search-green.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](pyproject.toml)

**Automated closed-loop YOLO object-detection training.**

`cv_agent` trains Ultralytics YOLO models in repeated rounds: validate data, train, evaluate, classify the outcome (Green / Yellow / Red), propose the next hyperparameters, and repeat. [Optuna](https://optuna.org/) drives hyperparameter search; a rule-based controller handles checkpoint commits, rollbacks, and local-optimum escape. An optional LLM assists only when training hits repeated failures and data gaps are suspected.

Run fully unattended (`auto`) or stay in the loop (`ask`) with diffs, confirmations, and natural-language guidance.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Interaction Modes](#interaction-modes)
- [Decision System](#decision-system)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Checkpoints & Resume](#checkpoints--resume)
- [Dataset Validation & Supplement](#dataset-validation--supplement)
- [Single-Class Optimization](#single-class-optimization)
- [Configuration](#configuration)
- [LLM Integration](#llm-integration)
- [MLflow Tracking](#mlflow-tracking)
- [Project Layout](#project-layout)
- [Development](#development)
- [Security](#security)
- [License](#license)

---

## Features

- **Closed-loop training** — multi-round train → evaluate → decide without manual scripting
- **Three-state decisions** — Green (commit), Yellow (escape local optimum), Red (rollback / recovery)
- **Optuna integration** — Bayesian search on Green; random walk / simulated annealing / Bayesian on Yellow
- **Per-class metrics** — validation enrichment with per-class mAP for targeted optimization
- **Checkpoint management** — Top-N leaderboard, per-round snapshots, manual named saves
- **Resume & fork** — continue an experiment or start a new run from a saved checkpoint
- **Dataset validation** — image/label checks, class counts, object size, optional quality heuristics
- **Data supplement mode** — generates download/annotation helper scripts when data is insufficient
- **Ask or Auto** — human-in-the-loop with param diffs and guidance parsing, or hands-off runs
- **MLflow logging** — remote server or transparent local fallback
- **Optional LLM** — data-gap analysis after 3 consecutive Reds (heuristic fallback if no API key)

---

## How It Works

Each training session runs a state machine until `max_rounds` is reached or the user quits:

```
INIT → VALIDATE_DATA → TRAIN → EVALUATE → DECIDE ─┐
         │                    ↑                    │
         └→ DATA_SUPPLEMENT ──┘                    │
                              └────────────────────┘
                                        ↓
                                      DONE
```

| Component | Responsibility |
|-----------|----------------|
| `TrainingEngine` | Main loop, round lifecycle, artifact I/O |
| `DatasetValidator` | Startup data quality checks |
| `YoloTrainer` + `Evaluator` | Ultralytics training and reward scoring |
| `ThreeStateDecisionEngine` | Green / Yellow / Red classification and recovery actions |
| `OptunaOptimizer` | Next-round hyperparameter proposals |
| `LLMAdvisor` | Confusion-matrix and data-gap analysis on **3× Red escalation only** |
| `CheckpointManager` | Top-N saves, manual saves, resume metadata |
| `MLflowManager` | Experiment tracking |

**Normal rounds do not call the LLM.** Hyperparameter changes come from the rule controller + Optuna. User feedback in Ask mode is parsed by rule-based `guidance` (e.g. “only lr”, “don't change mosaic”), not by the LLM.

---

## Requirements

- **Python** 3.10+
- **PyTorch** + **Ultralytics YOLO** (GPU recommended)
- **MLflow server** — optional (local `./mlruns` fallback works out of the box)
- **LLM API key** — optional (heuristic fallback works out of the box)

The CLI checks at startup that `torch` and `ultralytics` are importable. Use a dedicated conda/venv with those packages installed (many teams use an env named `yolo`).

---

## Installation

```bash
git clone <your-repo-url>
cd cv_agent

# Activate your YOLO environment (example)
conda activate yolo

pip install -e .

# Optional: dev tools
pip install -e ".[dev]"
```

---

## Quick Start

### Smoke test (COCO128)

The repo ships `coco128.yaml` and defaults in `cv_agent.yaml` point at it. No dataset path is required for a first run:

```bash
cv_agent run
```

This downloads COCO128 if needed, trains `yolo26n` for a few epochs per round, and runs the full closed loop.

> **Note:** Startup can look idle while Ultralytics loads the model, scans the dataloader, and runs the AMP check. Once you see `Starting training for N epochs...`, training is underway.

### Train on your dataset

```bash
cv_agent run --data-yaml path/to/dataset.yaml --model yolo26s --max-rounds 10
```

### Validate only (no training)

```bash
cv_agent validate --data-yaml path/to/dataset.yaml
```

### Resume or fork

```bash
# Continue the same experiment directory
cv_agent resume --run-dir runs/exp_20260122_143052

# Or use the startup wizard / flags on `run`
cv_agent run --start resume --run-dir runs/exp_20260122_143052
cv_agent run --start from-checkpoint --checkpoint-id exp_20260122_143052:top:1

# List saved checkpoints
cv_agent list-checkpoints
```

On an interactive TTY, `cv_agent run` also prompts for:

1. **New from pretrained** — default; uses `model_variant` weights
2. **Resume experiment** — same `exp_*` directory; restores round, hyperparameters, Optuna study
3. **New from checkpoint** — new `exp_*` directory; fine-tunes from a Top-N or manual save

---

## CLI Reference

```
cv_agent [--config PATH] [--interaction auto|ask] [--version] <command>

Commands:
  run               Start closed-loop training
  validate          Dataset validation only
  resume            Resume from a prior experiment directory
  list-checkpoints  List Top-N, manual, and resumable checkpoints
```

### `cv_agent run` options

| Option | Description |
|--------|-------------|
| `--data-yaml PATH` | Dataset YAML (defaults to COCO128 bootstrap if omitted) |
| `--model TEXT` | Model variant (`yolo26n`, `yolov8n`, `yolo11s`, …) |
| `--max-rounds INT` | Override `max_rounds` from config |
| `--optimize-for TEXT` | Prioritize one class in the reward function |
| `--start fresh\|resume\|from-checkpoint` | Startup mode (non-interactive) |
| `--run-dir PATH` | Run directory for `--start resume` |
| `--checkpoint-id TEXT` | Checkpoint ID from `list-checkpoints` |

Global flags:

| Flag | Description |
|------|-------------|
| `-c, --config PATH` | Config file (default: `cv_agent.yaml`) |
| `--interaction auto\|ask` | Override interaction mode |

Supported model variants: `yolo26{n,s,m,l,x}`, `yolov8{n,s,m,l,x}`, `yolo11{n,s,m,l,x}`.

---

## Interaction Modes

### Ask mode (default)

- Shows a **decision panel** after each round: color, action, reason, proposed hyperparameters, rollback hint
- Blocks for explicit choices: apply proposal, reject, skip rollback, add guidance, quit
- **Guidance parsing** — common phrases constrain the next proposal:
  - `only lr` / `just lr` / `只改lr` → adjust learning rate only
  - `don't change mosaic` / `keep batch` / `不要改mosaic` → freeze fields
- Optional **manual checkpoint save** at the DECIDE prompt
- On dataset validation failure: choose supplement, retry validation, or abort

### Auto mode

- Auto-approves decisions after a configurable countdown (`auto_prompt_seconds`, default 10s)
- Press `A` during countdown to review the current round in Ask; `Q` to quit
- No blocking prompts — suitable for overnight runs
- On unrecoverable dataset errors: generates supplement scripts and **exits** (cannot self-heal missing data)

You can switch Ask ↔ Auto **per round** at the DECIDE checkpoint. The chosen mode persists for later rounds until changed again.

---

## Decision System

After each round, the reward score is compared to the historical best:

| State | Condition (vs. best) | Typical action |
|-------|----------------------|----------------|
| Green | improvement ≥ `green_threshold_pct` (default 3%) | Commit best checkpoint; Optuna Bayesian proposal for next round |
| Yellow | between green and red thresholds | Local-optimum escape (random walk / SA / Bayesian per config) |
| Red | drop ≤ `red_threshold_pct` (default −5%) | Diagnose overfit/underfit; rollback and/or aggressive param change |

The **first round** is always accepted as the Green baseline.

### Red escalation (3× consecutive Reds)

When `red_escalation_count` (default 3) is hit:

1. Force rollback to the historical best checkpoint
2. LLM (or heuristic fallback) analyzes the confusion matrix
3. Writes `data_gap_report.md` / `.json`
4. Enters **Data Supplement Mode**

`yellow_resets_red_count: true` (default) resets the Red streak after a Yellow round.

---

## Hyperparameter Optimization

| Round color | Optuna strategy |
|-------------|-----------------|
| Green | TPE Bayesian via `study.ask()` / `study.tell()` |
| Yellow | `yellow_strategy`: `random_walk` (default), `simulated_annealing`, or `bayesian` |
| Red | Controller-driven recovery; pending Optuna trials are abandoned |

Key behaviors:

- Each run uses its own `optuna_study.db` under the experiment directory
- `n_trials` caps how many `ask()` calls Optuna makes per session; after the budget, proposals keep current params
- Only scores from **actually executed** proposed params are reported to Optuna; mismatches mark the trial `FAIL`
- Search space is fully configurable in YAML (`optuna.search_space`)

Legacy `search_strategy: random_walk|simulated_annealing` still maps to the Yellow escape strategy when non-Bayesian.

---

## Checkpoints & Resume

Each experiment lives under `runs/exp_<timestamp>/`:

```
runs/exp_20260122_143052/
├── weights/
│   ├── best.pt                 # active weights (updated during training)
│   └── last.pt
├── best_snapshots/             # immutable per-round copies for rollback
│   └── round_001_best.pt
├── checkpoints/
│   ├── leaderboard.json        # Top-N ranking
│   ├── top/
│   │   └── rank01_score0.6579_round1.pt
│   └── manual/
│       └── my_save/
│           ├── weights.pt
│           └── manifest.json
├── optuna_study.db             # per-run Optuna study
├── session_state.json          # resume metadata
├── metrics.json
├── decision_log.json
├── args.yaml                   # Ultralytics training args
├── results.csv
├── data_gap_report.md          # after 3× Red escalation
├── supplement_scripts/         # on validation / data-gap failure
└── cv_agent.log
```

| Mechanism | Purpose |
|-----------|---------|
| `best_snapshots/` | Rollback history after Red decisions |
| `checkpoints/top/` | Score-ranked library for forking new experiments |
| `checkpoints/manual/` | User-named saves from Ask mode |
| `session_state.json` | Resume the same `exp_*` directory |

Configure in `cv_agent.yaml`:

```yaml
checkpoints:
  top_n: 5
  auto_save_top: true
  manual_save_dir: manual
```

Training stops when `round_num >= max_rounds`. It does **not** run indefinitely — raise `max_rounds` or resume with an updated config for more rounds.

---

## Dataset Validation & Supplement

At startup, `DatasetValidator` checks:

- Minimum images per split (`min_images`)
- Minimum annotations per class (`min_ann_per_class`)
- Image ↔ label pairing (missing labels, orphan labels)
- Object pixel-area distribution (`min_pixel_area`)
- Optional brightness and angle diversity

Dataset paths follow Ultralytics conventions: absolute `path:` is used as-is; relative `path:` resolves against the YAML directory, then Ultralytics' `datasets_dir`.

| Severity | Behavior |
|----------|----------|
| **error** | Enter Data Supplement Mode |
| **warning** | Log and continue training |

### Data Supplement Mode

Triggered by validation **errors** or **3× Red escalation**. Generates helper scripts under `supplement_scripts/`:

- `roboflow_download.py`
- `openimages_download.sh`
- `huggingface_download.py`
- `annotation_tools.md` (when annotations are missing)

| Mode | Behavior |
|------|----------|
| **ask** | Prompt to retry validation after you import/fix data |
| **auto** | Write scripts and end the session |

`cv_agent` does not auto-download data; run the scripts manually, fix your dataset, then re-run or retry validation.

---

## Single-Class Optimization

Prioritize one class in the reward function:

```bash
cv_agent run --data-yaml dataset.yaml --optimize-for vehicle
```

Reward formula when `--optimize-for` is set:

```
reward = 0.3 × global_mAP50 + 1.7 × target_class_mAP50
```

Without `--optimize-for`, `reward = global_mAP50`.

Per-class mAP keys appear in metrics after validation enrichment (`mAP50_class_<id>`).

---

## Configuration

Settings live in `cv_agent.yaml` (tracked template). Create `cv_agent.local.yaml` (git-ignored) for secrets and personal overrides — it is deep-merged on top. CLI flags override YAML fields.

```yaml
model_variant: yolo26n
epochs_per_round: 3          # raise for real training (e.g. 50–100)
max_rounds: 7
interaction_mode: ask        # auto | ask
auto_prompt_seconds: 10

optimize_for_class: null     # or e.g. "person"

data:
  data_yaml: coco128.yaml
  min_images: 50
  min_ann_per_class: 1
  min_pixel_area: 64
  validate_brightness: true
  validate_angles: true

initial_hyperparams:
  lr0: 0.01
  batch: 16
  mosaic: 1.0
  # ... full augmentation and loss weights

decision:
  green_threshold_pct: 3.0
  red_threshold_pct: -5.0
  red_escalation_count: 3
  yellow_resets_red_count: true

optuna:
  n_trials: 50
  yellow_strategy: random_walk   # random_walk | simulated_annealing | bayesian
  n_startup_trials: 10
  pruner: median                 # median | hyperband | none
  search_space:
    lr0: [0.001, 0.1]
    batch: [4, 8, 16, 32]
    mosaic: [0.0, 1.0]
    # ... widen or narrow any range without code changes

checkpoints:
  top_n: 5
  auto_save_top: true

llm:
  api_base: https://api.deepseek.com
  api_key: ""                  # use env var or cv_agent.local.yaml
  model: deepseek-v4-flash
  max_calls_per_session: 20

mlflow_uri: http://localhost:5000
experiment_name: cv_agent
output_root: runs
```

> Ultralytics optimizer stays on `auto` so `lr0` scales appropriately for the chosen optimizer (e.g. AdamW vs SGD).

---

## LLM Integration

Default backend: **DeepSeek** (OpenAI-compatible API). Any OpenAI-compatible endpoint works — change `api_base` and `model` in config.

The LLM is used **only** after 3 consecutive Red rounds:

1. Confusion-matrix analysis
2. Structured data-gap report generation

It is **not** on the hot path for normal Green/Yellow/Red hyperparameter decisions. LLM-suggested loss weights are reported but **not** automatically applied to YOLO training.

Without an API key, or when the call limit / network fails, `cv_agent` falls back to heuristic statistical analysis and the closed loop continues.

---

## MLflow Tracking

If `mlflow_uri` points to a running server, metrics, parameters, and artifacts are logged remotely.

If the server is unreachable, `cv_agent` transparently falls back to a local file store (`./mlruns`) so training never blocks on a dead endpoint.

```bash
mlflow ui    # browse at http://localhost:5000
```

`mlruns/` is git-ignored.

---

## Project Layout

```
cv_agent/
├── cv_agent.yaml           # default config (tracked)
├── coco128.yaml            # demo dataset spec
├── pyproject.toml
├── src/cv_agent/
│   ├── cli/                # Click CLI entry point
│   ├── core/               # engine, config, state machine
│   ├── data/               # validation, supplement, bootstrap
│   ├── decision/           # three-state, Optuna, LLM, guidance
│   ├── interaction/        # ask/auto handlers, mode control
│   ├── tracking/           # checkpoints, MLflow, run dirs
│   ├── trainer/            # YOLO trainer, evaluator
│   └── ui/                 # Rich console, live panel, prompts
└── tests/
```

---

## Development

```bash
# Run tests (Windows: use a dedicated basetemp)
pytest tests/ --basetemp=.tmp_pytest -p no:cacheprovider

# Lint
ruff check src tests
```

Activate the pre-commit secret scanner (recommended):

```bash
git config core.hooksPath .githooks
```

---

## Security

**Never commit real API keys.**

Provide the LLM key via (priority order):

1. Environment variable:
   ```bash
   export CV_AGENT_LLM_KEY="sk-..."
   # or
   export DEEPSEEK_API_KEY="sk-..."
   ```

2. Git-ignored local config `cv_agent.local.yaml`:
   ```yaml
   llm:
     api_key: "sk-your-real-key"
   ```

3. Leave `api_key` empty → heuristic-only mode (no LLM calls).

The `.githooks/pre-commit` hook scans staged files for common secret patterns (`sk-...`, tokens, private keys) and blocks the commit on a match.

---

## License

MIT — see `pyproject.toml`.
