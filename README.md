# cv_agent — Automated Closed-Loop YOLO Training CLI

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-YOLOv8%2F11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20opt-green.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-blue.svg)](https://mlflow.org/)

**cv_agent** is an automated closed-loop CLI training system built on Ultralytics
YOLO. It combines **Optuna Bayesian hyperparameter optimization** with
**LLM-based strategic reasoning** to continuously improve an object-detection
model round after round, with a three-state (Green / Yellow / Red) decision
engine driving checkpoint commit, local-optimum escape, and rollback.

It runs two ways: fully autonomous (`auto`) or human-in-the-loop (`ask`,
with change diffs and Y/n gates).

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Run Modes](#run-modes)
- [Three-State Decision System](#three-state-decision-system)
- [Data Validation & Supplement](#data-validation--supplement)
- [Single-Class Optimization](#single-class-optimization---optimize-for)
- [Configuration](#configuration)
- [Secrets & API Keys](#secrets--api-keys)
- [Experiment Directory](#experiment-directory)
- [LLM Backend](#llm-backend)
- [MLflow Tracking](#mlflow-tracking)
- [Requirements](#requirements)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              cv_agent run --data-yaml dataset.yaml           │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │         TrainingEngine (main loop)   │
        │  INIT → VALIDATE → TRAIN → EVAL →   │
        │                DECIDE → (next round) │
        └──┬────────┬──────────┬──────────┬───┘
           │        │          │          │
      ┌────▼──┐ ┌──▼─────┐ ┌──▼──────┐ ┌─▼──────────────┐
      │ Data  │ │ YOLO   │ │ Optuna  │ │ LLM Advisor    │
      │Valid. │ │Trainer │ │ Hparam  │ │ (DeepSeek API + │
      │+Suppl.│ │ + Eval │ │ Search  │ │ heuristic fallb)│
      └───────┘ └────────┘ └─────────┘ └────────────────┘
                           │
              ┌────────────▼────────────┐
              │ MLflow (local fallback)  │
              │  + runs/exp_<ts>/        │
              │  snapshots & logs        │
              └──────────────────────────┘
```

Each round: validate (once) → train N epochs → evaluate → decide
(Green/Yellow/Red) → mutate hyperparameters → repeat until `max_rounds`.

## Quick Start

### 1. Prerequisites

```bash
# Activate the conda env that has ultralytics + torch + CUDA
conda activate yolo

# Install cv_agent (editable, into the yolo env)
cd cv_agent
pip install -e .
```

The CLI verifies at startup that `ultralytics` and `torch` are importable and
exits with a clear message if the wrong environment is active.

### 2. First run (COCO128 demo)

The repo ships a `coco128.yaml` and the default config points at it, so you
can verify the full pipeline in ~1 minute:

```bash
cv_agent run
```

This trains `yolov8n` on COCO128 for a few epochs and runs the closed loop.
**Be patient through startup** — model loading, dataloader scanning, and the
AMP check all happen before the first epoch prints; GPU idles during this
phase, which is normal. Once you see `Starting training for N epochs...`,
training is underway.

### 3. Real usage

```bash
# Full closed-loop training on your dataset, prioritizing one class
cv_agent run --data-yaml path/to/dataset.yaml --optimize-for vehicle

# Fully autonomous (no prompts)
cv_agent run --data-yaml dataset.yaml --interaction auto --max-rounds 10

# Dry-run: validate the dataset only, no training
cv_agent validate --data-yaml path/to/dataset.yaml

# Resume a prior experiment
cv_agent resume --run-dir runs/exp_20260122_143052
```

### CLI reference

```
cv_agent [--config PATH] [--interaction auto|ask] [--version] <command>

Commands:
  run        Start automated closed-loop training
  validate   Run dataset validation only (dry run)
  resume     Resume from a prior experiment directory

run options:
  --optimize-for TEXT    Class name to prioritize (e.g. "vehicle")
  --max-rounds INT       Override max training rounds
  --data-yaml PATH       Override dataset YAML path
  --model TEXT           Override model variant (yolov8n/s/m/l/x)
```

## Run Modes

### Auto Mode (`--interaction auto`)
- Executes all parameter mutations, training, and rollbacks automatically
- Prints color-coded decision logs (🟢/🟡/🔴) via Rich
- Never blocks — ideal for unattended / overnight training
- On unrecoverable dataset validation errors it aborts cleanly (an unattended
  run cannot fix missing data by re-validating in a loop)

### Ask-before-edit Mode (`--interaction ask`, default)
- Before any hyperparameter change, renders a **change diff** via a Rich panel
- Blocks for **Y/n** confirmation (questionary)
- Accepts **natural-language feedback** (e.g. "Don't change Mosaic, just LR")
- NL feedback is folded into the next round's LLM context
- On dataset validation failure, prompts whether to retry after importing data

## Three-State Decision System

After each round, the result is classified relative to the historical best:

| State | Condition | Action |
|-------|-----------|--------|
| 🟢 **Green** | clear improvement | Commit checkpoint, Optuna Bayesian proposal for next round |
| 🟡 **Yellow** | within oscillation band | Random walk or simulated annealing to escape the local optimum |
| 🔴 **Red** | clear degradation | Diagnose overfit/underfit, adjust params or rollback to best |

The first round is always accepted as the Green baseline.

**Red × 3 escalation**: after 3 consecutive Red states the system
1. Force-rolls back to the historical best checkpoint
2. Calls the LLM (or heuristic fallback) to analyze the validation confusion matrix
3. Generates a **Data Gap Report** (`data_gap_report.md` + `.json`)
4. Enters **Data Supplement Mode**

## Data Validation & Supplement

At startup the dataset is validated for:
- Image count ≥ `min_images` per split
- Per-class annotation count ≥ `min_ann_per_class`
- Every image has a label file (and vice-versa)
- Object size distribution (pixel area)
- Optional brightness diversity

Paths follow Ultralytics conventions: an absolute `path:` is used as-is; a
relative `path:` resolves against the YAML directory, then Ultralytics'
`datasets_dir` (where auto-downloaded datasets like COCO128 land).

**Data Supplement Mode** (triggered on validation errors):
- Diagnoses which classes / splits are deficient
- Generates executable download scripts under `supplement_scripts/`:
  `roboflow_download.py`, `openimages_download.sh`,
  `huggingface_download.py`, `annotation_tools.md`
- **ask mode**: prompts whether to retry validation after you import data
- **auto mode**: writes the scripts and aborts the session (an unattended run
  cannot self-heal missing data) — run the scripts, add the data, re-run

## Single-Class Optimization (`--optimize-for`)

```bash
cv_agent run --optimize-for vehicle
```

The reward function weights the target class's `mAP@0.5` 3×:

```
reward = 0.3 × global_mAP50 + 1.7 × target_class_mAP50
```

Without `--optimize-for`, `reward = global_mAP50`.

## Configuration

Settings live in `cv_agent.yaml` (tracked template). Any field can be
overridden by CLI flags. Key fields:

```yaml
model_variant: yolov8n
epochs_per_round: 100       # epochs per round
max_rounds: 10              # total closed-loop rounds
interaction_mode: ask       # auto | ask
optimize_for_class: null    # or e.g. "vehicle"

data:
  data_yaml: coco128.yaml
  min_images: 100
  min_ann_per_class: 50
  min_pixel_area: 64
  validate_brightness: true

initial_hyperparams:
  lr0: 0.01
  batch: 16
  mosaic: 1.0
  mixup: 0.0
  # ... full augmentation set: hsv_*, degrees, scale, shear, flipud, fliplr, ...

optuna:
  n_trials: 50
  search_strategy: bayesian   # bayesian | random_walk | simulated_annealing
  n_startup_trials: 10
  pruner: median              # median | hyperband | none
  search_space:
    lr0: [0.001, 0.1]
    batch: [4, 8, 16, 32]
    mosaic: [0.0, 1.0]
    mixup: [0.0, 0.5]
    # ... tune or widen any range without code changes

llm:
  api_base: https://api.deepseek.com
  api_key: ""                 # leave empty; use env var or local file (see Secrets)
  model: deepseek-v4-flash
  max_tokens: 4096
  temperature: 0.3
  max_calls_per_session: 20

mlflow_uri: http://localhost:5000   # falls back to local file store if unreachable
experiment_name: cv_agent
output_root: runs
```

> The optimizer is left on Ultralytics' `auto` so `lr0` is scaled appropriately
> for the chosen optimizer (e.g. AdamW wants ~1e-3, not a SGD-style 0.01).

## Secrets & API Keys

**Never commit a real API key.** Three safe ways to supply the LLM key, in
priority order:

1. **Environment variable** (recommended):
   ```bash
   export CV_AGENT_LLM_KEY="sk-..."     # or DEEPSEEK_API_KEY
   ```

2. **Local override file** `cv_agent.local.yaml` (git-ignored, deep-merged on
   top of `cv_agent.yaml`):
   ```yaml
   llm:
     api_key: "sk-your-real-key"
   ```

3. Leave `api_key` empty in `cv_agent.yaml` → heuristic fallback (no LLM).

A **pre-commit hook** (`.githooks/pre-commit`) scans staged content for common
secret patterns (`sk-...`, GitHub tokens, private keys, …) and aborts the
commit on a match. Activate it once:
```bash
git config core.hooksPath .githooks
```

## Experiment Directory

Each session creates `runs/exp_<timestamp>/`:

```
runs/exp_20260122_143052/
├── weights/
│   ├── best.pt
│   └── last.pt
├── args.yaml                 # Ultralytics training args
├── results.csv               # per-epoch metrics (Ultralytics)
├── metrics.json              # cv_agent round metrics
├── decision_log.json         # per-round Green/Yellow/Red decisions
├── data_gap_report.md        # on Red×3 escalation
├── data_gap_report.json
├── supplement_scripts/       # on validation failure
│   ├── roboflow_download.py
│   ├── openimages_download.sh
│   ├── huggingface_download.py
│   └── annotation_tools.md
└── cv_agent.log
```

## LLM Backend

Default: **DeepSeek API** (`api.deepseek.com`), OpenAI-compatible. Works with
any OpenAI-compatible endpoint — change `api_base` / `model` in the config.

The LLM is only used for two structured tasks: confusion-matrix analysis and
data-gap-report generation (both after Red×3 escalation). It is **not** on the
hot path of normal rounds.

**Heuristic fallback**: with no API key, or if the API is unreachable / the
call limit is hit, cv_agent falls back to rule-based statistical analysis
(recall thresholds, off-diagonal confusion rates, average-recall-driven loss
weights). The closed loop keeps running with zero LLM calls.

## MLflow Tracking

If `mlflow_uri` points at a running MLflow server, metrics/params/artifacts
are logged there. **If the server is unreachable, cv_agent transparently
falls back to a local file store (`./mlruns`)** so Ultralytics' built-in
MLflow callback never stalls `model.train()` on a dead network endpoint.
Either way you can browse results with:
```bash
mlflow ui        # → http://localhost:5000
```

(`mlruns/` is git-ignored.)

## Requirements

- Python 3.10+
- A conda environment with Ultralytics YOLO, PyTorch, CUDA (the `yolo` env)
- MLflow server — **optional** (local file-store fallback works out of the box)
- LLM API key — **optional** (heuristic fallback works out of the box)
