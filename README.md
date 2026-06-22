# cv_agent — Automated Closed-Loop YOLO Training CLI

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-YOLOv8%2F11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20opt-green.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-blue.svg)](https://mlflow.org/)

**cv_agent** is an automated closed-loop CLI training system based on Ultralytics YOLO that combines **Optuna Bayesian hyperparameter optimization** with **LLM-based strategic reasoning** to continuously improve object detection models without human supervision.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    cv_agent run --data-yaml dataset.yaml     │
└──────────────────────────┬───────────────────────────────────┘
                           │
     ┌─────────────────────▼──────────────────────┐
     │          TrainingEngine (main loop)         │
     │  INIT → VALIDATE → TRAIN → EVAL → DECIDE   │
     └──┬────────┬──────────┬──────────┬──────────┘
        │        │          │          │
   ┌────▼──┐ ┌──▼────┐ ┌───▼───┐ ┌───▼────────────┐
   │Data   │ │YOLO   │ │Optuna │ │LLM Advisor     │
   │Valid. │ │Trainer│ │Hparam │ │(DeepSeek API +  │
   │+Suppl.│ │+Eval  │ │Search │ │ heuristic fallb)│
   └───────┘ └───────┘ └───────┘ └────────────────┘
                           │
              ┌────────────▼────────────┐
              │  MLflow + runs/exp_ts/  │
              │  Tracking & Snapshots   │
              └─────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Activate the conda environment with ultralytics installed
conda activate yolo

# Install cv_agent
cd cv_agent
pip install -e .
```

### Basic Usage

```bash
# Create a config file (or use defaults)
cv_agent --help

# Run full closed-loop training on your dataset
cv_agent run --data-yaml path/to/dataset.yaml --optimize-for vehicle

# Dry-run: validate dataset only
cv_agent validate --data-yaml path/to/dataset.yaml

# Run in fully autonomous mode (no prompts)
cv_agent run --data-yaml dataset.yaml --interaction auto --max-rounds 5

# Resume a prior experiment
cv_agent resume --run-dir runs/exp_20260122_143052
```

## Run Modes

### Auto Mode (`--interaction auto`)
- Executes all parameter mutations, training, and rollbacks automatically
- Prints color-coded decision logs (Green/Yellow/Red) via Rich
- Writes `decision_log.json` to the run directory
- Never blocks — ideal for unattended training

### Ask-before-edit Mode (`--interaction ask`, default)
- Before any config modification, renders a **change diff** via Rich Panel
- Blocks and waits for **Y/n** confirmation via questionary
- Accepts **natural language explanations** (e.g., "Don't change Mosaic, just adjust LR")
- NL feedback is incorporated into the next round's LLM context

## Three-State Decision System

After each training round, the system classifies the result into one of three states:

| State | Threshold | Action |
|-------|-----------|--------|
| 🟢 **Green** | ≥ +3% relative | Commit checkpoint, Optuna Bayesian proposal |
| 🟡 **Yellow** | ±3% oscillation | Random walk or simulated annealing to escape local optimum |
| 🔴 **Red** | ≤ -5% relative | Diagnose overfit/underfit, adjust params or rollback |

**Red×3 escalation**: If 3 consecutive Red states occur, the system:
1. Force-rolls back to the historical best checkpoint
2. Calls the LLM to analyze the confusion matrix on the validation set
3. Generates a **Data Gap Report** (Markdown + JSON) identifying which classes need more data
4. Transitions to **Data Supplement Mode**

## Data Validation & Supplement

At startup, the system validates:
- Image count ≥ minimum threshold
- Per-class annotation count
- Every image has a label file (and vice versa)
- Object size distribution
- Brightness diversity

If validation fails → **Data Supplement Mode**:
- Diagnoses data distribution defects
- Generates executable download scripts:
  - `roboflow_download.py`
  - `openimages_download.sh`
  - `huggingface_download.py`
  - `annotation_tools.md`
- In ask mode: blocks until user imports data
- In auto mode: prints scripts and continues

## Single-Class Optimization (`--optimize-for`)

```bash
cv_agent run --optimize-for vehicle
```

The reward function prioritizes the target class's `mAP@0.5` with 3× weight:
```
reward = 0.3 × global_mAP50 + 1.7 × target_class_mAP50
```

## Configuration

Create a `cv_agent.yaml` file:

```yaml
model_variant: yolov8n
epochs_per_round: 100
max_rounds: 10
interaction_mode: ask  # auto | ask
optimize_for_class: null  # or "vehicle"

data:
  data_yaml: dataset.yaml
  min_images: 100
  min_ann_per_class: 50
  min_pixel_area: 64
  validate_brightness: true
  validate_angles: true

initial_hyperparams:
  lr0: 0.01
  batch: 16
  mosaic: 1.0
  mixup: 0.0

optuna:
  n_trials: 50
  search_strategy: bayesian  # bayesian | random_walk | simulated_annealing
  n_startup_trials: 10
  pruner: median
  search_space:
    lr0: [0.001, 0.1]
    batch: [4, 8, 16, 32]
    mosaic: [0.0, 1.0]
    mixup: [0.0, 0.5]

llm:
  api_base: https://api.deepseek.com/v1
  api_key: ""  # Reads from CV_AGENT_LLM_KEY or DEEPSEEK_API_KEY env vars
  model: deepseek-chat
  max_tokens: 4096

mlflow_uri: http://localhost:5000
experiment_name: cv_agent
output_root: runs
```

## Experiment Directory Structure

```
runs/
  exp_20260122_143052/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── args.yaml
    ├── results.csv
    ├── metrics.json
    ├── decision_log.json
    ├── data_gap_report.md
    ├── data_gap_report.json
    ├── supplement_scripts/
    │   ├── roboflow_download.py
    │   ├── openimages_download.sh
    │   ├── huggingface_download.py
    │   └── annotation_tools.md
    └── cv_agent.log
```

## LLM Backend

Default: **DeepSeek API** (`api.deepseek.com/v1`). Also works with any OpenAI-compatible API.

Set your API key via environment variable:
```bash
export CV_AGENT_LLM_KEY="sk-your-key-here"
# or
export DEEPSEEK_API_KEY="sk-your-key-here"
```

**Heuristic fallback**: If no API key is configured or the API is unreachable, cv_agent falls back to rule-based statistical analysis of confusion matrices — no LLM calls needed.

## Requirements

- Python 3.10+
- Conda environment with Ultralytics YOLO, PyTorch, CUDA
- MLflow server (optional — metrics are also saved locally)
- LLM API key (optional — heuristic fallback available)