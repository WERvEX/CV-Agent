"""Configuration models for cv_agent.

All configuration domains are defined as Pydantic BaseModels, providing:
- Automatic YAML/JSON deserialization
- Field validation at load time
- Environment variable fallback for secrets
- CLI override support via model_copy(update=...)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# YOLO model variants supported
# ---------------------------------------------------------------------------
YOLO_VARIANTS = {
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
}


# ---------------------------------------------------------------------------
# Hyperparameter models
# ---------------------------------------------------------------------------

class HyperParams(BaseModel):
    """YOLO training hyperparameters that Optuna can tune."""

    lr0: float = 0.01
    lrf: float = 0.01
    batch: int = 16
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5

    @field_validator("batch")
    @classmethod
    def batch_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch must be >= 1, got {v}")
        return v

    @field_validator("lr0", "lrf")
    @classmethod
    def lr_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"learning rate must be > 0, got {v}")
        return v


class OptunaSearchSpace(BaseModel):
    """Configurable search ranges for each tunable hyperparameter.

    Conservative defaults that work well for most datasets.
    Users can widen or tighten ranges in their config YAML.
    """

    lr0: tuple[float, float] = (0.001, 0.01)
    lrf: tuple[float, float] = (0.001, 0.01)
    batch: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])
    momentum: tuple[float, float] = (0.8, 0.98)
    weight_decay: tuple[float, float] = (0.0, 0.001)
    mosaic: tuple[float, float] = (0.0, 1.0)
    mixup: tuple[float, float] = (0.0, 0.5)
    copy_paste: tuple[float, float] = (0.0, 0.3)
    hsv_h: tuple[float, float] = (0.0, 0.05)
    hsv_s: tuple[float, float] = (0.0, 1.0)
    hsv_v: tuple[float, float] = (0.0, 1.0)
    degrees: tuple[float, float] = (0.0, 30.0)
    translate: tuple[float, float] = (0.0, 0.2)
    scale: tuple[float, float] = (0.1, 0.9)
    shear: tuple[float, float] = (0.0, 10.0)
    perspective: tuple[float, float] = (0.0, 0.001)
    flipud: tuple[float, float] = (0.0, 0.5)
    fliplr: tuple[float, float] = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Sub-system configuration models
# ---------------------------------------------------------------------------

class OptunaConfig(BaseModel):
    """Optuna optimizer configuration."""

    n_trials: int = 50
    search_strategy: Literal["bayesian", "random_walk", "simulated_annealing"] = "bayesian"
    yellow_strategy: Literal["random_walk", "simulated_annealing", "bayesian"] = "random_walk"
    n_startup_trials: int = 10
    pruner: Literal["median", "hyperband", "none"] = "none"
    random_walk_min_step_scale: float = 0.02
    search_space: OptunaSearchSpace = Field(default_factory=OptunaSearchSpace)

    def effective_yellow_strategy(self) -> str:
        """YELLOW escape strategy; legacy ``search_strategy`` overrides when non-bayesian."""
        if self.search_strategy in ("random_walk", "simulated_annealing"):
            return self.search_strategy
        return self.yellow_strategy


class DecisionConfig(BaseModel):
    """Three-state decision thresholds and escalation policy."""

    green_threshold_pct: float = 3.0
    green_threshold_abs: float | None = None
    red_threshold_pct: float = -5.0
    red_threshold_abs: float | None = None
    soft_red_threshold_pct: float = -3.0
    accept_marginal_improvement: bool = True
    marginal_green_use_optuna: bool = False
    red_escalation_count: int = 3
    yellow_resets_red_count: bool = True


class LLMConfig(BaseModel):
    """LLM API configuration (OpenAI-compatible, defaults to DeepSeek)."""

    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.3
    max_calls_per_session: int = 20
    guidance_enabled: bool = True
    guidance_fallback_regex: bool = True

    @field_validator("api_key")
    @classmethod
    def resolve_api_key(cls, v: str) -> str:
        if not v:
            v = os.environ.get("CV_AGENT_LLM_KEY", "")
        if not v:
            v = os.environ.get("DEEPSEEK_API_KEY", "")
        return v


class DataConfig(BaseModel):
    """Dataset configuration and validation thresholds."""

    data_yaml: Path = Path("data.yaml")
    min_images: int = 50
    min_ann_per_class: int = 1
    min_pixel_area: int = 64
    validate_brightness: bool = True
    validate_angles: bool = True


class CheckpointConfig(BaseModel):
    """Checkpoint Top-N and manual save settings."""

    top_n: int = Field(default=5, ge=1, le=50)
    auto_save_top: bool = True
    manual_save_dir: str = "manual"


class TrainConfig(BaseModel):
    """Top-level configuration aggregating all sub-systems."""

    data: DataConfig = Field(default_factory=DataConfig)
    model_variant: str = "yolo26s"
    epochs_per_round: int = 50
    max_rounds: int = 6
    interaction_mode: Literal["auto", "ask"] = "ask"
    auto_prompt_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    optimize_for_class: str | None = None
    # Ultralytics device: auto (all visible GPUs), cpu, 0, or 0,1,2,3 (DDP)
    device: str = "auto"
    workers: int | None = None  # DataLoader workers; None = 8 on Linux, 0 on Windows
    use_amp: bool = True  # false skips Ultralytics AMP check (no yolo26n.pt needed)
    initial_hyperparams: HyperParams = Field(default_factory=HyperParams)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "cv_agent"
    output_root: Path = Path("runs")
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("model_variant")
    @classmethod
    def validate_model_variant(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in YOLO_VARIANTS:
            raise ValueError(f"Unknown model variant '{v}'. Must be one of: {sorted(YOLO_VARIANTS)}")
        return v_lower