# AI Strategy Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise cv_agent from rule/Optuna-driven auto-tuning to an AI-assisted strategy system where the LLM plans search phases and constraints while Optuna keeps precise numeric optimization.

**Architecture:** Add a strategy layer above the existing Green/Yellow/Red decision flow. The LLM outputs structured, bounded `StrategyPatch` objects: search-space changes, frozen params, objective-weight changes, phase choice, and stop/retry hints. Existing rule logic remains the safety layer, and Optuna remains the numeric optimizer.

**Tech Stack:** Python 3.10+, Pydantic v2, Optuna, pytest, existing OpenAI-compatible `LLMAdvisor`, existing `TrainingEngine`.

---

## File Structure

- Modify `src/cv_agent/core/config.py`
  - Add strategy-related config: enable/disable strategy planner, cadence, max patch bounds, objective weights.
- Create `src/cv_agent/decision/strategy.py`
  - Owns `StrategyPhase`, `ObjectiveWeights`, `SearchSpacePatch`, `StrategyPatch`, validation, and application helpers.
- Modify `src/cv_agent/decision/llm_advisor.py`
  - Add `plan_strategy()` and a strict JSON prompt for strategy patches.
- Modify `src/cv_agent/decision/optuna_optimizer.py`
  - Allow an effective per-phase search space and frozen parameter set without changing the base config.
- Modify `src/cv_agent/trainer/evaluator.py`
  - Centralize reward calculation behind objective weights.
- Modify `src/cv_agent/core/engine.py`
  - Call strategy planner at a controlled cadence after evaluation and before Optuna proposal.
- Create `src/cv_agent/decision/strategy_memory.py`
  - Persist long-running experiment lessons across rounds.
- Modify `src/cv_agent/tracking/run_dir.py`
  - Save/load `strategy_log.json` and `strategy_memory.json`.
- Add tests:
  - `tests/test_strategy_patch.py`
  - `tests/test_strategy_planner.py`
  - `tests/test_optuna_strategy_constraints.py`
  - `tests/test_strategy_memory.py`
  - `tests/test_engine_strategy_flow.py`

---

### Task 1: Fix Reliability Blockers Before Adding AI Strategy

**Files:**
- Modify: `src/cv_agent/data/bootstrap.py`
- Modify: `src/cv_agent/cli/main.py`
- Modify: `src/cv_agent/trainer/yolo_trainer.py`
- Modify: `src/cv_agent/core/engine.py`
- Test: `tests/test_bootstrap.py`
- Test: `tests/test_cli_config.py`
- Test: `tests/test_engine_train_failure.py`

- [ ] **Step 1: Write a failing bootstrap logging import test**

Add this to `tests/test_bootstrap.py`:

```python
def test_ensure_dataset_none_uses_console_logging_without_name_error(monkeypatch, tmp_path):
    from cv_agent.data import bootstrap

    expected = tmp_path / "coco.yaml"
    expected.write_text(
        "path: .\ntrain: images\nval: images\nnames: {0: object}\n",
        encoding="utf-8",
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_text("x", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "resolve_datasets_dir", lambda datasets_dir=None: tmp_path)
    monkeypatch.setattr(bootstrap, "_download_registry_dataset", lambda datasets_dir, name: expected)

    assert bootstrap.ensure_dataset(None) == expected
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest tests/test_bootstrap.py::test_ensure_dataset_none_uses_console_logging_without_name_error -v
```

Expected: FAIL with `NameError: name 'log_info' is not defined`.

- [ ] **Step 3: Import the console logging helpers**

In `src/cv_agent/data/bootstrap.py`, add:

```python
from cv_agent.ui.console import log_info, log_success, log_warning
```

- [ ] **Step 4: Add CLI SessionQuit coverage**

Add this to `tests/test_cli_config.py`:

```python
def test_prompt_start_mode_resume_cancel_exits(monkeypatch, tmp_path):
    import sys

    import pytest

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
```

- [ ] **Step 5: Fix CLI SessionQuit import**

In `src/cv_agent/cli/main.py`, inside the `start_override == "resume"` branch, add:

```python
from cv_agent.interaction.types import SessionQuit
```

before the `try`.

- [ ] **Step 6: Write a training failure round-count test**

Create `tests/test_engine_train_failure.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.core.state_machine import TrainingLoopState


def test_training_failure_does_not_consume_successful_round(tmp_path: Path):
    engine = TrainingEngine()
    engine._config = TrainConfig(max_rounds=2, output_root=tmp_path)
    engine._run_dir = tmp_path
    engine._current_params = HyperParams()
    engine._round_num = 0
    engine._best_checkpoint = None
    engine._decision_log = []
    engine._mlflow = MagicMock()
    engine._yolo_trainer = MagicMock()
    engine._yolo_trainer.train.side_effect = RuntimeError("cuda transient")

    engine._do_train()

    assert engine._round_num == 0
    assert engine._state is TrainingLoopState.TRAIN
    assert engine._decision_log[0]["action"] == "training_crash"
```

- [ ] **Step 7: Make failed training attempts non-consuming**

In `src/cv_agent/core/engine.py`, change `_do_train()` so it stores `attempt_round = self._round_num + 1`, uses that for logging and MLflow, and only assigns `self._round_num = attempt_round` after training succeeds. The critical shape is:

```python
attempt_round = self._round_num + 1
print_section(f"Training Round {attempt_round}/{self._config.max_rounds}")
self._mlflow.start_round(attempt_round)

try:
    ...
    self._round_num = attempt_round
    self._last_artifacts = artifacts
    self._state = TrainingLoopState.EVALUATE
except Exception as e:
    ...
    self._decision_log.append({"round": attempt_round, ...})
    self._mlflow.end_round()
    self._state = TrainingLoopState.TRAIN
```

- [ ] **Step 8: Run targeted reliability tests**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest tests/test_bootstrap.py tests/test_cli_config.py tests/test_engine_train_failure.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add src/cv_agent/data/bootstrap.py src/cv_agent/cli/main.py src/cv_agent/core/engine.py tests/test_bootstrap.py tests/test_cli_config.py tests/test_engine_train_failure.py
git commit -m "fix: harden autonomous run reliability"
```

---

### Task 2: Add Strategy Patch Data Model

**Files:**
- Create: `src/cv_agent/decision/strategy.py`
- Modify: `src/cv_agent/core/config.py`
- Test: `tests/test_strategy_patch.py`

- [ ] **Step 1: Write strategy model tests**

Create `tests/test_strategy_patch.py`:

```python
from __future__ import annotations

from cv_agent.core.config import OptunaSearchSpace
from cv_agent.decision.strategy import ObjectiveWeights, StrategyPatch, StrategyPhase


def test_strategy_patch_clamps_search_space_to_base_bounds():
    base = OptunaSearchSpace(lr0=(0.001, 0.01), mosaic=(0.0, 1.0))
    patch = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="recover from red rounds",
        search_space_patch={"lr0": (0.00001, 0.02), "mosaic": (0.2, 0.8)},
    )

    effective = patch.apply_to_search_space(base)

    assert effective.lr0 == (0.001, 0.01)
    assert effective.mosaic == (0.2, 0.8)


def test_objective_weights_normalize_positive_values():
    weights = ObjectiveWeights(map50_95=2.0, recall=1.0, precision=1.0)
    normalized = weights.normalized()

    assert round(normalized.map50_95, 6) == 0.5
    assert round(normalized.recall, 6) == 0.25
    assert round(normalized.precision, 6) == 0.25
```

- [ ] **Step 2: Run tests to verify missing module failure**

Run:

```powershell
python -m pytest tests/test_strategy_patch.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement strategy models**

Create `src/cv_agent/decision/strategy.py`:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cv_agent.core.config import OptunaSearchSpace


class StrategyPhase(StrEnum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    RECOVERY = "recovery"
    DATA_GAP = "data_gap"
    STABILITY_CHECK = "stability_check"


class ObjectiveWeights(BaseModel):
    map50_95: float = Field(default=0.45, ge=0.0)
    map50: float = Field(default=0.15, ge=0.0)
    recall: float = Field(default=0.20, ge=0.0)
    precision: float = Field(default=0.10, ge=0.0)
    overfit_penalty: float = Field(default=0.10, ge=0.0)
    cost_penalty: float = Field(default=0.0, ge=0.0)

    def normalized(self) -> "ObjectiveWeights":
        data = self.model_dump()
        total = sum(float(v) for v in data.values())
        if total <= 0:
            return ObjectiveWeights()
        return ObjectiveWeights(**{k: float(v) / total for k, v in data.items()})


class StrategyPatch(BaseModel):
    phase: StrategyPhase = StrategyPhase.EXPLORATION
    reason: str = ""
    search_space_patch: dict[str, tuple[float, float]] = Field(default_factory=dict)
    freeze: set[str] = Field(default_factory=set)
    objective_weights: ObjectiveWeights | None = None
    max_trials_for_phase: int | None = Field(default=None, ge=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("freeze")
    @classmethod
    def validate_freeze_fields(cls, value: set[str]) -> set[str]:
        valid = set(OptunaSearchSpace.model_fields.keys())
        return {field for field in value if field in valid}

    def apply_to_search_space(self, base: OptunaSearchSpace) -> OptunaSearchSpace:
        data = base.model_dump()
        for key, bounds in self.search_space_patch.items():
            if key not in data:
                continue
            base_bounds = data[key]
            if not isinstance(base_bounds, tuple) or len(base_bounds) != 2:
                continue
            low = max(float(bounds[0]), float(base_bounds[0]))
            high = min(float(bounds[1]), float(base_bounds[1]))
            if low <= high:
                data[key] = (low, high)
        return OptunaSearchSpace(**data)
```

- [ ] **Step 4: Add strategy config**

In `src/cv_agent/core/config.py`, add:

```python
class StrategyConfig(BaseModel):
    enabled: bool = True
    planner_cadence: int = Field(default=1, ge=1)
    min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    memory_enabled: bool = True
    max_memory_items: int = Field(default=50, ge=1, le=500)
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
```

Import `ObjectiveWeights` carefully to avoid circular imports. If direct import creates a cycle, place `ObjectiveWeights` in `core/config.py` and import it from there in `strategy.py`.

Then add this field to `TrainConfig`:

```python
strategy: StrategyConfig = Field(default_factory=StrategyConfig)
```

- [ ] **Step 5: Run strategy tests**

Run:

```powershell
python -m pytest tests/test_strategy_patch.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/cv_agent/core/config.py src/cv_agent/decision/strategy.py tests/test_strategy_patch.py
git commit -m "feat: add strategy patch model"
```

---

### Task 3: Add LLM Strategy Planner

**Files:**
- Modify: `src/cv_agent/decision/llm_advisor.py`
- Test: `tests/test_strategy_planner.py`

- [ ] **Step 1: Write planner fallback test**

Create `tests/test_strategy_planner.py`:

```python
from __future__ import annotations

from cv_agent.core.config import LLMConfig
from cv_agent.decision.llm_advisor import LLMAdvisor
from cv_agent.decision.strategy import StrategyPhase


def test_plan_strategy_without_api_key_returns_heuristic_recovery_patch():
    advisor = LLMAdvisor(LLMConfig(api_key=""))

    patch = advisor.plan_strategy(
        round_num=4,
        decision_summary={"color": "red", "action": "rollback", "reason": "score dropped"},
        metrics={"mAP50-95": 0.31, "recall": 0.24, "precision": 0.6},
        history=[{"round": 1, "score": 0.4}, {"round": 2, "score": 0.35}],
        memory={},
        base_search_space={},
    )

    assert patch.phase is StrategyPhase.RECOVERY
    assert "lr0" in patch.search_space_patch
    assert patch.confidence > 0
```

- [ ] **Step 2: Run test to verify method is missing**

Run:

```powershell
python -m pytest tests/test_strategy_planner.py -v
```

Expected: FAIL with `AttributeError: 'LLMAdvisor' object has no attribute 'plan_strategy'`.

- [ ] **Step 3: Add strategy prompt and parser**

In `src/cv_agent/decision/llm_advisor.py`, import:

```python
from cv_agent.decision.strategy import ObjectiveWeights, StrategyPatch, StrategyPhase
```

Add a prompt constant:

```python
STRATEGY_PROMPT = """You are a computer vision training strategist.

You do not output exact hyperparameter values. You output bounded strategy patches
that constrain Optuna and adjust objective weighting.

Return JSON only:
{
  "phase": "exploration | exploitation | recovery | data_gap | stability_check",
  "reason": "brief factual reason",
  "search_space_patch": {"lr0": [low, high], "mosaic": [low, high]},
  "freeze": ["batch"],
  "objective_weights": {
    "map50_95": 0.45,
    "map50": 0.15,
    "recall": 0.20,
    "precision": 0.10,
    "overfit_penalty": 0.10,
    "cost_penalty": 0.0
  },
  "max_trials_for_phase": 5,
  "confidence": 0.0
}

Decision summary:
{decision_summary}

Metrics:
{metrics}

History:
{history}

Strategy memory:
{memory}

Base search space:
{base_search_space}
"""
```

- [ ] **Step 4: Implement `plan_strategy()`**

Add this method to `LLMAdvisor`:

```python
def plan_strategy(
    self,
    round_num: int,
    decision_summary: dict[str, Any],
    metrics: dict[str, float],
    history: list[dict[str, Any]],
    memory: dict[str, Any],
    base_search_space: dict[str, Any],
) -> StrategyPatch:
    if self._client is not None and self._call_count < self.config.max_calls_per_session:
        prompt = STRATEGY_PROMPT.format(
            decision_summary=json.dumps(decision_summary, indent=2, default=str),
            metrics=json.dumps(metrics, indent=2, default=str),
            history=json.dumps(history[-10:], indent=2, default=str),
            memory=json.dumps(memory, indent=2, default=str),
            base_search_space=json.dumps(base_search_space, indent=2, default=str),
        )
        response = self._call_llm(prompt)
        if response is not None:
            try:
                data = json.loads(response)
                return StrategyPatch(
                    phase=StrategyPhase(data.get("phase", "exploration")),
                    reason=str(data.get("reason") or ""),
                    search_space_patch={
                        str(k): (float(v[0]), float(v[1]))
                        for k, v in (data.get("search_space_patch") or {}).items()
                        if isinstance(v, list | tuple) and len(v) == 2
                    },
                    freeze=set(data.get("freeze") or []),
                    objective_weights=ObjectiveWeights(**data["objective_weights"])
                    if isinstance(data.get("objective_weights"), dict)
                    else None,
                    max_trials_for_phase=data.get("max_trials_for_phase"),
                    confidence=float(data.get("confidence", 0.5)),
                    metadata={"source": "llm", "round": round_num},
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse strategy planner response: {e}")

    return self._heuristic_plan_strategy(round_num, decision_summary, metrics)
```

Then add:

```python
def _heuristic_plan_strategy(
    self,
    round_num: int,
    decision_summary: dict[str, Any],
    metrics: dict[str, float],
) -> StrategyPatch:
    color = str(decision_summary.get("color", "")).lower()
    recall = float(metrics.get("recall", metrics.get("metrics/recall(B)", 0.0)) or 0.0)
    if color == "red":
        return StrategyPatch(
            phase=StrategyPhase.RECOVERY,
            reason="red decision triggered conservative recovery search",
            search_space_patch={"lr0": (0.001, 0.004), "mosaic": (0.0, 0.5)},
            freeze={"batch"},
            objective_weights=ObjectiveWeights(recall=0.35, overfit_penalty=0.25).normalized(),
            confidence=0.6,
            metadata={"source": "heuristic", "round": round_num},
        )
    if recall < 0.35:
        return StrategyPatch(
            phase=StrategyPhase.DATA_GAP,
            reason="low recall suggests data or class coverage gap",
            objective_weights=ObjectiveWeights(recall=0.4, precision=0.1).normalized(),
            confidence=0.55,
            metadata={"source": "heuristic", "round": round_num},
        )
    return StrategyPatch(
        phase=StrategyPhase.EXPLORATION,
        reason="no strong failure pattern; continue broad search",
        confidence=0.5,
        metadata={"source": "heuristic", "round": round_num},
    )
```

- [ ] **Step 5: Run planner tests**

Run:

```powershell
python -m pytest tests/test_strategy_planner.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/cv_agent/decision/llm_advisor.py tests/test_strategy_planner.py
git commit -m "feat: add llm strategy planner"
```

---

### Task 4: Let Optuna Execute Strategy Constraints

**Files:**
- Modify: `src/cv_agent/decision/optuna_optimizer.py`
- Test: `tests/test_optuna_strategy_constraints.py`

- [ ] **Step 1: Write constraints test**

Create `tests/test_optuna_strategy_constraints.py`:

```python
from __future__ import annotations

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.optuna_optimizer import OptunaOptimizer
from cv_agent.decision.strategy import StrategyPatch, StrategyPhase


def test_strategy_patch_constrains_bayesian_proposal(tmp_path):
    optimizer = OptunaOptimizer(OptunaConfig(n_trials=3), study_db=tmp_path / "study.db")
    optimizer.set_strategy_patch(
        StrategyPatch(
            phase=StrategyPhase.RECOVERY,
            search_space_patch={"lr0": (0.001, 0.002), "mosaic": (0.0, 0.1)},
            freeze={"batch"},
        )
    )

    params, from_optuna = optimizer.propose_next(HyperParams(batch=16), "green", current_score=0.5)

    assert from_optuna is True
    assert 0.001 <= params.lr0 <= 0.002
    assert 0.0 <= params.mosaic <= 0.1
    assert params.batch == 16
```

- [ ] **Step 2: Run test to verify missing behavior**

Run:

```powershell
python -m pytest tests/test_optuna_strategy_constraints.py -v
```

Expected: FAIL because `set_strategy_patch` is missing.

- [ ] **Step 3: Add strategy patch state to optimizer**

In `OptunaOptimizer.__init__`, add:

```python
self._strategy_patch = None
self._effective_search_space = self.search_space
self._frozen_fields: set[str] = set()
```

Import:

```python
from cv_agent.decision.strategy import StrategyPatch
```

Add:

```python
def set_strategy_patch(self, patch: StrategyPatch | None) -> None:
    self._strategy_patch = patch
    self._effective_search_space = (
        patch.apply_to_search_space(self.search_space) if patch is not None else self.search_space
    )
    self._frozen_fields = set(patch.freeze) if patch is not None else set()
```

- [ ] **Step 4: Use effective search space and frozen fields**

In `_trial_to_params`, replace:

```python
ss = self.search_space
```

with:

```python
ss = self._effective_search_space
```

After creating `params_dict` in `_propose_bayesian`, add:

```python
for field in self._frozen_fields:
    if hasattr(current_params, field):
        params_dict[field] = getattr(current_params, field)
```

- [ ] **Step 5: Run constraints test**

Run:

```powershell
python -m pytest tests/test_optuna_strategy_constraints.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/cv_agent/decision/optuna_optimizer.py tests/test_optuna_strategy_constraints.py
git commit -m "feat: constrain optuna with strategy patches"
```

---

### Task 5: Add Strategy Memory

**Files:**
- Create: `src/cv_agent/decision/strategy_memory.py`
- Modify: `src/cv_agent/tracking/run_dir.py`
- Test: `tests/test_strategy_memory.py`

- [ ] **Step 1: Write memory tests**

Create `tests/test_strategy_memory.py`:

```python
from __future__ import annotations

from cv_agent.decision.strategy import StrategyPatch, StrategyPhase
from cv_agent.decision.strategy_memory import StrategyMemory


def test_strategy_memory_records_effective_and_avoid_patterns():
    memory = StrategyMemory(max_items=3)

    memory.record_round(
        patch=StrategyPatch(phase=StrategyPhase.RECOVERY, reason="lower mosaic"),
        before_score=0.4,
        after_score=0.45,
        params={"mosaic": 0.2},
    )
    memory.record_round(
        patch=StrategyPatch(phase=StrategyPhase.EXPLORATION, reason="high lr"),
        before_score=0.45,
        after_score=0.30,
        params={"lr0": 0.01},
    )

    data = memory.model_dump()

    assert data["effective_patterns"]
    assert data["avoid_patterns"]
```

- [ ] **Step 2: Run test to verify missing module**

Run:

```powershell
python -m pytest tests/test_strategy_memory.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement memory model**

Create `src/cv_agent/decision/strategy_memory.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cv_agent.decision.strategy import StrategyPatch


class StrategyMemory(BaseModel):
    max_items: int = 50
    effective_patterns: list[str] = Field(default_factory=list)
    avoid_patterns: list[str] = Field(default_factory=list)
    open_hypotheses: list[str] = Field(default_factory=list)

    def record_round(
        self,
        patch: StrategyPatch,
        before_score: float,
        after_score: float,
        params: dict[str, Any],
    ) -> None:
        delta = after_score - before_score
        summary = f"{patch.phase.value}: {patch.reason}; params={params}"
        if delta > 0:
            self.effective_patterns.append(summary)
        elif delta < 0:
            self.avoid_patterns.append(summary)
        self.effective_patterns = self.effective_patterns[-self.max_items :]
        self.avoid_patterns = self.avoid_patterns[-self.max_items :]
        self.open_hypotheses = self.open_hypotheses[-self.max_items :]
```

- [ ] **Step 4: Add run_dir persistence helpers**

In `src/cv_agent/tracking/run_dir.py`, add:

```python
def save_strategy_memory(run_dir: Path, memory: dict[str, Any]) -> None:
    path = run_dir / "strategy_memory.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(memory, fh, indent=2, default=str, ensure_ascii=False)


def load_strategy_memory(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "strategy_memory.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
```

- [ ] **Step 5: Run memory tests**

Run:

```powershell
python -m pytest tests/test_strategy_memory.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/cv_agent/decision/strategy_memory.py src/cv_agent/tracking/run_dir.py tests/test_strategy_memory.py
git commit -m "feat: persist strategy memory"
```

---

### Task 6: Integrate Strategy Planner Into Engine Flow

**Files:**
- Modify: `src/cv_agent/core/engine.py`
- Modify: `src/cv_agent/tracking/run_dir.py`
- Test: `tests/test_engine_strategy_flow.py`

- [ ] **Step 1: Write engine integration test**

Create `tests/test_engine_strategy_flow.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.engine import TrainingEngine
from cv_agent.decision.strategy import StrategyPatch, StrategyPhase
from cv_agent.trainer.evaluator import RoundResult


def test_engine_applies_strategy_patch_before_next_proposal(tmp_path: Path):
    engine = TrainingEngine()
    engine._config = TrainConfig(output_root=tmp_path, max_rounds=2)
    engine._run_dir = tmp_path
    engine._round_num = 1
    engine._current_params = HyperParams()
    engine._history = [RoundResult(round_num=1, run_dir=tmp_path, score=0.5, metrics={"recall": 0.3})]
    engine._last_round_result = engine._history[0]
    engine._decision_log = []
    engine._llm_advisor = MagicMock()
    engine._llm_advisor.plan_strategy.return_value = StrategyPatch(
        phase=StrategyPhase.RECOVERY,
        reason="test recovery",
        search_space_patch={"lr0": (0.001, 0.002)},
    )
    engine._optuna = MagicMock()

    decision_summary = {"color": "red", "action": "rollback", "reason": "drop"}
    patch = engine._plan_strategy(decision_summary)

    assert patch.phase is StrategyPhase.RECOVERY
    engine._optuna.set_strategy_patch.assert_called_once()
```

- [ ] **Step 2: Run test to verify missing `_plan_strategy`**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest tests/test_engine_strategy_flow.py -v
```

Expected: FAIL with missing method.

- [ ] **Step 3: Add engine fields**

In `TrainingEngine.__init__`, add:

```python
self._strategy_log: list[dict[str, Any]] = []
self._strategy_memory = None
self._active_strategy_patch = None
```

- [ ] **Step 4: Initialize strategy memory**

In `_setup_subsystems`, after LLM advisor setup:

```python
from cv_agent.decision.strategy_memory import StrategyMemory

self._strategy_memory = StrategyMemory(max_items=config.strategy.max_memory_items)
```

When resuming, load persisted memory after `_setup_subsystems`:

```python
from cv_agent.decision.strategy_memory import StrategyMemory
from cv_agent.tracking.run_dir import load_strategy_memory

memory_data = load_strategy_memory(run_dir)
self._strategy_memory = StrategyMemory(**memory_data) if memory_data else self._strategy_memory
```

- [ ] **Step 5: Implement `_plan_strategy`**

Add to `TrainingEngine`:

```python
def _plan_strategy(self, decision_summary: dict[str, Any]):
    if not self._config.strategy.enabled:
        return None
    if self._round_num % self._config.strategy.planner_cadence != 0:
        return self._active_strategy_patch

    round_result = getattr(self, "_last_round_result", None)
    metrics = round_result.metrics if round_result is not None else {}
    memory = self._strategy_memory.model_dump() if self._strategy_memory is not None else {}
    history = [
        {"round": r.round_num, "score": r.score, "metrics": r.metrics}
        for r in self._history[-10:]
    ]
    patch = self._llm_advisor.plan_strategy(
        round_num=self._round_num,
        decision_summary=decision_summary,
        metrics=metrics,
        history=history,
        memory=memory,
        base_search_space=self._config.optuna.search_space.model_dump(),
    )
    if patch.confidence < self._config.strategy.min_confidence:
        return self._active_strategy_patch

    self._active_strategy_patch = patch
    if self._optuna is not None:
        self._optuna.set_strategy_patch(patch)
    self._strategy_log.append(patch.model_dump())
    return patch
```

- [ ] **Step 6: Call planner during decision**

In `_do_decide`, after `decision = self._decision_engine.decide(...)`, add:

```python
strategy_patch = self._plan_strategy(decision.to_dict())
if strategy_patch is not None:
    decision.metadata["strategy_patch"] = strategy_patch.model_dump()
```

Keep existing rollback and Ask/Auto behavior unchanged.

- [ ] **Step 7: Persist strategy artifacts**

In `_save_session_state`, after `save_session_state(...)`, add:

```python
from cv_agent.tracking.run_dir import save_strategy_memory

if self._strategy_memory is not None:
    save_strategy_memory(self._run_dir, self._strategy_memory.model_dump())
save_artifacts(run_dir=self._run_dir, decision_log=self._decision_log)
```

Add `strategy_log` to artifact saving only if `save_artifacts` is extended for it; otherwise save it with a small dedicated helper `save_strategy_log`.

- [ ] **Step 8: Run engine strategy test**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest tests/test_engine_strategy_flow.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```powershell
git add src/cv_agent/core/engine.py src/cv_agent/tracking/run_dir.py tests/test_engine_strategy_flow.py
git commit -m "feat: integrate strategy planner into engine"
```

---

### Task 7: Add Objective-Weighted Scoring

**Files:**
- Modify: `src/cv_agent/trainer/evaluator.py`
- Modify: `src/cv_agent/core/engine.py`
- Test: `tests/test_evaluator_strategy_objective.py`

- [ ] **Step 1: Write objective scoring test**

Create `tests/test_evaluator_strategy_objective.py`:

```python
from __future__ import annotations

from cv_agent.decision.strategy import ObjectiveWeights
from cv_agent.trainer.evaluator import compute_weighted_score


def test_compute_weighted_score_uses_recall_and_overfit_penalty():
    metrics = {
        "mAP50-95": 0.5,
        "mAP50": 0.7,
        "recall": 0.2,
        "precision": 0.8,
        "overfit_penalty": 0.3,
    }
    weights = ObjectiveWeights(
        map50_95=0.4,
        map50=0.1,
        recall=0.4,
        precision=0.1,
        overfit_penalty=0.2,
    )

    score = compute_weighted_score(metrics, weights)

    assert round(score, 3) == 0.370
```

- [ ] **Step 2: Run test to verify missing function**

Run:

```powershell
python -m pytest tests/test_evaluator_strategy_objective.py -v
```

Expected: FAIL with missing `compute_weighted_score`.

- [ ] **Step 3: Implement weighted score function**

In `src/cv_agent/trainer/evaluator.py`, add:

```python
from cv_agent.decision.strategy import ObjectiveWeights


def compute_weighted_score(metrics: dict[str, float], weights: ObjectiveWeights) -> float:
    w = weights.normalized()
    map50_95 = float(metrics.get("mAP50-95", metrics.get("metrics/mAP50-95(B)", 0.0)) or 0.0)
    map50 = float(metrics.get("mAP50", metrics.get("metrics/mAP50(B)", 0.0)) or 0.0)
    recall = float(metrics.get("recall", metrics.get("metrics/recall(B)", 0.0)) or 0.0)
    precision = float(metrics.get("precision", metrics.get("metrics/precision(B)", 0.0)) or 0.0)
    overfit_penalty = float(metrics.get("overfit_penalty", 0.0) or 0.0)
    cost_penalty = float(metrics.get("cost_penalty", 0.0) or 0.0)
    return (
        map50_95 * w.map50_95
        + map50 * w.map50
        + recall * w.recall
        + precision * w.precision
        - overfit_penalty * w.overfit_penalty
        - cost_penalty * w.cost_penalty
    )
```

- [ ] **Step 4: Wire active strategy weights into evaluation**

In `TrainingEngine._do_evaluate`, after `round_result` is enriched, if an active strategy patch has `objective_weights`, recompute `round_result.score`:

```python
from cv_agent.trainer.evaluator import compute_weighted_score

if self._active_strategy_patch and self._active_strategy_patch.objective_weights:
    round_result.score = compute_weighted_score(
        round_result.metrics,
        self._active_strategy_patch.objective_weights,
    )
```

- [ ] **Step 5: Run scoring tests**

Run:

```powershell
python -m pytest tests/test_evaluator_strategy_objective.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/cv_agent/trainer/evaluator.py src/cv_agent/core/engine.py tests/test_evaluator_strategy_objective.py
git commit -m "feat: support strategy-weighted objective scoring"
```

---

### Task 8: Full Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `cv_agent.yaml`
- Modify: `cv_agent.quick.yaml`

- [ ] **Step 1: Add config defaults**

Add to `cv_agent.yaml` and `cv_agent.quick.yaml`:

```yaml
strategy:
  enabled: true
  planner_cadence: 1
  min_confidence: 0.35
  memory_enabled: true
  max_memory_items: 50
  objective_weights:
    map50_95: 0.45
    map50: 0.15
    recall: 0.20
    precision: 0.10
    overfit_penalty: 0.10
    cost_penalty: 0.0
```

- [ ] **Step 2: Document AI strategy division of labor**

Add a README section:

```markdown
### AI strategy planner

The LLM does not emit exact training hyperparameters. It emits bounded strategy
patches: search-space narrowing, frozen fields, objective weights, and phase
selection. Optuna still proposes precise numeric values inside those validated
bounds. This keeps numeric optimization deterministic and auditable while using
the LLM for diagnosis and strategy selection.
```

- [ ] **Step 3: Run targeted tests**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest tests/test_strategy_patch.py tests/test_strategy_planner.py tests/test_optuna_strategy_constraints.py tests/test_strategy_memory.py tests/test_engine_strategy_flow.py tests/test_evaluator_strategy_objective.py -v
```

Expected: PASS.

- [ ] **Step 4: Run full suite**

Run:

```powershell
$env:YOLO_CONFIG_DIR='C:\Workspace\cv_agent\.tmp_ultralytics'; python -m pytest
```

Expected: PASS.

- [ ] **Step 5: Run lint**

Run:

```powershell
python -m ruff check .
```

Expected: PASS after either fixing existing lint issues or explicitly scoping this feature branch to new files and separately scheduling lint cleanup.

- [ ] **Step 6: Commit**

```powershell
git add README.md cv_agent.yaml cv_agent.quick.yaml
git commit -m "docs: describe ai strategy planner"
```

---

## Self-Review

- Spec coverage: The plan keeps Optuna responsible for exact numeric proposals, adds LLM strategy patches, adds objective-weight control, adds long-term memory, and integrates strategy planning into the engine.
- Placeholder scan: No `TBD`, `TODO`, or undefined future-only module remains without a task creating it first.
- Type consistency: `StrategyPatch`, `StrategyPhase`, `ObjectiveWeights`, `StrategyMemory`, and `set_strategy_patch()` are introduced before later tasks consume them.
- Scope check: Data auto-download/merge/annotation is intentionally not included. This plan raises AI strategy for training optimization first; autonomous data acquisition should be a separate plan.
