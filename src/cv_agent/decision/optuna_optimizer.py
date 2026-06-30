"""Optuna-driven hyperparameter optimization.

GREEN: TPESampler (Bayesian) via study.ask/tell.
YELLOW: random_walk | simulated_annealing | bayesian (configurable).
RED: handled by ThreeStateDecisionEngine — pending Optuna trials are abandoned.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from cv_agent.core.config import HyperParams, OptunaConfig
from cv_agent.decision.strategy import StrategyPatch
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna as the backend."""

    def __init__(self, config: OptunaConfig, study_db: Path | None = None) -> None:
        self.config = config
        self.search_space = config.search_space
        self._strategy_patch: StrategyPatch | None = None
        self._effective_search_space = self.search_space
        self._frozen_fields: set[str] = set()
        self.study_db = str(study_db) if study_db else "optuna_study.db"
        self._study = None
        self._pending_trial = None
        self._pending_params: HyperParams | None = None
        self._trial_count = 0

        self._sa_temperature: float = 1.0
        self._sa_decay: float = 0.9
        self._sa_best_params: HyperParams | None = None
        self._sa_best_score: float = 0.0

        self._rw_step_scale: float = 0.1

    def _init_study(self) -> None:
        if self._study is not None:
            return

        import optuna

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            multivariate=True,
            seed=42,
        )

        pruner = None
        if self.config.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        elif self.config.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()

        self._study = optuna.create_study(
            study_name="cv_agent_optuna",
            storage=f"sqlite:///{self.study_db}",
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True,
        )

        self._sync_trial_count_from_study()
        logger.info(
            f"Optuna study initialized ({len(self._study.trials)} prior trials, "
            f"budget used {self._trial_count}/{self.config.n_trials})"
        )

    @property
    def trial_count(self) -> int:
        return self._trial_count

    def set_trial_count(self, count: int) -> None:
        """Restore in-memory trial budget counter (e.g. on session resume)."""
        self._trial_count = max(0, count)

    def set_strategy_patch(self, patch: StrategyPatch | None) -> None:
        """Apply or clear strategy constraints for future Optuna proposals."""
        self._strategy_patch = patch
        self._effective_search_space = (
            patch.apply_to_search_space(self.search_space)
            if patch is not None
            else self.search_space
        )
        self._frozen_fields = set(patch.freeze) if patch is not None else set()

    def _sync_trial_count_from_study(self) -> None:
        if self._study is None:
            return
        import optuna

        terminal = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        )
        self._trial_count = sum(1 for t in self._study.trials if t.state in terminal)

    def report_result(self, score: float, actual_params: HyperParams) -> None:
        """Report the completed round for the pending Optuna trial, if params match."""
        if self._pending_trial is None or self._pending_params is None:
            return
        if self._study is None:
            return

        import optuna

        trial_num = self._pending_trial.number
        if self._pending_params.model_dump() == actual_params.model_dump():
            try:
                self._study.tell(trial_num, score)
                logger.info(f"Reported score {score:.4f} to Optuna trial #{trial_num}")
            except Exception as e:
                logger.warning(f"Failed to report to Optuna: {e}")
        else:
            try:
                self._study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
                logger.info(
                    f"Marked Optuna trial #{trial_num} as FAIL — "
                    "actual params differed from proposal."
                )
            except Exception as e:
                logger.warning(f"Failed to fail Optuna trial: {e}")

        self._pending_trial = None
        self._pending_params = None

    def abandon_pending(self) -> None:
        """Mark the pending trial as failed when it will not be evaluated."""
        if self._pending_trial is None or self._study is None:
            self._pending_trial = None
            self._pending_params = None
            return

        import optuna

        trial_num = self._pending_trial.number
        try:
            self._study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
            logger.info(f"Abandoned pending Optuna trial #{trial_num}")
        except Exception as e:
            logger.warning(f"Failed to abandon Optuna trial: {e}")

        self._pending_trial = None
        self._pending_params = None

    def propose_next(
        self,
        current_params: HyperParams,
        state: str,
        current_score: float | None = None,
    ) -> tuple[HyperParams, bool]:
        """Propose hyperparameters for the next round.

        Returns:
            (params, from_optuna) — ``from_optuna`` is True when params came from study.ask().
        """
        if state == "green":
            return self._propose_bayesian(current_params)

        if state == "yellow":
            yellow = self.config.effective_yellow_strategy()
            if yellow == "simulated_annealing":
                return self._propose_simulated_annealing(current_params, current_score), False
            if yellow == "bayesian":
                return self._propose_bayesian(current_params)
            return self._propose_random_walk(current_params), False

        return current_params, False

    def propose(
        self,
        current_params: HyperParams,
        current_score: float | None,
        state: str,
    ) -> HyperParams:
        """Legacy combined API — prefer report_result + propose_next."""
        if current_score is not None:
            self.report_result(current_score, current_params)
        params, _ = self.propose_next(current_params, state)
        return params

    def _trial_budget_exhausted(self) -> bool:
        return self._trial_count >= self.config.n_trials

    def _propose_bayesian(self, current_params: HyperParams) -> tuple[HyperParams, bool]:
        if self._trial_budget_exhausted():
            logger.info("Optuna trial budget exhausted; keeping current params.")
            return current_params, False

        self._init_study()
        trial = self._study.ask()
        self._pending_trial = trial
        params_dict = self._trial_to_params(trial)
        for field in self._frozen_fields:
            if hasattr(current_params, field):
                params_dict[field] = getattr(current_params, field)
        params = HyperParams(**params_dict)
        self._pending_params = params
        self._trial_count += 1

        logger.info(
            f"Optuna Bayesian trial #{self._trial_count}: "
            f"lr0={params.lr0:.5f}, batch={params.batch}, mosaic={params.mosaic:.3f}"
        )
        return params, True

    def _trial_to_params(self, trial) -> dict:
        ss = self._effective_search_space

        def suggest_float_range(name: str, bounds: tuple[float, float]) -> float:
            return trial.suggest_float(name, bounds[0], bounds[1])

        def suggest_categorical_int(name: str, choices: list[int]) -> int:
            return trial.suggest_categorical(name, choices)

        return {
            "lr0": suggest_float_range("lr0", ss.lr0),
            "lrf": suggest_float_range("lrf", ss.lrf),
            "batch": suggest_categorical_int("batch", ss.batch),
            "momentum": suggest_float_range("momentum", ss.momentum),
            "weight_decay": suggest_float_range("weight_decay", ss.weight_decay),
            "mosaic": suggest_float_range("mosaic", ss.mosaic),
            "mixup": suggest_float_range("mixup", ss.mixup),
            "copy_paste": suggest_float_range("copy_paste", ss.copy_paste),
            "hsv_h": suggest_float_range("hsv_h", ss.hsv_h),
            "hsv_s": suggest_float_range("hsv_s", ss.hsv_s),
            "hsv_v": suggest_float_range("hsv_v", ss.hsv_v),
            "degrees": suggest_float_range("degrees", ss.degrees),
            "translate": suggest_float_range("translate", ss.translate),
            "scale": suggest_float_range("scale", ss.scale),
            "shear": suggest_float_range("shear", ss.shear),
            "perspective": suggest_float_range("perspective", ss.perspective),
            "flipud": suggest_float_range("flipud", ss.flipud),
            "fliplr": suggest_float_range("fliplr", ss.fliplr),
        }

    def _neighbor_batch(self, current_batch: int) -> int:
        choices = sorted(self.search_space.batch)
        if not choices:
            return current_batch
        if current_batch not in choices:
            return random.choice(choices)
        idx = choices.index(current_batch)
        neighbors = [current_batch]
        if idx > 0:
            neighbors.append(choices[idx - 1])
        if idx < len(choices) - 1:
            neighbors.append(choices[idx + 1])
        return random.choice(neighbors)

    def _propose_random_walk(self, current_params: HyperParams) -> HyperParams:
        min_scale = self.config.random_walk_min_step_scale
        self._rw_step_scale = max(self._rw_step_scale * 0.95, min_scale)
        ss = self.search_space
        current = current_params.model_dump()
        perturbed: dict = {}

        for key, value in current.items():
            if key == "batch":
                perturbed[key] = self._neighbor_batch(int(value))
                continue

            bounds = getattr(ss, key, None)
            if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
                perturbed[key] = value
                continue

            low, high = bounds
            noise_std = (high - low) * self._rw_step_scale
            new_val = value + random.gauss(0, noise_std)
            perturbed[key] = max(low, min(high, new_val))

        logger.info(
            f"Random walk proposal (step_scale={self._rw_step_scale:.4f}): "
            f"lr0={perturbed['lr0']:.5f}, mosaic={perturbed['mosaic']:.3f}"
        )
        return HyperParams(**perturbed)

    def _propose_simulated_annealing(
        self,
        current_params: HyperParams,
        current_score: float | None = None,
    ) -> HyperParams:
        neighbor = self._random_neighbor(current_params)
        score = current_score if current_score is not None else 0.0

        if self._sa_best_params is None:
            self._sa_best_params = current_params
            self._sa_best_score = score
            self._sa_temperature = 1.0

        delta = score - self._sa_best_score
        if score > self._sa_best_score:
            self._sa_best_params = current_params
            self._sa_best_score = score

        if delta >= 0:
            result = neighbor
            accepted = True
        else:
            prob = math.exp(delta / max(self._sa_temperature, 1e-8))
            accepted = random.random() < prob
            result = neighbor if accepted else current_params
            if accepted:
                logger.info(f"SA accepted neighbor with probability {prob:.3f}")

        self._sa_temperature *= self._sa_decay
        self._sa_temperature = max(self._sa_temperature, 0.01)

        logger.info(
            f"Simulated annealing: T={self._sa_temperature:.4f}, "
            f"best_score={self._sa_best_score:.4f}, accepted={accepted}"
        )
        return result

    def _random_neighbor(self, params: HyperParams) -> HyperParams:
        ss = self.search_space
        current = params.model_dump()
        neighbor: dict = {}

        for key, value in current.items():
            if key == "batch":
                neighbor[key] = self._neighbor_batch(int(value))
                continue

            bounds = getattr(ss, key, None)
            if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
                neighbor[key] = value
                continue

            low, high = bounds
            range_size = (high - low) * 0.2
            new_val = value + random.uniform(-range_size, range_size)
            neighbor[key] = max(low, min(high, new_val))

        return HyperParams(**neighbor)

    def _report_result(self, params: HyperParams, score: float) -> None:
        """Backward-compatible alias."""
        self.report_result(score, params)
