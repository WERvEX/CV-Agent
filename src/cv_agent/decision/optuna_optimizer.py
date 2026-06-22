"""Optuna-driven hyperparameter optimization.

Three search strategies:
1. Bayesian (default): TPESampler with multivariate=True
2. Random Walk (Yellow state): Gaussian perturbation with decaying step
3. Simulated Annealing (Yellow state): Accept worse params with exp(-delta/T)

Study state is persisted to optuna_study.db for resume support.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from cv_agent.core.config import HyperParams, OptunaConfig, OptunaSearchSpace
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna as the backend."""

    def __init__(self, config: OptunaConfig, study_db: Path | None = None) -> None:
        """Initialize the optimizer.

        Args:
            config: OptunaConfig with strategy, trial count, search space.
            study_db: Path to SQLite file for study persistence.
        """
        self.config = config
        self.search_space = config.search_space
        self.study_db = str(study_db) if study_db else "optuna_study.db"
        self._study = None
        self._trial_count = 0

        # Simulated annealing state
        self._sa_temperature: float = 1.0
        self._sa_decay: float = 0.9
        self._sa_best_params: HyperParams | None = None
        self._sa_best_score: float = 0.0

        # Random walk state
        self._rw_step_scale: float = 0.1

    # ------------------------------------------------------------------
    # Study initialization
    # ------------------------------------------------------------------

    def _init_study(self) -> None:
        """Lazy-initialize the Optuna study."""
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

        logger.info(f"Optuna study initialized ({len(self._study.trials)} prior trials loaded)")

    # ------------------------------------------------------------------
    # Hyperparameter proposal
    # ------------------------------------------------------------------

    def propose(
        self,
        current_params: HyperParams,
        current_score: float | None,
        state: str,  # "green" | "yellow" | "red"
    ) -> HyperParams:
        """Propose the next set of hyperparameters based on state.

        Args:
            current_params: The HyperParams used in the current round.
            current_score: The reward score achieved this round.
            state: Decision color — "green" → Bayesian, "yellow" → RW/SA, "red" → perturbation.

        Returns:
            Proposed HyperParams for the next round.
        """
        # Report current trial result to Optuna if a study is initialized
        if self._study is not None and current_score is not None and self._trial_count > 0:
            self._report_result(current_params, current_score)

        if state == "green":
            return self._propose_bayesian()
        elif state == "yellow":
            if self.config.search_strategy == "simulated_annealing":
                return self._propose_simulated_annealing(current_params, current_score)
            else:
                return self._propose_random_walk(current_params)
        else:  # red
            # For Red, the three_state engine already adjusted params — return as-is
            return current_params

    # ------------------------------------------------------------------
    # Bayesian search (via Optuna TPESampler)
    # ------------------------------------------------------------------

    def _propose_bayesian(self) -> HyperParams:
        """Use Optuna TPESampler to suggest the next trial."""
        self._init_study()
        import optuna

        trial = self._study.ask()

        params_dict = self._trial_to_params(trial)
        self._trial_count += 1

        logger.info(
            f"Optuna Bayesian trial #{self._trial_count}: "
            f"lr0={params_dict['lr0']:.5f}, batch={params_dict['batch']}, "
            f"mosaic={params_dict['mosaic']:.3f}, mixup={params_dict['mixup']:.3f}"
        )
        return HyperParams(**params_dict)

    def _report_result(self, params: HyperParams, score: float) -> None:
        """Report a completed trial result back to Optuna."""
        import optuna

        # Find the trial number (last asked trial)
        trial_num = len(self._study.trials)

        # We use a fixed-trial reporting approach since we can't easily correlate
        # Optuna trial numbers with our round numbers
        try:
            self._study.tell(trial_num, score)
            logger.info(f"Reported score {score:.4f} to Optuna trial #{trial_num}")
        except Exception as e:
            logger.warning(f"Failed to report to Optuna: {e}")

    def _trial_to_params(self, trial) -> dict:
        """Convert an Optuna trial to a HyperParams dict using configured search space."""
        ss = self.search_space

        # Helper: suggest from a (low, high) tuple
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

    # ------------------------------------------------------------------
    # Random Walk (Yellow escape)
    # ------------------------------------------------------------------

    def _propose_random_walk(self, current_params: HyperParams) -> HyperParams:
        """Add Gaussian noise to current params with decaying step size."""
        self._rw_step_scale *= 0.95  # decay step size

        ss = self.search_space
        current = current_params.model_dump()

        perturbed = {}
        for key, value in current.items():
            if key == "batch":
                # Batch is categorical — randomly pick from configured choices
                perturbed[key] = random.choice(ss.batch)
                continue

            # Get the bounds for this parameter
            bounds = getattr(ss, key, None)
            if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
                perturbed[key] = value
                continue

            low, high = bounds
            # Add Gaussian noise scaled by step size
            noise_std = (high - low) * self._rw_step_scale
            new_val = value + random.gauss(0, noise_std)
            new_val = max(low, min(high, new_val))
            perturbed[key] = new_val

        logger.info(
            f"Random walk proposal (step_scale={self._rw_step_scale:.4f}): "
            f"lr0={perturbed['lr0']:.5f}, mosaic={perturbed['mosaic']:.3f}"
        )
        return HyperParams(**perturbed)

    # ------------------------------------------------------------------
    # Simulated Annealing (Yellow escape)
    # ------------------------------------------------------------------

    def _propose_simulated_annealing(
        self,
        current_params: HyperParams,
        current_score: float | None,
    ) -> HyperParams:
        """Simulated annealing: propose a neighbor and accept if better OR with probability exp(-delta/T)."""
        # Initialize SA state on first call
        if self._sa_best_params is None:
            self._sa_best_params = current_params
            self._sa_best_score = current_score or 0.0
            self._sa_temperature = 1.0

        # Generate a neighbor (random perturbation)
        neighbor = self._random_neighbor(current_params)

        # Decide acceptance
        score = current_score or 0.0
        delta = score - self._sa_best_score

        accepted = False
        if delta > 0:
            # Better — always accept
            self._sa_best_params = current_params
            self._sa_best_score = score
            accepted = True
        else:
            prob = math.exp(delta / max(self._sa_temperature, 1e-8))
            if random.random() < prob:
                accepted = True
                logger.info(f"SA accepted worse proposal with probability {prob:.3f}")

        # Cool down
        self._sa_temperature *= self._sa_decay
        self._sa_temperature = max(self._sa_temperature, 0.01)

        logger.info(
            f"Simulated annealing: T={self._sa_temperature:.4f}, "
            f"best_score={self._sa_best_score:.4f}, accepted={accepted}"
        )

        return neighbor if accepted else current_params

    def _random_neighbor(self, params: HyperParams) -> HyperParams:
        """Generate a random parameter neighbor for SA."""
        ss = self.search_space
        current = params.model_dump()

        neighbor = {}
        for key, value in current.items():
            if key == "batch":
                neighbor[key] = random.choice(ss.batch)
                continue

            bounds = getattr(ss, key, None)
            if bounds is None or not isinstance(bounds, tuple) or len(bounds) != 2:
                neighbor[key] = value
                continue

            low, high = bounds
            # Uniform perturbation within 20% of the range
            range_size = (high - low) * 0.2
            new_val = value + random.uniform(-range_size, range_size)
            neighbor[key] = max(low, min(high, new_val))

        return HyperParams(**neighbor)