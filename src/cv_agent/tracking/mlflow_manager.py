"""MLflow integration for cv_agent experiment tracking.

Manages the MLflow run lifecycle:
- Parent run = full cv_agent session
- Nested runs = individual training rounds

Handles graceful degradation when MLflow server is unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Manager for MLflow experiment tracking across cv_agent sessions."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "cv_agent",
    ) -> None:
        """Initialize MLflow tracking.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active: bool = False
        self._parent_run_id: str | None = None
        self._active_run_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self, run_name: str) -> bool:
        """Start the parent MLflow run for this cv_agent session.

        Args:
            run_name: Display name for the run (e.g., exp_20260122_143052).

        Returns:
            True if MLflow is active, False if server is unreachable.
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        try:
            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=run_name, nested=False)
            self._parent_run_id = run.info.run_id
            self._active_run_id = run.info.run_id
            self._active = True
            logger.info(f"MLflow session started: {run_name} (run_id={self._parent_run_id})")
            return True
        except Exception as e:
            logger.warning(f"MLflow server unreachable at {self.tracking_uri}: {e}")
            logger.warning("Continuing without MLflow — metrics will be saved locally only.")
            self._active = False
            return False

    def start_round(self, round_num: int) -> None:
        """Start a nested MLflow run for a training round.

        Args:
            round_num: Current round number (1-indexed).
        """
        if not self._active:
            return
        run = mlflow.start_run(run_name=f"round_{round_num:03d}", nested=True)
        self._active_run_id = run.info.run_id
        logger.info(f"MLflow round {round_num} started (run_id={self._active_run_id})")

    def end_round(self) -> None:
        """End the current training round's MLflow nested run."""
        if not self._active:
            return
        mlflow.end_run()
        logger.info("MLflow round ended.")

    def end_session(self) -> None:
        """End the parent MLflow run."""
        if not self._active:
            return
        try:
            mlflow.end_run()
            logger.info("MLflow session ended.")
        except Exception as e:
            logger.warning(f"Failed to cleanly end MLflow session: {e}")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to the current MLflow run.

        Args:
            params: Key-value pairs of parameters.
        """
        if not self._active:
            return
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics to the current MLflow run.

        Args:
            metrics: Key-value pairs of metrics (must be numeric).
            step: Optional step/epoch number.
        """
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, local_dir: Path) -> None:
        """Log all files in a directory as MLflow artifacts.

        Args:
            local_dir: Directory containing artifacts to log.
        """
        if not self._active:
            return
        if not local_dir.exists():
            return
        mlflow.log_artifacts(str(local_dir))

    def log_artifact(self, local_path: Path) -> None:
        """Log a single file as an MLflow artifact.

        Args:
            local_path: Path to the file.
        """
        if not self._active:
            return
        if local_path.exists():
            mlflow.log_artifact(str(local_path))

    def log_decision(self, decision: dict[str, Any], round_num: int) -> None:
        """Log a decision record as MLflow params.

        Args:
            decision: Decision dictionary with color, action, reason, etc.
            round_num: Current round number.
        """
        if not self._active:
            return
        mlflow.log_param(f"round_{round_num}_color", decision.get("color", "unknown"))
        mlflow.log_param(f"round_{round_num}_action", decision.get("action", "unknown"))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether MLflow tracking is active."""
        return self._active

    @property
    def parent_run_id(self) -> str | None:
        """The parent session run ID."""
        return self._parent_run_id

    @property
    def active_run_id(self) -> str | None:
        """The currently active (nested) run ID."""
        return self._active_run_id