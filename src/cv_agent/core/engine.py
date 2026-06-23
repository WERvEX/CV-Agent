"""Central orchestrator — wires all subsystems together.

The TrainingEngine implements the main loop:
    INIT → VALIDATE_DATA → TRAIN → EVALUATE → DECIDE → (loop)

With DATA_SUPPLEMENT as off-ramp when validation fails, and DONE
when max rounds are reached or an unrecoverable error occurs.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from cv_agent.core.config import HyperParams, TrainConfig
from cv_agent.core.state_machine import (
    DecisionAction,
    DecisionColor,
    RedCountTracker,
    TrainingLoopState,
)
from cv_agent.data.gap_report import DataGapReport
from cv_agent.data.supplement import DataSupplementer
from cv_agent.data.validator import DatasetValidator, ValidationIssue
from cv_agent.decision.llm_advisor import LLMAdvisor
from cv_agent.decision.optuna_optimizer import OptunaOptimizer
from cv_agent.decision.three_state import Decision, ThreeStateDecisionEngine
from cv_agent.interaction.ask_mode import AskModeHandler
from cv_agent.interaction.auto_mode import AutoModeHandler
from cv_agent.tracking.mlflow_manager import MLflowManager
from cv_agent.tracking.run_dir import (
    create_run_dir,
    load_metrics,
    save_artifacts,
    save_data_gap_report,
    snapshot_best_checkpoint,
)
from cv_agent.trainer.evaluator import Evaluator, RoundResult
from cv_agent.trainer.yolo_trainer import YOLOTrainer
from cv_agent.ui.console import (
    log_error,
    log_info,
    log_success,
    log_warning,
    print_final_summary,
    print_section,
)
from cv_agent.ui.live_panel import LivePanel
from cv_agent.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TrainingEngine
# ---------------------------------------------------------------------------


class TrainingEngine:
    """Central orchestrator for automated closed-loop YOLO training."""

    def __init__(self) -> None:
        self._state = TrainingLoopState.INIT
        self._config: TrainConfig | None = None
        self._run_dir: Path | None = None

        # Subsystems (initialized lazily in _setup_subsystems)
        self._validator: DatasetValidator | None = None
        self._supplementer: DataSupplementer | None = None
        self._yolo_trainer: YOLOTrainer | None = None
        self._evaluator: Evaluator | None = None
        self._decision_engine: ThreeStateDecisionEngine | None = None
        self._optuna: OptunaOptimizer | None = None
        self._llm_advisor: LLMAdvisor | None = None
        self._interaction: AutoModeHandler | AskModeHandler | None = None
        self._mlflow: MLflowManager | None = None

        # State tracking
        self._red_tracker = RedCountTracker()
        self._round_num: int = 0
        self._history: list[RoundResult] = []
        self._best_checkpoint: Path | None = None
        self._best_score: float = 0.0
        self._best_round: int = 0
        self._current_params: HyperParams | None = None
        self._decision_log: list[dict[str, Any]] = []
        self._llm_context_accumulator: str = ""

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, config: TrainConfig) -> None:
        """Execute the full closed-loop training session."""
        self._config = config

        # Configure logging with file output
        ts_dir = create_run_dir(config.output_root)
        log_file = ts_dir / "cv_agent.log"
        setup_logging(log_file=log_file)
        self._run_dir = ts_dir

        log_info("Starting cv_agent training session...")
        log_info(f"Run directory: {self._run_dir}")
        log_info(
            f"Model={config.model_variant}, epochs/round={config.epochs_per_round}, "
            f"max_rounds={config.max_rounds}, interaction={config.interaction_mode}"
        )

        self._setup_subsystems(config)

        # Start MLflow session
        self._mlflow.start_session(self._run_dir.name)

        self._live_panel = LivePanel(self)
        # NOTE: the Rich Live panel is currently disabled because it deadlocks
        # with Ultralytics' stdout/tqdm output on Windows (model.train() stalls
        # on its first print while the Live refresh thread holds the console).
        # Re-enable once the two are reconciled (e.g. alternate screen buffer,
        # or pausing Live around model.train()). For now, fall back to plain
        # console output: Ultralytics prints its own epoch progress, and cv_agent
        # prints section headers / decision tables between rounds.
        try:
            self._main_loop()
        except KeyboardInterrupt:
            log_warning("Training interrupted by user.")
        except Exception as e:
            log_error(f"Fatal error: {e}")
            logger.exception("Fatal error traceback:")
        finally:
            self._mlflow.end_session()
            self._print_summary()

    def resume(self, run_dir: Path) -> None:
        """Resume training from a prior experiment directory."""
        if not run_dir.exists():
            log_error(f"Run directory not found: {run_dir}")
            return

        # Load config from args.yaml
        args_yaml = run_dir / "args.yaml"
        if not args_yaml.exists():
            log_error(f"args.yaml not found in {run_dir} — cannot resume.")
            return

        log_info(f"Resuming from {run_dir}...")
        log_warning("Resume mode: full implementation pending (Phase 10).")
        log_info("Loading config and decision log from run directory...")

        # Placeholder — Phase 10 will fully implement resume
        self._run_dir = run_dir
        metrics = load_metrics(run_dir)
        if metrics:
            log_info(f"Loaded metrics: best_score={metrics.get('best_score', 'N/A')}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """Execute the training state machine loop."""
        config = self._config
        self._state = TrainingLoopState.INIT
        self._current_params = config.initial_hyperparams
        self._round_num = 0

        # Drive the state machine until it reaches DONE. Basing the loop guard
        # on the state (not round_num) ensures the final round still runs through
        # EVALUATE and DECIDE — otherwise max_rounds==N skips them for round N
        # because _do_train has already incremented _round_num to N.
        while self._state is not TrainingLoopState.DONE:
            match self._state:
                case TrainingLoopState.INIT:
                    self._set_stage("INIT")
                    self._state = TrainingLoopState.VALIDATE_DATA

                case TrainingLoopState.VALIDATE_DATA:
                    self._set_stage("VALIDATE")
                    self._do_validate_data()

                case TrainingLoopState.DATA_SUPPLEMENT:
                    self._set_stage("DATA_SUPPLEMENT")
                    self._do_data_supplement()

                case TrainingLoopState.TRAIN:
                    self._set_stage("TRAIN")
                    # Guard: stop starting new rounds once we've hit max_rounds.
                    if self._round_num >= config.max_rounds:
                        self._state = TrainingLoopState.DONE
                        continue
                    self._do_train()

                case TrainingLoopState.EVALUATE:
                    self._set_stage("EVALUATE")
                    self._do_evaluate()

                case TrainingLoopState.DECIDE:
                    self._set_stage("DECIDE")
                    self._do_decide()

                case TrainingLoopState.DONE:
                    break

        log_info(f"Training loop finished after {self._round_num} round(s).")

    def _set_stage(self, stage: str) -> None:
        """Update the live panel's stage indicator (no-op if panel inactive)."""
        panel = getattr(self, "_live_panel", None)
        if panel is not None:
            panel.set_stage(stage)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _do_validate_data(self) -> None:
        """VALIDATE_DATA: Check dataset, transition to TRAIN or DATA_SUPPLEMENT."""
        print_section("Dataset Validation")
        log_info(f"Validating dataset: {self._config.data.data_yaml}")

        issues = self._validator.validate()

        if not issues:
            log_success("Dataset validation passed — all checks OK.")
            self._state = TrainingLoopState.TRAIN
            return

        errors = [i for i in issues if i.severity == "error"]
        if errors:
            log_warning(f"Found {len(errors)} error(s) — entering data supplement mode.")
            self._state = TrainingLoopState.DATA_SUPPLEMENT
            # Store issues for supplement handler
            self._pending_issues = issues
        else:
            log_warning(f"Found {len(issues)} warning(s) only — proceeding with training.")
            self._state = TrainingLoopState.TRAIN

    def _do_data_supplement(self) -> None:
        """DATA_SUPPLEMENT: Generate download scripts, wait for user, retry validation."""
        print_section("Data Supplement Mode")
        issues = getattr(self, "_pending_issues", [])
        should_retry = self._supplementer.handle(issues, self._run_dir)
        if should_retry:
            # User (ask mode) indicated data was fixed — re-validate.
            self._state = TrainingLoopState.VALIDATE_DATA
        else:
            # Auto mode, or user chose to abort: stop the session cleanly.
            log_error("Aborting training session due to unresolved dataset validation errors.")
            self._state = TrainingLoopState.DONE

    def _do_train(self) -> None:
        """TRAIN: Run a single YOLO training round."""
        self._round_num += 1
        print_section(f"Training Round {self._round_num}/{self._config.max_rounds}")

        log_info(f"Hyperparameters: lr0={self._current_params.lr0:.5f}, "
                 f"batch={self._current_params.batch}, mosaic={self._current_params.mosaic:.3f}")

        # MLflow nested run for this round
        self._mlflow.start_round(self._round_num)
        self._mlflow.log_params(self._current_params.model_dump())

        try:
            artifacts = self._yolo_trainer.train(
                model_variant=self._config.model_variant,
                data_yaml=self._config.data.data_yaml,
                hyperparams=self._current_params,
                epochs=self._config.epochs_per_round,
                run_dir=self._run_dir,
            )
            self._last_artifacts = artifacts
            self._state = TrainingLoopState.EVALUATE
        except Exception as e:
            log_error(f"Training failed in round {self._round_num}: {e}")
            logger.exception("Training error traceback:")
            # Auto-rollback to best checkpoint if available
            if self._best_checkpoint and self._best_checkpoint.exists():
                log_warning("Rolling back to best checkpoint...")
                shutil.copy2(self._best_checkpoint, self._run_dir / "weights" / "best.pt")
            self._decision_log.append({
                "round": self._round_num,
                "color": "red",
                "action": "training_crash",
                "reason": str(e),
            })
            self._mlflow.end_round()
            # Don't count this as a real round, just retry
            self._state = TrainingLoopState.TRAIN

    def _do_evaluate(self) -> None:
        """EVALUATE: Extract metrics, compute reward, compare with history."""
        print_section(f"Evaluation Round {self._round_num}")

        artifacts = getattr(self, "_last_artifacts", None)
        if artifacts is None:
            log_error("No training artifacts found — cannot evaluate.")
            self._state = TrainingLoopState.TRAIN
            return

        config = self._config
        round_result = self._evaluator.extract_metrics(
            results_csv=artifacts.results_csv,
            data_yaml=config.data.data_yaml,
            round_num=self._round_num,
            run_dir=self._run_dir,
        )

        comparison = self._evaluator.compare(round_result, self._history)

        # Log metrics to MLflow
        self._mlflow.log_metrics(round_result.metrics, step=self._round_num)
        self._mlflow.log_metrics({
            "score": round_result.score,
            "train_loss": round_result.train_loss_final or 0,
            "val_loss": round_result.val_loss_final or 0,
        }, step=self._round_num)

        # Save artifacts
        save_artifacts(
            run_dir=self._run_dir,
            metrics=round_result.metrics | {
                "score": round_result.score,
                "best_score": max(self._best_score, round_result.score),
                "overfitting": round_result.overfitting,
                "underfitting": round_result.underfitting,
            },
        )

        self._last_round_result = round_result
        self._last_comparison = comparison
        self._history.append(round_result)
        self._state = TrainingLoopState.DECIDE

    def _do_decide(self) -> None:
        """DECIDE: Classify round, determine action, mutate params, check for done."""
        print_section(f"Decision Round {self._round_num}")

        round_result = getattr(self, "_last_round_result", None)
        comparison = getattr(self, "_last_comparison", None)

        if round_result is None or comparison is None:
            log_error("No evaluation result to decide on.")
            self._state = TrainingLoopState.TRAIN
            return

        # Classify
        decision = self._decision_engine.decide(
            comparison=comparison,
            red_count=self._red_tracker.count,
            current_params=self._current_params,
        )

        # --- Propagate decision via interaction handler ---
        user_feedback = ""
        if isinstance(self._interaction, AskModeHandler):
            user_feedback = self._interaction.propagate_decision(
                decision.to_dict(), self._round_num
            )
        else:
            self._interaction.propagate_decision(decision.to_dict(), self._round_num)

        if user_feedback:
            decision.llm_context = user_feedback
            self._llm_context_accumulator += f"\n[Round {self._round_num} user feedback]: {user_feedback}"

        # --- Three-state routing ---
        color = decision.color

        if color == DecisionColor.GREEN.value:
            self._handle_green(decision, round_result)
        elif color == DecisionColor.YELLOW.value:
            self._handle_yellow(decision)
        else:  # red
            self._handle_red(decision)

        # --- Record decision ---
        self._decision_log.append(decision.to_dict())
        self._mlflow.log_decision(decision.to_dict(), self._round_num)

        # --- Mutate hyperparameters ---
        # Ask-mode: confirm before applying changes
        approved = True
        if isinstance(self._interaction, AskModeHandler):
            approved, feedback = self._interaction.confirm_config_change(
                old_params=self._current_params.model_dump(),
                new_params=decision.next_hyperparams.model_dump(),
                context=f"Round {self._round_num} ({color.upper()})",
            )
            if feedback:
                self._llm_context_accumulator += f"\n[Config feedback]: {feedback}"

        if approved:
            # Use Optuna for the next proposal unless a specific action overrides
            if decision.action in (DecisionAction.ACCEPT.value, DecisionAction.ESCAPE_LOCAL_OPTIMUM.value):
                self._current_params = self._optuna.propose(
                    current_params=self._current_params,
                    current_score=round_result.score,
                    state=color,
                )
            else:
                self._current_params = decision.next_hyperparams
        else:
            # User rejected — keep current params with small perturbation
            self._current_params = self._decision_engine._perturb_params(self._current_params)

        # Save decision log
        save_artifacts(run_dir=self._run_dir, decision_log=self._decision_log)

        self._mlflow.end_round()

        # Check if we're done
        if self._round_num >= self._config.max_rounds:
            self._state = TrainingLoopState.DONE
        else:
            self._state = TrainingLoopState.TRAIN

    # ------------------------------------------------------------------
    # Decision handlers
    # ------------------------------------------------------------------

    def _handle_green(self, decision: Decision, round_result: RoundResult) -> None:
        """Green: commit checkpoint, update best, reset red counter."""
        log_success(f"GREEN — {decision.reason}")

        # Commit checkpoint as new best
        best_pt = self._run_dir / "weights" / "best.pt"
        if best_pt.exists():
            self._best_checkpoint = snapshot_best_checkpoint(self._run_dir, self._round_num)
            self._best_score = round_result.score
            self._best_round = self._round_num
            log_info(f"New best checkpoint: round {self._round_num}, score={round_result.score:.4f}")

        self._red_tracker.reset()

    def _handle_yellow(self, decision: Decision) -> None:
        """Yellow: oscillation — trigger local optimum escape."""
        log_warning(f"YELLOW — {decision.reason}")
        # Optuna will handle the escape strategy in propose()
        self._red_tracker.reset()

    def _handle_red(self, decision: Decision) -> None:
        """Red: degradation — diagnose and act. If 3 consecutive Reds, escalate."""
        log_error(f"RED — {decision.reason}")
        is_escalated = self._red_tracker.increment()

        log_warning(f"Red count: {self._red_tracker.count}/{self._red_tracker.max_consecutive}")

        if decision.should_rollback:
            self._do_rollback(decision.rollback_checkpoint)

        if is_escalated:
            log_error(
                f"RED x{self._red_tracker.max_consecutive} — ESCALATION! "
                "Force rollback + LLM data gap analysis."
            )
            self._do_escalation(decision)

    def _do_rollback(self, checkpoint_path: str | None = None) -> None:
        """Roll back to the best known checkpoint."""
        best_pt = self._run_dir / "weights" / "best.pt"

        def _safe_copy(src: Path) -> bool:
            """Copy src to best_pt, skipping (no-op) if they're the same file."""
            try:
                if src.resolve() == best_pt.resolve():
                    log_info(f"Best checkpoint already in place: {best_pt}")
                    return True
                shutil.copy2(src, best_pt)
                return True
            except OSError as e:
                log_warning(f"Rollback copy failed ({src} → {best_pt}): {e}")
                return False

        if self._best_checkpoint and self._best_checkpoint.exists():
            if _safe_copy(self._best_checkpoint):
                log_info(f"Rolled back to best checkpoint (round {self._best_round}).")
        elif checkpoint_path and Path(checkpoint_path).exists():
            if _safe_copy(Path(checkpoint_path)):
                log_info(f"Rolled back to specified checkpoint: {checkpoint_path}")
        else:
            log_warning("No checkpoint available for rollback.")

    def _do_escalation(self, decision: Decision) -> None:
        """3 consecutive Reds: force rollback, generate LLM data gap report, trigger supplement."""
        # Force rollback
        if self._best_checkpoint and self._best_checkpoint.exists():
            best_pt = self._run_dir / "weights" / "best.pt"
            if self._best_checkpoint.resolve() != best_pt.resolve():
                shutil.copy2(self._best_checkpoint, best_pt)
            log_info("Hard rollback to best checkpoint completed.")

        # Generate Data Gap Report via LLM Advisor
        class_names = {}
        try:
            class_names = Evaluator.load_class_names(self._config.data.data_yaml)
        except Exception:
            pass

        round_result = getattr(self, "_last_round_result", None)
        current_metrics = round_result.metrics if round_result else {}
        best_round = max(self._history, key=lambda r: r.score) if self._history else None
        best_metrics = best_round.metrics if best_round else {}

        # Heuristic CM analysis first (fast), then LLM if available
        cm_analysis = self._llm_advisor.analyze_confusion_matrix(
            confusion=round_result.confusion_matrix if round_result and round_result.confusion_matrix is not None
            else [],
            class_names=class_names,
            metrics=current_metrics,
            global_metrics=current_metrics,
        )

        gap_report = self._llm_advisor.generate_data_gap_report(
            cm_analysis=cm_analysis,
            current_metrics=current_metrics,
            best_metrics=best_metrics,
            class_names=class_names,
            trigger_round=self._round_num,
            confusion_snapshot=None,
        )

        # Save the report
        save_data_gap_report(
            run_dir=self._run_dir,
            report_md=gap_report.to_markdown(),
            report_json=gap_report.model_dump(),
        )
        log_info(f"Data gap report saved to {self._run_dir}")

        # Reset red counter
        self._red_tracker.reset()

        # Transition to data supplement mode
        self._pending_issues = self._gap_report_to_issues(gap_report)
        self._state = TrainingLoopState.DATA_SUPPLEMENT

    # ------------------------------------------------------------------
    # Setup & helpers
    # ------------------------------------------------------------------

    def _setup_subsystems(self, config: TrainConfig) -> None:
        """Initialize all subsystem components."""
        setup_logging(log_file=self._run_dir / "cv_agent.log")

        # Data validation
        self._validator = DatasetValidator(config.data)

        # Interaction handler
        if config.interaction_mode == "auto":
            self._interaction = AutoModeHandler()
        else:
            self._interaction = AskModeHandler()

        # Data supplementer
        self._supplementer = DataSupplementer(interaction=self._interaction)

        # YOLO trainer
        self._yolo_trainer = YOLOTrainer()

        # Evaluator with optional class optimization
        optimize_id = None
        if config.optimize_for_class:
            optimize_id = Evaluator.resolve_class_id(
                config.optimize_for_class, config.data.data_yaml
            )
            if optimize_id is not None:
                log_info(f"Optimizing for class: {config.optimize_for_class} (id={optimize_id})")
            else:
                log_warning(
                    f"Class '{config.optimize_for_class}' not found in dataset. "
                    "Falling back to global mAP optimization."
                )
        self._evaluator = Evaluator(optimize_for_class_id=optimize_id)

        # Decision engine
        self._decision_engine = ThreeStateDecisionEngine()

        # Optuna optimizer
        self._optuna = OptunaOptimizer(
            config=config.optuna,
            study_db=config.output_root / "optuna_study.db",
        )

        # LLM advisor
        self._llm_advisor = LLMAdvisor(config=config.llm)

        # MLflow manager
        self._mlflow = MLflowManager(
            tracking_uri=config.mlflow_uri,
            experiment_name=config.experiment_name,
        )

    def _print_summary(self) -> None:
        """Print final training session summary."""
        print_final_summary(
            rounds_run=self._round_num,
            best_round=self._best_round,
            best_score=self._best_score,
            run_dir=self._run_dir or Path("runs"),
        )

    @staticmethod
    def _gap_report_to_issues(report: DataGapReport) -> list[ValidationIssue]:
        """Convert a DataGapReport into validation issues for the supplement handler."""
        issues: list[ValidationIssue] = []
        for pc in report.problem_classes:
            issues.append(ValidationIssue(
                severity="error",
                category="data_gap",
                detail=f"Class '{pc.class_name}' (id={pc.class_id}): {pc.issue}",
                suggestion=pc.suggested_remedy,
            ))
        return issues
