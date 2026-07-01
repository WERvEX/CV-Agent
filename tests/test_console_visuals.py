from __future__ import annotations

from io import StringIO

from rich.console import Console

from cv_agent.ui.console import (
    print_decision_timeline,
    print_guidance_applied,
    print_round_evaluation,
    print_strategy_patch,
    summarize_per_class_map50,
)
from cv_agent.decision.strategy import ObjectiveWeights, StrategyPatch, StrategyPhase


def test_print_round_evaluation_renders_without_error():
    out = StringIO()
    console = Console(file=out, width=120, force_terminal=True)

    from cv_agent.ui import console as console_mod
    original = console_mod.console
    console_mod.console = console
    try:
        print_round_evaluation(
            round_num=2,
            score=0.65,
            metrics={"mAP50": 0.62, "mAP50_95": 0.41, "mAP50_class_0": 0.7},
            delta_percent=-1.2,
            best_score=0.66,
            best_round=1,
            overfitting=False,
            underfitting=False,
        )
        text = out.getvalue()
    finally:
        console_mod.console = original

    assert "Round 2 Evaluation" in text
    assert "0.6500" in text
    assert "-1.20%" in text


def test_per_class_summary_compact():
    metrics = {f"mAP50_class_{i}": 0.1 * i for i in range(10)}
    summary = summarize_per_class_map50(
        metrics,
        class_names={0: "person", 1: "bike", 9: "truck"},
    )
    assert summary is not None
    assert "10 classes" in summary
    assert "low:" in summary
    assert "high:" in summary
    assert "person" in summary


def test_print_guidance_applied_shows_diff():
    out = StringIO()
    console = Console(file=out, width=120, force_terminal=True)

    from cv_agent.ui import console as console_mod
    original = console_mod.console
    console_mod.console = console
    try:
        print_guidance_applied(
            before_params={"lr0": 0.01, "batch": 16},
            after_params={"lr0": 0.005, "batch": 16},
            source="llm",
            reason="Halve learning rate",
            raw_text="lr 减半",
            pause=False,
        )
        text = out.getvalue()
    finally:
        console_mod.console = original

    assert "Guidance Applied" in text
    assert "LLM" in text
    assert "Halve learning rate" in text


def test_print_decision_timeline_renders_rows():
    out = StringIO()
    console = Console(file=out, width=120, force_terminal=True)

    from cv_agent.ui import console as console_mod
    original = console_mod.console
    console_mod.console = console
    try:
        print_decision_timeline(
            [
                {"color": "green", "action": "accept", "reason": "improved"},
                {"color": "yellow", "action": "escape", "reason": "flat"},
            ],
            round_scores=[(1, 0.66), (2, 0.65)],
        )
        text = out.getvalue()
    finally:
        console_mod.console = original

    assert "Decision Timeline" in text
    assert "GREEN" in text
    assert "YELLOW" in text


def test_print_strategy_patch_renders_compact_summary():
    out = StringIO()
    console = Console(file=out, width=120, force_terminal=True)

    from cv_agent.ui import console as console_mod
    original = console_mod.console
    console_mod.console = console
    try:
        print_strategy_patch(
            StrategyPatch(
                phase=StrategyPhase.EXPLOITATION,
                reason="narrow lr after stable improvement",
                search_space_patch={"lr0": (0.001, 0.003)},
                freeze={"batch"},
                objective_weights=ObjectiveWeights(recall=0.4, precision=0.1),
                max_trials_for_phase=2,
                confidence=0.72,
                metadata={"raw_prompt": "verbose internal details"},
            )
        )
        text = out.getvalue()
    finally:
        console_mod.console = original

    assert "LLM Strategy" in text
    assert "exploitation" in text
    assert "lr0" in text
    assert "batch" in text
    assert "0.72" in text
    assert "raw_prompt" not in text
