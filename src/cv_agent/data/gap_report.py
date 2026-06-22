"""Data Gap Report models and Markdown rendering.

Used when Red×3 escalation triggers LLM analysis of model weaknesses.
Produces structured JSON + human-readable Markdown reports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProblemClass(BaseModel):
    """A class identified as having data quality issues."""

    class_name: str
    class_id: int
    issue: str           # "low recall", "confused with X", "small objects"
    suggested_remedy: str


class DataSource(BaseModel):
    """A recommended data source for supplementing training data."""

    name: str            # "Roboflow Universe", "OpenImages", "HuggingFace"
    url: str | None = None
    estimated_samples: int = 0
    instructions: str = ""


class DataGapReport(BaseModel):
    """Complete data gap analysis report."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    trigger_round: int
    problem_classes: list[ProblemClass] = Field(default_factory=list)
    recommended_sources: list[DataSource] = Field(default_factory=list)
    llm_analysis: str = ""          # raw LLM text output
    confusion_matrix_snapshot: list[list[int]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the data gap report as Markdown."""
        lines = [
            f"# Data Gap Report",
            f"",
            f"**Generated:** {self.timestamp}",
            f"**Triggered at round:** {self.trigger_round}",
            f"",
            "---",
            "",
            "## Problem Classes",
            "",
        ]

        if not self.problem_classes:
            lines.append("_No specific problem classes identified._")
        else:
            lines.append("| Class | ID | Issue | Suggested Remedy |")
            lines.append("|-------|----|-------|-----------------|")
            for pc in self.problem_classes:
                lines.append(f"| {pc.class_name} | {pc.class_id} | {pc.issue} | {pc.suggested_remedy} |")

        lines.extend([
            "",
            "---",
            "",
            "## Recommended Data Sources",
            "",
        ])

        if not self.recommended_sources:
            lines.append("_No specific sources recommended._")
        else:
            for i, src in enumerate(self.recommended_sources, 1):
                lines.append(f"### {i}. {src.name}")
                lines.append(f"- **Estimated samples:** {src.estimated_samples}")
                if src.url:
                    lines.append(f"- **URL:** {src.url}")
                lines.append(f"- **Instructions:** {src.instructions}")
                lines.append("")

        if self.llm_analysis:
            lines.extend([
                "---",
                "",
                "## AI Analysis (LLM)",
                "",
                self.llm_analysis,
                "",
            ])

        return "\n".join(lines)


def build_data_gap_report(
    trigger_round: int,
    problem_classes: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    llm_raw: str = "",
    cm_snapshot: list[list[int]] | None = None,
) -> DataGapReport:
    """Construct a DataGapReport from component dicts.

    Args:
        trigger_round: Round number that triggered the report.
        problem_classes: List of problem class dicts.
        sources: List of data source dicts.
        llm_raw: Raw LLM analysis text (if available).
        cm_snapshot: Confusion matrix snapshot.

    Returns:
        DataGapReport instance.
    """
    return DataGapReport(
        trigger_round=trigger_round,
        problem_classes=[ProblemClass(**pc) for pc in problem_classes],
        recommended_sources=[DataSource(**src) for src in sources],
        llm_analysis=llm_raw,
        confusion_matrix_snapshot=cm_snapshot or [],
    )