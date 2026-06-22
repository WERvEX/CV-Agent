"""LLM Advisor — OpenAI-compatible API for strategic reasoning.

Two primary functions:
1. analyze_confusion_matrix(): Identifies confused class pairs, recommends loss weight adjustments
2. generate_data_gap_report(): Produces structured data sourcing recommendations

Includes a heuristic fallback mode (rule-based) when no API key is configured
or the API is unreachable. Defaults to DeepSeek API.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from openai import OpenAI

from cv_agent.core.config import LLMConfig
from cv_agent.data.gap_report import DataGapReport, DataSource, ProblemClass, build_data_gap_report
from cv_agent.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured output schemas (described in prompts, parsed via JSON)
# ---------------------------------------------------------------------------


class ConfusionAnalysis:
    """Result of LLM / heuristic confusion matrix analysis."""

    def __init__(
        self,
        confused_pairs: list[dict[str, Any]],
        per_class_assessment: dict[str, str],
        loss_weight_suggestions: dict[str, float],
    ) -> None:
        self.confused_pairs = confused_pairs
        self.per_class_assessment = per_class_assessment
        self.loss_weight_suggestions = loss_weight_suggestions


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CM_ANALYSIS_PROMPT = """You are a computer vision expert analyzing object detection model performance.

Given the following confusion matrix and per-class metrics, identify:

1. **Confused class pairs**: Which classes are the model confusing with which? Note the direction (A→B means A is misclassified as B, bidirectional means both).
2. **Per-class assessment**: For each class, briefly assess performance (good / acceptable / poor).
3. **Loss weight suggestions**: Recommend adjusted loss weights (box_loss, cls_loss, dfl_loss) as numerical values between 0.1 and 20.0. Increase cls weight for classes with low precision/recall.

Confusion Matrix (rows = ground truth, columns = predictions):
{confusion_matrix}

Class names (index → name):
{class_names}

Per-class metrics:
{per_class_metrics}

Global metrics:
{global_metrics}

Return a JSON object with this structure:
{{
    "confused_pairs": [
        {{
            "class_a": "class_name",
            "class_b": "class_name",
            "count": estimated_count,
            "direction": "bidirectional" or "A→B"
        }}
    ],
    "per_class_assessment": {{
        "class_name": "good / acceptable / poor — brief reason"
    }},
    "loss_weight_suggestions": {{
        "box": float,
        "cls": float,
        "dfl": float
    }}
}}
"""

GAP_REPORT_PROMPT = """You are a computer vision data strategist. A YOLO object detection model has
shown persistent performance degradation (3 consecutive red rounds). Analyze the data and recommend
specific data-sourcing strategies.

Confusion Matrix Analysis:
{cm_analysis}

Current metrics:
{metrics_summary}

Best historical metrics:
{best_metrics}

Problem classes identified:
{problem_classes_summary}

Task: Generate a data gap report. For each problem class, recommend specific datasets and sources.

Return a JSON object with this structure:
{{
    "problem_classes": [
        {{
            "class_name": "string",
            "class_id": int,
            "issue": "low recall" | "confused with X" | "small objects" | "few samples",
            "suggested_remedy": "specific suggestion"
        }}
    ],
    "recommended_sources": [
        {{
            "name": "Roboflow Universe" | "OpenImages" | "HuggingFace" | "Custom Collection",
            "url": "specific URL or null",
            "estimated_samples": int,
            "instructions": "how to obtain the data"
        }}
    ],
    "llm_analysis": "free-text summary of the situation and recommended strategy"
}}
"""

# ---------------------------------------------------------------------------
# LLM Advisor
# ---------------------------------------------------------------------------


class LLMAdvisor:
    """Strategic advisor powered by an external LLM (DeepSeek default).

    Falls back to heuristic analysis when no API key is available or the API fails.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM advisor.

        Args:
            config: LLMConfig with API settings.
        """
        self.config = config
        self._client: OpenAI | None = None
        self._call_count: int = 0
        self._has_api_key = bool(config.api_key)

        if self._has_api_key:
            self._client = OpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
            )
            logger.info(f"LLM Advisor initialized: model={config.model}, base={config.api_base}")
        else:
            logger.warning("No LLM API key configured — using heuristic fallback mode.")

    # ------------------------------------------------------------------
    # Confusion Matrix Analysis
    # ------------------------------------------------------------------

    def analyze_confusion_matrix(
        self,
        confusion: np.ndarray | list[list[int]],
        class_names: dict[int, str],
        metrics: dict[str, float],
        global_metrics: dict[str, float] | None = None,
    ) -> ConfusionAnalysis:
        """Analyze confusion matrix for confused class pairs and loss weight recommendations.

        Args:
            confusion: Confusion matrix (rows=GT, cols=predictions).
            class_names: Mapping from class_id to class_name.
            metrics: Per-class metrics dict.
            global_metrics: Global mAP/precision/recall.

        Returns:
            ConfusionAnalysis with confused pairs, assessments, and weight suggestions.
        """
        if self._client is not None and self._call_count < self.config.max_calls_per_session:
            result = self._llm_analyze_cm(confusion, class_names, metrics, global_metrics or {})
            if result is not None:
                return result

        # Fallback to heuristic
        logger.info("Using heuristic confusion matrix analysis (fallback mode).")
        return self._heuristic_analyze_cm(confusion, class_names, metrics)

    def _llm_analyze_cm(
        self,
        confusion: np.ndarray | list[list[int]],
        class_names: dict[int, str],
        metrics: dict[str, float],
        global_metrics: dict[str, float],
    ) -> ConfusionAnalysis | None:
        """Call LLM for confusion matrix analysis."""
        prompt = CM_ANALYSIS_PROMPT.format(
            confusion_matrix=self._format_matrix(confusion),
            class_names=json.dumps(class_names, indent=2),
            per_class_metrics=json.dumps(metrics, indent=2),
            global_metrics=json.dumps(global_metrics, indent=2),
        )

        response = self._call_llm(prompt)
        if response is None:
            return None

        try:
            data = json.loads(response)
            return ConfusionAnalysis(
                confused_pairs=data.get("confused_pairs", []),
                per_class_assessment=data.get("per_class_assessment", {}),
                loss_weight_suggestions=data.get("loss_weight_suggestions", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM CM analysis response: {e}")
            return None

    def _heuristic_analyze_cm(
        self,
        confusion: np.ndarray | list[list[int]],
        class_names: dict[int, str],
        metrics: dict[str, float],
    ) -> ConfusionAnalysis:
        """Rule-based heuristic confusion matrix analysis."""
        cm = np.array(confusion) if not isinstance(confusion, np.ndarray) else confusion
        n_classes = cm.shape[0]

        confused_pairs: list[dict[str, Any]] = []
        per_class_assessment: dict[str, str] = {}

        # For each class, find the most common off-diagonal confusion
        for i in range(n_classes):
            name_i = class_names.get(i, f"class_{i}")

            # Check per-class metrics for assessment
            cls_precision = metrics.get(f"precision_class_{i}", 0)
            cls_recall = metrics.get(f"recall_class_{i}", 0)

            if cls_recall < 0.3:
                per_class_assessment[name_i] = f"poor — very low recall ({cls_recall:.2f})"
            elif cls_recall < 0.6:
                per_class_assessment[name_i] = f"acceptable — moderate recall ({cls_recall:.2f})"
            else:
                per_class_assessment[name_i] = f"good — recall {cls_recall:.2f}"

            # Find off-diagonal confusion
            for j in range(n_classes):
                if i == j:
                    continue
                if cm[i, j] > 0:
                    # Fraction of GT class i misclassified as j
                    total_gt_i = cm[i, :].sum()
                    if total_gt_i > 0 and cm[i, j] / total_gt_i > 0.05:
                        name_j = class_names.get(j, f"class_{j}")
                        confused_pairs.append({
                            "class_a": name_i,
                            "class_b": name_j,
                            "count": int(cm[i, j]),
                            "direction": f"{name_i} → {name_j}",
                        })

        # Heuristic loss weight suggestions
        # If overall recall is low, boost cls weight
        avg_recall = np.mean([metrics.get(f"recall_class_{i}", 0) for i in range(n_classes)])
        cls_weight = 0.5 if avg_recall > 0.6 else (1.5 if avg_recall > 0.3 else 3.0)

        return ConfusionAnalysis(
            confused_pairs=confused_pairs[:10],  # top 10
            per_class_assessment=per_class_assessment,
            loss_weight_suggestions={"box": 7.5, "cls": cls_weight, "dfl": 1.5},
        )

    # ------------------------------------------------------------------
    # Data Gap Report
    # ------------------------------------------------------------------

    def generate_data_gap_report(
        self,
        cm_analysis: ConfusionAnalysis,
        current_metrics: dict[str, float],
        best_metrics: dict[str, float],
        class_names: dict[int, str],
        trigger_round: int,
        confusion_snapshot: list[list[int]] | None = None,
    ) -> DataGapReport:
        """Generate a structured data gap report.

        Args:
            cm_analysis: Result of confusion matrix analysis.
            current_metrics: Current round's metrics.
            best_metrics: Best historical metrics.
            class_names: Class ID to name mapping.
            trigger_round: Round number that triggered this report.
            confusion_snapshot: Confusion matrix snapshot.

        Returns:
            DataGapReport with problem classes and data source recommendations.
        """
        if self._client is not None and self._call_count < self.config.max_calls_per_session:
            result = self._llm_generate_gap_report(
                cm_analysis, current_metrics, best_metrics, class_names, trigger_round, confusion_snapshot
            )
            if result is not None:
                return result

        # Fallback to heuristic
        logger.info("Using heuristic data gap report generation (fallback mode).")
        return self._heuristic_generate_gap_report(
            cm_analysis, current_metrics, best_metrics, class_names, trigger_round, confusion_snapshot
        )

    def _llm_generate_gap_report(
        self,
        cm_analysis: ConfusionAnalysis,
        current_metrics: dict[str, float],
        best_metrics: dict[str, float],
        class_names: dict[int, str],
        trigger_round: int,
        confusion_snapshot: list[list[int]] | None,
    ) -> DataGapReport | None:
        """Call LLM for data gap report generation."""
        prompt = GAP_REPORT_PROMPT.format(
            cm_analysis=json.dumps({
                "confused_pairs": cm_analysis.confused_pairs,
                "per_class_assessment": cm_analysis.per_class_assessment,
            }, indent=2),
            metrics_summary=json.dumps(current_metrics, indent=2),
            best_metrics=json.dumps(best_metrics, indent=2),
            problem_classes_summary=json.dumps(
                [k for k, v in cm_analysis.per_class_assessment.items() if "poor" in v],
                indent=2,
            ),
        )

        response = self._call_llm(prompt)
        if response is None:
            return None

        try:
            data = json.loads(response)
            return build_data_gap_report(
                trigger_round=trigger_round,
                problem_classes=data.get("problem_classes", []),
                sources=data.get("recommended_sources", []),
                llm_raw=data.get("llm_analysis", ""),
                cm_snapshot=confusion_snapshot,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM gap report response: {e}")
            return None

    def _heuristic_generate_gap_report(
        self,
        cm_analysis: ConfusionAnalysis,
        current_metrics: dict[str, float],
        best_metrics: dict[str, float],
        class_names: dict[int, str],
        trigger_round: int,
        confusion_snapshot: list[list[int]] | None,
    ) -> DataGapReport:
        """Heuristic data gap report generation without LLM."""
        problem_classes: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []

        for class_name, assessment in cm_analysis.per_class_assessment.items():
            if "poor" in assessment:
                # Find class_id from class_names
                class_id = None
                for cid, cname in class_names.items():
                    if cname == class_name:
                        class_id = cid
                        break

                problem_classes.append({
                    "class_name": class_name,
                    "class_id": class_id or -1,
                    "issue": assessment.replace("poor — ", "").strip('"'),
                    "suggested_remedy": (
                        f"Add 200-500 more annotated samples of '{class_name}' with varied "
                        "backgrounds, lighting, and angles. Consider sourcing from Roboflow Universe."
                    ),
                })

        # Generic source recommendations
        sources = [
            {
                "name": "Roboflow Universe",
                "url": "https://universe.roboflow.com/",
                "estimated_samples": 500,
                "instructions": f"Search for datasets containing: {', '.join(p['class_name'] for p in problem_classes)}. Download in YOLOv8 format.",
            },
            {
                "name": "OpenImages v7",
                "url": "https://storage.googleapis.com/openimages/web/index.html",
                "estimated_samples": 1000,
                "instructions": "Use FiftyOne to download OpenImages data for the target classes.",
            },
            {
                "name": "Custom Collection",
                "url": None,
                "estimated_samples": 200,
                "instructions": "Manually collect and annotate images for problem classes using CVAT or LabelImg.",
            },
        ]

        llm_raw = (
            "## Heuristic Analysis (No LLM Available)\n\n"
            f"The following classes show poor performance and likely need additional training data:\n\n"
            + "\n".join(
                f"- **{p['class_name']}**: {p['issue']}"
                for p in problem_classes
            )
            + "\n\n"
            "Recommended: acquire 200-500 additional labeled samples per problem class "
            "from Roboflow Universe or OpenImages, ensuring diversity in backgrounds, "
            "lighting conditions, and object scales."
        )

        return build_data_gap_report(
            trigger_round=trigger_round,
            problem_classes=problem_classes,
            sources=sources,
            llm_raw=llm_raw,
            cm_snapshot=confusion_snapshot,
        )

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str | None:
        """Call the LLM API with retry logic.

        Args:
            prompt: The formatted prompt.

        Returns:
            Response content string, or None on failure.
        """
        if self._client is None:
            return None

        if self._call_count >= self.config.max_calls_per_session:
            logger.warning(f"LLM call limit reached ({self.config.max_calls_per_session}). Using fallback.")
            return None

        for attempt in range(3):
            try:
                self._call_count += 1
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a computer vision expert. Always respond with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                logger.info(f"LLM call #{self._call_count} succeeded ({len(content or '')} chars).")
                return content

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/3): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        logger.error("LLM API call failed after 3 retries. Using heuristic fallback.")
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _format_matrix(matrix: np.ndarray | list[list[int]]) -> str:
        """Format a confusion matrix for prompt inclusion."""
        m = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
        # Truncate large matrices for prompt
        if m.shape[0] > 20:
            return f"[{m.shape[0]}x{m.shape[1]} matrix — too large to display]"
        return np.array2string(m, max_line_width=120, threshold=100)