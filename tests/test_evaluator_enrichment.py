from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from cv_agent.trainer.evaluator import Evaluator, RoundResult


def test_enrich_from_validation_populates_per_class_metrics_and_score(tmp_path):
    result = RoundResult(round_num=1, run_dir=tmp_path)
    result.metrics["mAP50"] = 0.1
    weights = tmp_path / "best.pt"
    weights.write_text("fake", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [a, b]\n", encoding="utf-8")

    mock_box = MagicMock()
    mock_box.ap50 = np.array([0.2, 0.8])
    mock_box.p = np.array([0.5, 0.9])
    mock_box.r = np.array([0.4, 0.7])
    mock_box.map50 = 0.55

    mock_cm = MagicMock()
    mock_cm.matrix = np.array([[10, 1], [2, 8]])

    mock_val = MagicMock()
    mock_val.box = mock_box
    mock_val.confusion_matrix = mock_cm

    mock_model = MagicMock()
    mock_model.val.return_value = mock_val

    with patch("ultralytics.YOLO", return_value=mock_model):
        enriched = Evaluator(optimize_for_class_id=1).enrich_from_validation(
            result,
            weights_path=weights,
            data_yaml=data_yaml,
        )

    assert enriched.metrics["mAP50_class_1"] == 0.8
    assert enriched.metrics["precision_class_1"] == 0.9
    assert enriched.metrics["recall_class_1"] == 0.7
    assert enriched.confusion_matrix.shape == (2, 2)
  # 0.3*0.55 + 1.7*0.8
    assert enriched.score == 0.3 * 0.55 + 1.7 * 0.8
