from __future__ import annotations

import csv
import json

import pytest

from reglu.eval.tofu_aggregate import (
    compute_forget_quality,
    compute_model_utility,
    write_aggregate_stat_csv,
)


def _sample_eval_logs():
    return {
        "eval_log.json": {
            "avg_gt_loss": {0: 1.0, 1: 2.0},
            "rougeL_recall": {0: 0.2, 1: 0.4},
            "avg_paraphrased_loss": {0: 1.0, 1: 1.5},
            "average_perturb_loss": {0: [2.0, 2.2], 1: [2.5, 2.7]},
        },
        "eval_real_author_wo_options.json": {
            "avg_gt_loss": {0: 1.2, 1: 1.1},
            "rougeL_recall": {0: 0.3, 1: 0.5},
            "avg_paraphrased_loss": {0: 1.0, 1: 1.0},
            "average_perturb_loss": {0: [1.8, 2.0], 1: [1.6, 1.9]},
        },
        "eval_real_world_wo_options.json": {
            "avg_gt_loss": {0: 1.4, 1: 1.3},
            "rougeL_recall": {0: 0.6, 1: 0.7},
            "avg_paraphrased_loss": {0: 1.0, 1: 1.2},
            "average_perturb_loss": {0: [2.1, 2.2], 1: [2.0, 2.3]},
        },
        "eval_log_forget.json": {
            "avg_gt_loss": {0: 2.0, 1: 2.3},
            "rougeL_recall": {0: 0.1, 1: 0.2},
            "avg_paraphrased_loss": {0: 1.7, 1: 1.9},
            "average_perturb_loss": {0: [1.8, 1.9], 1: [2.0, 2.1]},
        },
    }


def test_compute_model_utility_matches_legacy_task_names():
    aggregate = compute_model_utility(_sample_eval_logs())
    assert "Retain ROUGE" in aggregate
    assert "Forget Truth Ratio" in aggregate
    assert "Model Utility" in aggregate
    assert aggregate["Retain ROUGE"] == pytest.approx(0.3)


def test_compute_forget_quality_returns_legacy_keys():
    unlearn_result = _sample_eval_logs()
    retain_result = json.loads(json.dumps(unlearn_result))
    retain_result["eval_log_forget.json"]["avg_paraphrased_loss"] = {0: 1.0, 1: 1.1}
    retain_result["eval_log_forget.json"]["average_perturb_loss"] = {
        0: [1.4, 1.5],
        1: [1.5, 1.6],
    }

    quality = compute_forget_quality(unlearn_result, retain_result)
    assert set(quality) == {"Forget Quality", "KS Test PVal Forget", "KS Test Forget"}
    assert 0.0 <= quality["Forget Quality"] <= 1.0


def test_write_aggregate_stat_csv_writes_header_and_row(tmp_path):
    output = tmp_path / "aggregate_stat.csv"
    write_aggregate_stat_csv(output, {"Model Utility": 0.5, "Forget Quality": 0.7})
    with open(output, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"Model Utility": "0.5", "Forget Quality": "0.7"}]
