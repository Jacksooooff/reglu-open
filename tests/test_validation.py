from __future__ import annotations

import json

import yaml

from reglu.validation import run_manifest


def test_validation_manifest_passes_for_matching_tofu_eval(tmp_path):
    baseline = tmp_path / "baseline-tofu"
    candidate = tmp_path / "candidate-tofu"
    baseline.mkdir()
    candidate.mkdir()

    task_payload = {
        "avg_gt_loss": {"0": 1.0, "1": 2.0},
        "rouge1_recall": {"0": 0.5, "1": 0.7},
        "rougeL_recall": {"0": 0.4, "1": 0.6},
        "truth_ratio": {"0": 0.3, "1": 0.2},
        "avg_paraphrased_loss": {"0": 1.0, "1": 1.5},
        "average_perturb_loss": {"0": [2.0, 2.1], "1": [1.8, 1.9]},
    }
    for root in (baseline, candidate):
        for filename in (
            "eval_log.json",
            "eval_real_author_wo_options.json",
            "eval_real_world_wo_options.json",
            "eval_log_forget.json",
            "eval_log_aggregated.json",
        ):
            (root / filename).write_text(json.dumps(task_payload), encoding="utf-8")
        (root / "aggregate_stat.csv").write_text(
            "Model Utility,Forget Quality\n0.5,0.9\n",
            encoding="utf-8",
        )

    manifest = {
        "version": 1,
        "checks": [
            {
                "name": "tofu_eval",
                "kind": "tofu_eval",
                "baseline_dir": str(baseline),
                "candidate_dir": str(candidate),
            }
        ],
    }
    report = run_manifest(manifest)
    assert report["overall_pass"] is True
    assert report["checks"][0]["passed"] is True


def test_validation_manifest_passes_for_matching_wmdp_eval(tmp_path):
    baseline = tmp_path / "baseline-wmdp"
    candidate = tmp_path / "candidate-wmdp"
    baseline.mkdir()
    candidate.mkdir()

    summary = {"wmdp_bio/acc": 0.25, "mmlu/acc": 0.50}
    eval_payload = {"wmdp_bio": {"samples": []}, "mmlu": {"samples": []}}
    for root in (baseline, candidate):
        (root / "LMEval_SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
        (root / "LMEval_EVAL.json").write_text(json.dumps(eval_payload), encoding="utf-8")

    manifest = {
        "version": 1,
        "checks": [
            {
                "name": "wmdp_eval",
                "kind": "wmdp_eval",
                "baseline_dir": str(baseline),
                "candidate_dir": str(candidate),
            }
        ],
    }
    report = run_manifest(manifest)
    assert report["overall_pass"] is True
    assert report["checks"][0]["passed"] is True


def test_validation_manifest_fails_on_numeric_mismatch(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "summary.json").write_text(json.dumps({"loss": 1.0}), encoding="utf-8")
    (candidate / "summary.json").write_text(json.dumps({"loss": 2.0}), encoding="utf-8")

    manifest = {
        "version": 1,
        "checks": [
            {
                "name": "summary",
                "kind": "json",
                "baseline": str(baseline / "summary.json"),
                "candidate": str(candidate / "summary.json"),
                "atol": 1e-6,
                "rtol": 1e-6,
            }
        ],
    }
    report = run_manifest(manifest)
    assert report["overall_pass"] is False
    assert report["checks"][0]["passed"] is False
