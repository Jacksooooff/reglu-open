from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TOFU_TASK_FILES = (
    "eval_log.json",
    "eval_real_author_wo_options.json",
    "eval_real_world_wo_options.json",
    "eval_log_forget.json",
)
TOFU_REQUIRED_FILES = TOFU_TASK_FILES + (
    "eval_log_aggregated.json",
    "aggregate_stat.csv",
)
WMDP_REQUIRED_FILES = (
    "LMEval_SUMMARY.json",
    "LMEval_EVAL.json",
)


@dataclass
class NumericTolerance:
    atol: float = 5e-4
    rtol: float = 1e-3


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_csv_first_row(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    return {}


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numeric(value, child_prefix))
        return result
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            child_prefix = f"{prefix}[{idx}]"
            result.update(_flatten_numeric(value, child_prefix))
        return result
    numeric = _to_float(payload)
    if numeric is not None and prefix:
        result[prefix] = numeric
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_nested(values: list[Any]) -> float:
    flat: list[float] = []
    for value in values:
        if isinstance(value, list):
            nested = [_to_float(item) for item in value]
            flat.append(_mean([item for item in nested if item is not None]))
        else:
            numeric = _to_float(value)
            if numeric is not None:
                flat.append(numeric)
    return _mean(flat)


def _summarize_tofu_task(eval_logs: dict[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if "avg_gt_loss" in eval_logs:
        summary["avg_gt_loss_mean"] = _mean(
            [value for value in (_to_float(v) for v in eval_logs["avg_gt_loss"].values()) if value is not None]
        )
    if "rouge1_recall" in eval_logs:
        summary["rouge1_recall_mean"] = _mean(
            [value for value in (_to_float(v) for v in eval_logs["rouge1_recall"].values()) if value is not None]
        )
    if "rougeL_recall" in eval_logs:
        summary["rougeL_recall_mean"] = _mean(
            [value for value in (_to_float(v) for v in eval_logs["rougeL_recall"].values()) if value is not None]
        )
    if "truth_ratio" in eval_logs:
        summary["truth_ratio_mean"] = _mean(
            [value for value in (_to_float(v) for v in eval_logs["truth_ratio"].values()) if value is not None]
        )
    if "average_perturb_loss" in eval_logs:
        summary["avg_perturb_loss_mean"] = _mean_nested(list(eval_logs["average_perturb_loss"].values()))
    if "normalized_gt_loss" in eval_logs:
        summary["normalized_gt_loss_mean"] = _mean(
            [value for value in (_to_float(v) for v in eval_logs["normalized_gt_loss"].values()) if value is not None]
        )
    summary["num_records"] = float(
        len(eval_logs.get("avg_gt_loss", {}))
        or len(eval_logs.get("generated_text", {}))
        or 0
    )
    return summary


def _compare_numeric_maps(
    baseline: dict[str, float],
    candidate: dict[str, float],
    tolerance: NumericTolerance,
) -> dict[str, Any]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    missing_in_candidate = sorted(baseline_keys - candidate_keys)
    extra_in_candidate = sorted(candidate_keys - baseline_keys)
    diffs = []
    compared = 0
    for key in sorted(baseline_keys & candidate_keys):
        b = baseline[key]
        c = candidate[key]
        compared += 1
        abs_diff = abs(c - b)
        limit = tolerance.atol + tolerance.rtol * abs(b)
        if not math.isfinite(abs_diff):
            diffs.append(
                {
                    "key": key,
                    "baseline": b,
                    "candidate": c,
                    "abs_diff": abs_diff,
                    "allowed": limit,
                }
            )
            continue
        if abs_diff > limit:
            diffs.append(
                {
                    "key": key,
                    "baseline": b,
                    "candidate": c,
                    "abs_diff": abs_diff,
                    "allowed": limit,
                }
            )
    return {
        "compared_keys": compared,
        "missing_in_candidate": missing_in_candidate,
        "extra_in_candidate": extra_in_candidate,
        "exceeded": diffs,
        "passed": not missing_in_candidate and not extra_in_candidate and not diffs,
    }


def _check_required_files(root: Path, files: list[str]) -> list[str]:
    missing = []
    for relpath in files:
        if not (root / relpath).exists():
            missing.append(relpath)
    return missing


def _compare_json_files(
    baseline_file: Path,
    candidate_file: Path,
    tolerance: NumericTolerance,
) -> dict[str, Any]:
    baseline = _flatten_numeric(_load_json(baseline_file))
    candidate = _flatten_numeric(_load_json(candidate_file))
    return _compare_numeric_maps(baseline, candidate, tolerance)


def _compare_csv_trace(
    baseline_file: Path,
    candidate_file: Path,
    tolerance: NumericTolerance,
    step_key: str,
    value_key: str,
) -> dict[str, Any]:
    def _read_trace(path: Path) -> dict[str, float]:
        trace: dict[str, float] = {}
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                step = row.get(step_key)
                value = _to_float(row.get(value_key))
                if step is None or value is None:
                    continue
                trace[str(step)] = value
        return trace

    return _compare_numeric_maps(
        _read_trace(baseline_file),
        _read_trace(candidate_file),
        tolerance,
    )


def _run_tofu_eval_check(check: dict[str, Any], tolerance: NumericTolerance) -> dict[str, Any]:
    baseline_dir = Path(check["baseline_dir"])
    candidate_dir = Path(check["candidate_dir"])
    baseline_missing = _check_required_files(baseline_dir, list(TOFU_REQUIRED_FILES))
    candidate_missing = _check_required_files(candidate_dir, list(TOFU_REQUIRED_FILES))
    result: dict[str, Any] = {
        "kind": "tofu_eval",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_missing": baseline_missing,
        "candidate_missing": candidate_missing,
        "tasks": {},
    }
    if baseline_missing or candidate_missing:
        result["passed"] = False
        return result

    aggregate_compare = _compare_numeric_maps(
        {
            key: value
            for key, value in (
                (k, _to_float(v)) for k, v in _load_csv_first_row(baseline_dir / "aggregate_stat.csv").items()
            )
            if value is not None
        },
        {
            key: value
            for key, value in (
                (k, _to_float(v)) for k, v in _load_csv_first_row(candidate_dir / "aggregate_stat.csv").items()
            )
            if value is not None
        },
        tolerance,
    )
    result["aggregate_stat"] = aggregate_compare

    task_pass = True
    for task_file in TOFU_TASK_FILES:
        baseline_summary = _summarize_tofu_task(_load_json(baseline_dir / task_file))
        candidate_summary = _summarize_tofu_task(_load_json(candidate_dir / task_file))
        comparison = _compare_numeric_maps(baseline_summary, candidate_summary, tolerance)
        result["tasks"][task_file] = comparison
        task_pass = task_pass and comparison["passed"]

    result["passed"] = aggregate_compare["passed"] and task_pass
    return result


def _run_wmdp_eval_check(check: dict[str, Any], tolerance: NumericTolerance) -> dict[str, Any]:
    baseline_dir = Path(check["baseline_dir"])
    candidate_dir = Path(check["candidate_dir"])
    baseline_missing = _check_required_files(baseline_dir, list(WMDP_REQUIRED_FILES))
    candidate_missing = _check_required_files(candidate_dir, list(WMDP_REQUIRED_FILES))
    result: dict[str, Any] = {
        "kind": "wmdp_eval",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_missing": baseline_missing,
        "candidate_missing": candidate_missing,
    }
    if baseline_missing or candidate_missing:
        result["passed"] = False
        return result

    summary_compare = _compare_json_files(
        baseline_dir / "LMEval_SUMMARY.json",
        candidate_dir / "LMEval_SUMMARY.json",
        tolerance,
    )
    baseline_eval = _load_json(baseline_dir / "LMEval_EVAL.json")
    candidate_eval = _load_json(candidate_dir / "LMEval_EVAL.json")
    baseline_tasks = sorted(baseline_eval.keys()) if isinstance(baseline_eval, dict) else []
    candidate_tasks = sorted(candidate_eval.keys()) if isinstance(candidate_eval, dict) else []
    task_set_passed = baseline_tasks == candidate_tasks
    result["summary"] = summary_compare
    result["task_set"] = {
        "baseline_tasks": baseline_tasks,
        "candidate_tasks": candidate_tasks,
        "passed": task_set_passed,
    }
    result["passed"] = summary_compare["passed"] and task_set_passed
    return result


def _run_json_check(check: dict[str, Any], tolerance: NumericTolerance) -> dict[str, Any]:
    baseline = Path(check["baseline"])
    candidate = Path(check["candidate"])
    missing = []
    if not baseline.exists():
        missing.append(f"baseline:{baseline}")
    if not candidate.exists():
        missing.append(f"candidate:{candidate}")
    result: dict[str, Any] = {
        "kind": "json",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "missing": missing,
    }
    if missing:
        result["passed"] = False
        return result
    result["compare"] = _compare_json_files(baseline, candidate, tolerance)
    result["passed"] = result["compare"]["passed"]
    return result


def _run_csv_trace_check(check: dict[str, Any], tolerance: NumericTolerance) -> dict[str, Any]:
    baseline = Path(check["baseline"])
    candidate = Path(check["candidate"])
    missing = []
    if not baseline.exists():
        missing.append(f"baseline:{baseline}")
    if not candidate.exists():
        missing.append(f"candidate:{candidate}")
    step_key = str(check.get("step_key", "step"))
    value_key = str(check.get("value_key", "loss"))
    result: dict[str, Any] = {
        "kind": "csv_trace",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "step_key": step_key,
        "value_key": value_key,
        "missing": missing,
    }
    if missing:
        result["passed"] = False
        return result
    result["compare"] = _compare_csv_trace(baseline, candidate, tolerance, step_key, value_key)
    result["passed"] = result["compare"]["passed"]
    return result


def _run_run_dir_check(check: dict[str, Any], tolerance: NumericTolerance) -> dict[str, Any]:
    baseline_dir = Path(check["baseline_dir"])
    candidate_dir = Path(check["candidate_dir"])
    required_common = [str(item) for item in check.get("required_files", [])]
    required_baseline = required_common + [str(item) for item in check.get("required_files_baseline", [])]
    required_candidate = required_common + [str(item) for item in check.get("required_files_candidate", [])]
    baseline_missing = _check_required_files(baseline_dir, required_baseline)
    candidate_missing = _check_required_files(candidate_dir, required_candidate)
    result: dict[str, Any] = {
        "kind": "run_dir",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline_missing": baseline_missing,
        "candidate_missing": candidate_missing,
    }
    passed = not baseline_missing and not candidate_missing

    summary_relpath = check.get("summary_relpath")
    if summary_relpath:
        baseline_summary = baseline_dir / str(summary_relpath)
        candidate_summary = candidate_dir / str(summary_relpath)
        summary_missing = []
        if not baseline_summary.exists():
            summary_missing.append(f"baseline:{baseline_summary}")
        if not candidate_summary.exists():
            summary_missing.append(f"candidate:{candidate_summary}")
        result["summary_missing"] = summary_missing
        if not summary_missing:
            result["summary_compare"] = _compare_json_files(
                baseline_summary,
                candidate_summary,
                tolerance,
            )
            passed = passed and result["summary_compare"]["passed"]
        else:
            passed = False

    trace_relpath = check.get("trace_relpath")
    if trace_relpath:
        baseline_trace = baseline_dir / str(trace_relpath)
        candidate_trace = candidate_dir / str(trace_relpath)
        trace_missing = []
        if not baseline_trace.exists():
            trace_missing.append(f"baseline:{baseline_trace}")
        if not candidate_trace.exists():
            trace_missing.append(f"candidate:{candidate_trace}")
        result["trace_missing"] = trace_missing
        if not trace_missing:
            result["trace_compare"] = _compare_csv_trace(
                baseline_trace,
                candidate_trace,
                tolerance,
                step_key=str(check.get("step_key", "step")),
                value_key=str(check.get("value_key", "loss")),
            )
            passed = passed and result["trace_compare"]["passed"]
        else:
            passed = False

    result["passed"] = passed
    return result


def run_check(check: dict[str, Any], defaults: NumericTolerance) -> dict[str, Any]:
    tolerance = NumericTolerance(
        atol=float(check.get("atol", defaults.atol)),
        rtol=float(check.get("rtol", defaults.rtol)),
    )
    kind = str(check["kind"])
    if kind == "tofu_eval":
        result = _run_tofu_eval_check(check, tolerance)
    elif kind == "wmdp_eval":
        result = _run_wmdp_eval_check(check, tolerance)
    elif kind == "json":
        result = _run_json_check(check, tolerance)
    elif kind == "csv_trace":
        result = _run_csv_trace_check(check, tolerance)
    elif kind == "run_dir":
        result = _run_run_dir_check(check, tolerance)
    else:
        raise ValueError(f"Unsupported validation check kind: {kind}")
    result["name"] = str(check["name"])
    result["atol"] = tolerance.atol
    result["rtol"] = tolerance.rtol
    return result


def run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    defaults_payload = manifest.get("defaults", {}) or {}
    defaults = NumericTolerance(
        atol=float(defaults_payload.get("atol", 5e-4)),
        rtol=float(defaults_payload.get("rtol", 1e-3)),
    )
    checks_payload = manifest.get("checks", [])
    if not isinstance(checks_payload, list) or not checks_payload:
        raise ValueError("Manifest must contain a non-empty 'checks' list.")
    results = [run_check(check, defaults) for check in checks_payload]
    overall_pass = all(result.get("passed", False) for result in results)
    return {
        "version": manifest.get("version", 1),
        "overall_pass": overall_pass,
        "checks": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate legacy GPU runs against reglu-open outputs.")
    parser.add_argument("--manifest", required=True, help="Path to a YAML manifest describing the comparisons.")
    parser.add_argument("--report", default=None, help="Optional path to write the JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = _load_yaml(args.manifest)
    report = run_manifest(manifest)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
