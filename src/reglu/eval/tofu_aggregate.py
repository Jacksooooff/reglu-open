from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import hmean, ks_2samp


LEGACY_TASK_NAMES = {
    "eval_real_author_wo_options.json": "Real Authors",
    "eval_real_world_wo_options.json": "Real World",
    "eval_log.json": "Retain",
    "eval_log_forget.json": "Forget",
}


def _ordered_values(mapping: dict) -> list:
    return [mapping[key] for key in mapping.keys()]


def _mean_nested(values: list) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    if array.ndim > 1:
        return array.mean(axis=-1)
    return array


def compute_model_utility(eval_result_dict: dict[str, dict]) -> dict[str, float]:
    output_result: dict[str, float] = {}
    model_utility_terms: list[float] = []

    for task_file, task_label in LEGACY_TASK_NAMES.items():
        if task_file not in eval_result_dict:
            continue
        task_eval = eval_result_dict[task_file]

        if "eval_log" in task_file:
            gt_probs = np.exp(-1.0 * np.array(_ordered_values(task_eval["avg_gt_loss"]), dtype=np.float64))
            avg_gt_prob = float(np.mean(gt_probs))
        else:
            avg_true_prob = np.exp(
                -1.0 * np.array(_ordered_values(task_eval["avg_gt_loss"]), dtype=np.float64)
            )
            avg_false_prob = np.exp(
                -1.0 * np.array(_ordered_values(task_eval["average_perturb_loss"]), dtype=np.float64)
            )
            avg_all_prob = np.concatenate(
                [np.expand_dims(avg_true_prob, axis=-1), avg_false_prob],
                axis=1,
            ).sum(-1)
            avg_gt_prob = float(np.mean(avg_true_prob / avg_all_prob))
        output_result[f"{task_label} Probability"] = avg_gt_prob

        avg_rouge = float(
            np.array(_ordered_values(task_eval["rougeL_recall"]), dtype=np.float64).mean()
        )
        output_result[f"{task_label} ROUGE"] = avg_rouge

        avg_paraphrase_probs = np.exp(
            -1.0 * np.array(_ordered_values(task_eval["avg_paraphrased_loss"]), dtype=np.float64)
        )
        avg_perturbed_probs = np.exp(
            -1.0 * np.array(_ordered_values(task_eval["average_perturb_loss"]), dtype=np.float64)
        ).mean(axis=-1)
        paraphrased_perturb_ratio = avg_perturbed_probs / avg_paraphrase_probs

        if "forget" in task_file:
            truth_ratio = float(
                np.mean(np.minimum(paraphrased_perturb_ratio, 1.0 / paraphrased_perturb_ratio))
            )
        else:
            truth_ratio = float(np.mean(np.maximum(0.0, 1.0 - paraphrased_perturb_ratio)))
        output_result[f"{task_label} Truth Ratio"] = truth_ratio

        if task_label != "Forget":
            model_utility_terms.extend(
                [
                    output_result[f"{task_label} Probability"],
                    output_result[f"{task_label} ROUGE"],
                    output_result[f"{task_label} Truth Ratio"],
                ]
            )

    if model_utility_terms:
        output_result["Model Utility"] = float(hmean(model_utility_terms))
    return output_result


def compute_forget_quality(
    unlearn_result: dict[str, dict],
    retain_result: dict[str, dict],
) -> dict[str, float]:
    unlearn_forget_result = unlearn_result["eval_log_forget.json"]
    retain_forget_result = retain_result["eval_log_forget.json"]

    unlearn_paraphrase = np.array(
        _ordered_values(unlearn_forget_result["avg_paraphrased_loss"]),
        dtype=np.float64,
    )
    unlearn_perturbed = _mean_nested(
        _ordered_values(unlearn_forget_result["average_perturb_loss"])
    )

    retain_paraphrase = np.array(
        _ordered_values(retain_forget_result["avg_paraphrased_loss"]),
        dtype=np.float64,
    )
    retain_perturbed = _mean_nested(
        _ordered_values(retain_forget_result["average_perturb_loss"])
    )

    unlearn_truth_ratio = np.exp(unlearn_perturbed - unlearn_paraphrase)
    retain_truth_ratio = np.exp(retain_perturbed - retain_paraphrase)
    ks_result = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {
        "Forget Quality": float(ks_result.pvalue),
        "KS Test PVal Forget": float(ks_result.pvalue),
        "KS Test Forget": float(ks_result.statistic),
    }


def write_aggregate_stat_csv(path: str | Path, aggregate_stat: dict[str, float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_stat.keys()))
        writer.writeheader()
        writer.writerow(aggregate_stat)
