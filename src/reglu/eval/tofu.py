from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from reglu.artifacts import append_metrics, ensure_run_layout, write_config_snapshot, write_summary
from reglu.config import RunConfig
from reglu.data import custom_data_collator_with_indices, get_batch_loss
from reglu.eval.tofu_aggregate import (
    compute_forget_quality,
    compute_model_utility,
    write_aggregate_stat_csv,
)
from reglu.data.tofu import TofuTextDataset
from reglu.models import get_model_spec
from reglu.trainers.common import is_peft_adapter_dir, load_base_model, load_model, load_tokenizer


def _mean(values) -> float:
    series = list(values)
    return float(np.mean(series)) if series else 0.0


def _mean_metric_dict(metric: dict[Any, Any] | None) -> float:
    if not metric:
        return 0.0
    values = []
    for value in metric.values():
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            values.append(float(np.mean(value)))
        else:
            values.append(float(value))
    return _mean(values)


def _normalize_split_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\n", "\n")


def _resolve_split_candidates(config: RunConfig, model_spec) -> list[str]:
    candidates = []

    def _add(symbol: str | None) -> None:
        if symbol and symbol not in candidates:
            candidates.append(symbol)

    manual_symbol = _normalize_split_symbol(config.evaluation.split_symbol)
    if manual_symbol:
        _add(manual_symbol)
        if not manual_symbol.endswith("\n"):
            _add(manual_symbol + "\n")

    _add(model_spec.answer_tag)
    _add(model_spec.question_end_tag)
    _add("Answer: ")
    return candidates


def _split_prompt_and_answer(text: str, candidates: list[str]) -> tuple[str, str, str | None]:
    for symbol in candidates:
        parts = text.split(symbol, 1)
        if len(parts) > 1:
            return parts[0], parts[1], symbol
    return text, "", None


def _run_generation(config: RunConfig, batch: dict[str, torch.Tensor], model, tokenizer):
    decoded_strings = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    model_spec = get_model_spec(config.model_family)
    split_candidates = _resolve_split_candidates(config, model_spec)
    split_results = [_split_prompt_and_answer(text, split_candidates) for text in decoded_strings]
    missing = [idx for idx, (_, _, symbol) in enumerate(split_results) if symbol is None]
    if missing:
        example = decoded_strings[missing[0]].replace("\n", "\\n")
        raise ValueError(
            f"Failed to locate split symbol among {split_candidates} for sample "
            f"starting with '{example[:200]}'."
        )

    prompts = []
    ground_truths = []
    for prompt, answer, symbol in split_results:
        if model_spec.question_end_tag and symbol == model_spec.question_end_tag:
            prompt = prompt + model_spec.question_end_tag
        prompts.append(prompt)
        ground_truths.append(answer)

    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token
    original_pad_token_id = tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(
            prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        generate_kwargs = {
            "attention_mask": inputs.attention_mask,
            "max_length": config.evaluation.max_length,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if config.evaluation.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = config.evaluation.max_new_tokens
        outputs = model.generate(inputs.input_ids, **generate_kwargs)
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token = original_pad_token
        tokenizer.pad_token_id = original_pad_token_id

    generated = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    return prompts, generated, ground_truths


def _compute_rouge_recall(
    generations: list[str],
    ground_truths: list[str],
    indices: list[int],
) -> dict[str, dict[int, float]]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = {}
    rouge_l_recall = {}
    for generation, ground_truth, index in zip(generations, ground_truths, indices):
        score = scorer.score(ground_truth, generation)
        rouge1_recall[int(index)] = float(score["rouge1"].recall)
        rouge_l_recall[int(index)] = float(score["rougeL"].recall)
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}


def _compute_perturbation_logs(eval_loader, perturb_loader, model) -> dict[str, dict[int, Any]]:
    eval_logs: dict[str, dict[int, Any]] = {}
    for batch, perturb_batch in zip(eval_loader, perturb_loader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        indices = batch["indices"]
        perturb_input_ids = perturb_batch["input_ids"]
        perturb_labels = perturb_batch["labels"]
        perturb_attention_mask = perturb_batch["attention_mask"]
        if perturb_input_ids.ndim > 2:
            batch_size, perturb_count = perturb_input_ids.shape[0:2]
        else:
            batch_size = perturb_input_ids.shape[0]
            perturb_count = 1

        batch_payload = {
            "input_ids": input_ids.to(model.device),
            "labels": labels.to(model.device),
            "attention_mask": attention_mask.to(model.device),
        }
        perturb_payload = {
            "input_ids": perturb_input_ids.view(batch_size * perturb_count, -1).to(model.device),
            "labels": perturb_labels.view(batch_size * perturb_count, -1).to(model.device),
            "attention_mask": perturb_attention_mask.view(batch_size * perturb_count, -1).to(
                model.device
            ),
        }

        with torch.no_grad():
            outputs = model(**batch_payload)
            perturb_outputs = model(**perturb_payload)

        gt_loss = get_batch_loss(outputs.logits, batch_payload["labels"])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_payload["labels"]).view(
            batch_size, perturb_count
        )

        num_token_gt = (batch_payload["labels"] != -100).sum(-1)
        num_token_perturb = (perturb_payload["labels"] != -100).view(
            batch_size, perturb_count, -1
        ).sum(-1)

        perturb_loss_per_token = perturb_loss / num_token_perturb
        gt_loss_per_token = gt_loss / num_token_gt
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))

        index_values = indices.detach().cpu().tolist()
        logs_to_merge = {
            "average_perturb_loss": dict(
                zip(
                    index_values,
                    perturb_loss_per_token.detach().to(torch.float32).cpu().numpy().tolist(),
                )
            ),
            "avg_paraphrased_loss": dict(
                zip(index_values, gt_loss_per_token.detach().to(torch.float32).cpu().numpy().tolist())
            ),
            "truth_ratio": dict(
                zip(index_values, truth_ratio.detach().to(torch.float32).cpu().numpy().tolist())
            ),
            "paraphrased_loss": dict(
                zip(index_values, gt_loss.detach().to(torch.float32).cpu().numpy().tolist())
            ),
            "perturb_loss": dict(
                zip(index_values, perturb_loss.detach().to(torch.float32).cpu().numpy().tolist())
            ),
            "num_token_paraphrased": dict(
                zip(index_values, num_token_gt.detach().cpu().numpy().tolist())
            ),
            "num_token_perturb": dict(
                zip(index_values, num_token_perturb.detach().cpu().numpy().tolist())
            ),
        }
        for name, values in logs_to_merge.items():
            eval_logs.setdefault(name, {}).update(values)
    return eval_logs


def _build_loader(
    config: RunConfig,
    tokenizer,
    split: str,
    eval_task: str,
    question_key: str,
    answer_key: str,
    batch_size: int,
):
    subset_indices_file = None
    if eval_task == "eval_log_forget":
        subset_indices_file = config.data.unlearn_eval_subset_indices_file
    dataset = TofuTextDataset(
        data_path=config.data.path,
        tokenizer=tokenizer,
        model_family=config.model_family,
        max_length=config.evaluation.max_length,
        split=split,
        question_key=question_key,
        answer_key=answer_key,
        subset_indices_file=subset_indices_file,
    )
    if config.evaluation.ds_size:
        dataset.data = dataset.data.select(range(min(config.evaluation.ds_size, len(dataset.data))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_data_collator_with_indices,
    )


def _get_all_evals(
    config: RunConfig,
    model,
    tokenizer,
    eval_loader,
    base_eval_loader,
    perturb_loader,
    normalize_gt: bool = False,
) -> dict[str, Any]:
    eval_logs: dict[str, Any] = {}
    generations = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in eval_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        indices = batch["indices"]
        all_indices.extend(indices.detach().cpu().tolist())
        batch_payload = {
            "input_ids": input_ids.to(model.device),
            "labels": labels.to(model.device),
            "attention_mask": attention_mask.to(model.device),
        }

        with torch.no_grad():
            outputs = model(**batch_payload)
            batch_input_strings, batch_generations, batch_truths = _run_generation(
                config,
                {"input_ids": input_ids},
                model,
                tokenizer,
            )

        generations.extend(batch_generations)
        ground_truths.extend(batch_truths)
        input_strings.extend(batch_input_strings)

        gt_loss = get_batch_loss(outputs.logits, batch_payload["labels"])
        num_token_gt = (batch_payload["labels"] != -100).sum(-1)
        gt_loss_per_token = gt_loss / num_token_gt
        index_values = indices.detach().cpu().tolist()

        eval_logs.setdefault("avg_gt_loss", {}).update(
            dict(zip(index_values, gt_loss_per_token.detach().to(torch.float32).cpu().numpy().tolist()))
        )
        eval_logs.setdefault("gt_loss", {}).update(
            dict(zip(index_values, gt_loss.detach().to(torch.float32).cpu().numpy().tolist()))
        )
        eval_logs.setdefault("num_token_gt", {}).update(
            dict(zip(index_values, num_token_gt.detach().cpu().numpy().tolist()))
        )
        if config.evaluation.save_generated_text:
            eval_logs.setdefault("generated_text", {}).update(
                dict(zip(index_values, zip(batch_input_strings, batch_generations, batch_truths)))
            )

    eval_logs.update(_compute_rouge_recall(generations, ground_truths, all_indices))
    eval_logs.update(_compute_perturbation_logs(base_eval_loader, perturb_loader, model))

    if normalize_gt:
        normalized_gt_loss = {}
        avg_gt_loss = eval_logs["avg_gt_loss"]
        avg_perturb_loss = eval_logs["average_perturb_loss"]
        for index in avg_gt_loss.keys():
            truth_prob = np.exp(-1.0 * avg_gt_loss[index])
            perturb_prob = np.exp(-1.0 * np.array(avg_perturb_loss[index]))
            all_prob = np.array([truth_prob, *np.atleast_1d(perturb_prob)])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[index] = float(-1.0 * np.log(normalized_gt_prob))
        eval_logs["normalized_gt_loss"] = normalized_gt_loss

    return eval_logs


def _build_task_summary(eval_logs: dict[str, Any]) -> dict[str, float]:
    summary = {
        "avg_gt_loss_mean": _mean_metric_dict(eval_logs.get("avg_gt_loss")),
        "rouge1_recall_mean": _mean_metric_dict(eval_logs.get("rouge1_recall")),
        "rougeL_recall_mean": _mean_metric_dict(eval_logs.get("rougeL_recall")),
        "truth_ratio_mean": _mean_metric_dict(eval_logs.get("truth_ratio")),
        "avg_perturb_loss_mean": _mean_metric_dict(eval_logs.get("average_perturb_loss")),
    }
    if "normalized_gt_loss" in eval_logs:
        summary["normalized_gt_loss_mean"] = _mean_metric_dict(eval_logs["normalized_gt_loss"])
    return summary


def _task_specs(config: RunConfig) -> list[dict[str, Any]]:
    unlearn_split = f"{config.data.split}_perturbed"
    return [
        {
            "name": "retain",
            "eval_task": "eval_log",
            "split": "retain_perturbed",
            "question_key": config.data.question_key,
            "answer_key": config.data.answer_key,
            "base_answer_key": config.data.base_answer_key,
            "perturbed_answer_key": config.data.perturbed_answer_key,
            "normalize_gt": False,
        },
        {
            "name": "real_authors",
            "eval_task": "eval_real_author_wo_options",
            "split": "real_authors_perturbed",
            "question_key": config.data.question_key,
            "answer_key": config.data.answer_key,
            "base_answer_key": config.data.answer_key,
            "perturbed_answer_key": config.data.perturbed_answer_key,
            "normalize_gt": True,
        },
        {
            "name": "world_facts",
            "eval_task": "eval_real_world_wo_options",
            "split": "world_facts_perturbed",
            "question_key": config.data.question_key,
            "answer_key": config.data.answer_key,
            "base_answer_key": config.data.answer_key,
            "perturbed_answer_key": config.data.perturbed_answer_key,
            "normalize_gt": True,
        },
        {
            "name": "unlearn",
            "eval_task": "eval_log_forget",
            "split": unlearn_split,
            "question_key": config.data.question_key,
            "answer_key": config.data.answer_key,
            "base_answer_key": config.data.base_answer_key,
            "perturbed_answer_key": config.data.perturbed_answer_key,
            "normalize_gt": False,
        },
    ]


def run_tofu_eval(config: RunConfig) -> dict[str, Any]:
    layout = ensure_run_layout(config.runtime.output_dir)
    write_config_snapshot(layout["config"], config)
    summary: dict[str, Any] = {
        "command": "evaluate",
        "task": config.task,
        "model_family": config.model_family,
        "status": "dry_run" if config.runtime.dry_run else "ok",
    }
    if config.runtime.dry_run:
        write_summary(layout["summary"], summary)
        return summary

    overwrite = bool(config.evaluation.overwrite)
    from reglu.methods.reglu import build_lora_model, maybe_apply_rila_cache

    tokenizer = load_tokenizer(config)
    if config.evaluation.model_mode == "rila":
        from peft import PeftModel

        base_model = load_base_model(config)
        if is_peft_adapter_dir(config.model.model_path):
            model = PeftModel.from_pretrained(base_model, config.model.model_path)
        else:
            model = build_lora_model(base_model, config)
        if config.method.rila_cache_path:
            maybe_apply_rila_cache(model, config.method.rila_cache_path, mode="w_only")
    else:
        model = load_model(config)
    model = model.eval()
    task_summaries = {}
    aggregated_eval_logs = {}

    for spec in _task_specs(config):
        task_file = layout["root"] / f"{spec['eval_task']}.json"
        if not overwrite and task_file.is_file():
            eval_logs = json.loads(task_file.read_text(encoding="utf-8"))
        else:
            eval_batch_size = config.evaluation.batch_size
            aux_batch_size = max(1, config.evaluation.batch_size // 4)
            eval_loader = _build_loader(
                config,
                tokenizer,
                spec["split"],
                spec["eval_task"],
                spec["question_key"],
                spec["answer_key"],
                eval_batch_size,
            )
            base_eval_loader = _build_loader(
                config,
                tokenizer,
                spec["split"],
                spec["eval_task"],
                spec["question_key"],
                spec["base_answer_key"],
                aux_batch_size,
            )
            perturb_loader = _build_loader(
                config,
                tokenizer,
                spec["split"],
                spec["eval_task"],
                spec["question_key"],
                spec["perturbed_answer_key"],
                aux_batch_size,
            )
            eval_logs = _get_all_evals(
                config,
                model,
                tokenizer,
                eval_loader,
                base_eval_loader,
                perturb_loader,
                normalize_gt=spec["normalize_gt"],
            )
            with open(task_file, "w", encoding="utf-8") as handle:
                json.dump(eval_logs, handle, indent=2, ensure_ascii=False)
        aggregated_eval_logs[f"{spec['eval_task']}.json"] = eval_logs

        task_summary = _build_task_summary(eval_logs)
        task_summaries[spec["name"]] = {"eval_task": spec["eval_task"], **task_summary}
        append_metrics(layout["metrics"], {"task": spec["name"], **task_summaries[spec["name"]]})

    aggregated_eval_path = layout["root"] / "eval_log_aggregated.json"
    with open(aggregated_eval_path, "w", encoding="utf-8") as handle:
        json.dump(aggregated_eval_logs, handle, indent=2, ensure_ascii=False)

    model_utility = compute_model_utility(aggregated_eval_logs)
    aggregate_stat = dict(model_utility)
    if config.evaluation.retain_result:
        retain_result = json.loads(Path(config.evaluation.retain_result).read_text(encoding="utf-8"))
        aggregate_stat.update(compute_forget_quality(aggregated_eval_logs, retain_result))
    aggregate_stat_path = layout["root"] / "aggregate_stat.csv"
    write_aggregate_stat_csv(aggregate_stat_path, aggregate_stat)
    append_metrics(layout["metrics"], {"task": "aggregate", **aggregate_stat})

    summary["tasks"] = task_summaries
    summary["aggregated_eval_log"] = str(aggregated_eval_path)
    summary["aggregate_stat_file"] = str(aggregate_stat_path)
    summary["aggregate_stat"] = aggregate_stat
    if "Model Utility" in aggregate_stat:
        summary["model_utility"] = float(aggregate_stat["Model Utility"])
    if "Forget Quality" in aggregate_stat:
        summary["forget_quality"] = float(aggregate_stat["Forget Quality"])
    write_summary(layout["summary"], summary)
    return summary
