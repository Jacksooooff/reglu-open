from __future__ import annotations

import json
import logging
from pathlib import Path

from reglu.artifacts import ensure_run_layout, write_config_snapshot, write_summary
from reglu.config import RunConfig
from reglu.trainers.common import is_peft_adapter_dir, load_base_model, load_model, load_tokenizer


logger = logging.getLogger(__name__)


def _normalize_split_name(split: str) -> str:
    return str(split).lower().replace("wmdp_", "")


def _resolve_tasks(config: RunConfig) -> list[str]:
    tasks = list(getattr(config.evaluation, "tasks", []) or [])
    if tasks:
        return tasks
    return [f"wmdp_{_normalize_split_name(str(config.data.split))}", "mmlu"]


def _import_lm_eval():
    try:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError(
            "WMDP eval requires the `lm-eval` package. Install project dependencies "
            "from `pyproject.toml` so `reglu eval --config ...` can call lm_eval directly."
        ) from exc

    hflm_errors = []
    HFLM = None
    for module_path in (
        "lm_eval.models.hf_vlms",
        "lm_eval.models.huggingface",
    ):
        try:
            module = __import__(module_path, fromlist=["HFLM"])
            HFLM = getattr(module, "HFLM")
            break
        except Exception as exc:
            hflm_errors.append(f"{module_path}: {exc}")
    if HFLM is None:
        raise RuntimeError(
            "Failed to import `HFLM` from lm_eval. Tried: "
            + "; ".join(hflm_errors)
        )

    return simple_evaluate, TaskManager, HFLM


def _build_hflm(HFLM, model, tokenizer, batch_size: int, max_length: int | None = None):
    attempts = [
        {
            "pretrained": model,
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            **({"max_length": max_length} if max_length is not None else {}),
        },
        {
            "pretrained": model,
            "tokenizer": tokenizer,
            **({"max_length": max_length} if max_length is not None else {}),
        },
        {"pretrained": model},
    ]
    errors = []
    for kwargs in attempts:
        try:
            return HFLM(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs!r}: {exc}")
    try:
        return HFLM(model)
    except Exception as exc:
        errors.append(f"positional model: {exc}")
    raise RuntimeError("Failed to construct lm_eval HFLM wrapper: " + " | ".join(errors))


def _clean_metric_key(prefix: str, metric_name: str) -> str | None:
    if metric_name == "alias":
        return None
    base = metric_name.split(",", 1)[0].strip()
    return f"{prefix}/{base}"


def _summarize_task(task_manager, eval_results: dict, task_name: str) -> dict[str, float | int | str]:
    summary = {}
    if task_name in getattr(task_manager, "all_groups", set()):
        group_metrics = eval_results.get("groups", {}).get(task_name, {})
        for metric_name, value in group_metrics.items():
            key = _clean_metric_key(task_name, metric_name)
            if key is None:
                continue
            try:
                summary[key] = float(value)
            except (TypeError, ValueError):
                summary[key] = value
    else:
        task_metrics = eval_results.get("results", {}).get(task_name, {})
        for metric_name, value in task_metrics.items():
            key = _clean_metric_key(task_name, metric_name)
            if key is None:
                continue
            try:
                summary[key] = float(value)
            except (TypeError, ValueError):
                summary[key] = value
    return summary


def _load_eval_model(config: RunConfig):
    if config.evaluation.model_mode == "rila":
        from peft import PeftModel
        from reglu.methods.reglu import build_lora_model, maybe_apply_rila_cache

        base_model = load_base_model(config)
        if is_peft_adapter_dir(config.model.model_path):
            model = PeftModel.from_pretrained(base_model, config.model.model_path)
        else:
            model = build_lora_model(base_model, config)
        if config.method.rila_cache_path:
            maybe_apply_rila_cache(model, config.method.rila_cache_path, mode="w_only")
        return model.eval()
    return load_model(config).eval()


def run_wmdp_eval(config: RunConfig) -> dict:
    layout = ensure_run_layout(config.runtime.output_dir)
    write_config_snapshot(layout["config"], config)
    summary = {
        "command": "eval",
        "task": config.task,
        "model_family": config.model_family,
        "status": "dry_run" if config.runtime.dry_run else "ok",
    }
    if config.runtime.dry_run:
        write_summary(layout["summary"], summary)
        return summary

    simple_evaluate, TaskManager, HFLM = _import_lm_eval()
    tokenizer = load_tokenizer(config)
    model = _load_eval_model(config)
    lm = _build_hflm(
        HFLM,
        model,
        tokenizer,
        batch_size=int(config.evaluation.batch_size),
        max_length=int(config.evaluation.max_length) if config.evaluation.max_length else None,
    )

    task_manager = TaskManager()
    tasks = _resolve_tasks(config)
    eval_logs = {}
    aggregated_summary = {}
    overwrite = bool(config.evaluation.overwrite)
    eval_log_path = layout["root"] / "LMEval_EVAL.json"
    summary_path = layout["root"] / "LMEval_SUMMARY.json"

    if not overwrite and eval_log_path.is_file():
        eval_logs = json.loads(eval_log_path.read_text(encoding="utf-8"))
    if not overwrite and summary_path.is_file():
        aggregated_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for task_name in tasks:
        if not overwrite and task_name in eval_logs and eval_logs[task_name]:
            logger.info("Skipping lm_eval task %s because cached logs exist.", task_name)
            continue
        eval_logs.pop(task_name, None)
        results = simple_evaluate(
            model=lm,
            tasks=[task_name],
            task_manager=task_manager,
            batch_size=int(config.evaluation.batch_size),
            log_samples=True,
            apply_chat_template=False,
            system_instruction=None,
        )
        eval_logs[task_name] = results.get("samples", {})
        aggregated_summary.update(_summarize_task(task_manager, results, task_name))
        eval_log_path.write_text(
            json.dumps(eval_logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(aggregated_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary.update(
        {
            "tasks": tasks,
            "lmeval_log_file": str(eval_log_path),
            "lmeval_summary_file": str(summary_path),
        }
    )
    summary.update(aggregated_summary)
    write_summary(layout["summary"], summary)
    return summary
