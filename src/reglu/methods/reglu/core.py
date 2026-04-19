from __future__ import annotations

import copy
import csv
import importlib.util
import math
import os
import time
from pathlib import Path
from inspect import signature

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import Tensor, tensor
from torchmetrics.functional.classification.confusion_matrix import (
    _multiclass_confusion_matrix_format,
)
from torchmetrics.functional.classification.hinge import (
    _hinge_loss_compute,
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
)
from torchmetrics.utilities.data import to_onehot
from transformers import Trainer

from reglu.config import RunConfig
from reglu.models import get_model_spec


def _custom_multiclass_hinge_loss_update(
    preds,
    target,
    alpha,
    squared,
    multiclass_mode="crammer-singer",
):
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)
    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]
    measures = torch.clamp(alpha + margin, 0)
    if squared:
        measures = measures.pow(2)
    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total


def multiclass_hinge_loss(
    preds,
    target,
    num_classes,
    alpha=1.0,
    squared=False,
    multiclass_mode="crammer-singer",
    ignore_index=None,
    validate_args=True,
):
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(
        preds,
        target,
        ignore_index,
        convert_to_labels=False,
    )
    measures, total = _custom_multiclass_hinge_loss_update(
        preds,
        target,
        alpha,
        squared,
        multiclass_mode,
    )
    return _hinge_loss_compute(measures, total)


def build_lora_model(model, config: RunConfig):
    if not config.lora.enabled or config.lora.r == 0:
        return model
    model_spec = get_model_spec(config.model_family)
    target_modules = model_spec.default_lora_targets[config.lora.targets]
    lora_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        target_modules=target_modules,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def maybe_apply_rila_cache(model, cache_path: str | None, mode: str = "w_only") -> bool:
    if not cache_path:
        return False
    path = Path(cache_path)
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    layers = payload.get("layers", {})
    if not isinstance(layers, dict):
        return False
    for name, module in model.named_modules():
        if name not in layers:
            continue
        entry = layers[name]
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer")):
            continue
        with torch.no_grad():
            if "W" in entry:
                module.base_layer.weight.copy_(
                    entry["W"].to(
                        dtype=module.base_layer.weight.dtype,
                        device=module.base_layer.weight.device,
                    ).contiguous()
                )
            if mode == "all" and "A" in entry and "B" in entry:
                adapter = "default"
                module.lora_A[adapter].weight.copy_(
                    entry["A"].to(
                        dtype=module.lora_A[adapter].weight.dtype,
                        device=module.lora_A[adapter].weight.device,
                    ).contiguous()
                )
                module.lora_B[adapter].weight.copy_(
                    entry["B"].to(
                        dtype=module.lora_B[adapter].weight.dtype,
                        device=module.lora_B[adapter].weight.device,
                    ).contiguous()
                )
    return True


def build_training_arguments(config: RunConfig, output_dir: str, train_size: int):
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    batch_size = config.training.batch_size
    grad_acc = config.training.gradient_accumulation_steps
    global_batch = max(1, batch_size * grad_acc * max(1, num_devices))
    requested_max_steps = config.training.max_steps
    num_epochs = max(1, int(config.training.num_epochs))

    if requested_max_steps is not None:
        max_steps = max(1, int(requested_max_steps))
        steps_per_epoch = max(1, math.ceil(train_size / global_batch))
        steps_per_epoch = min(steps_per_epoch, max_steps)
        effective_epochs = max(1, math.ceil(max_steps / max(1, steps_per_epoch)))
    else:
        # 与 TOFU/finetune.py 一致：max_steps = num_epochs * len(dataset) // (B * G * world_size)
        max_steps = max(1, num_epochs * train_size // global_batch)
        effective_epochs = num_epochs
        steps_per_epoch = max(1, max_steps // num_epochs)

    if config.training.warmup_ratio is not None:
        warmup_steps = max(1, math.ceil(max_steps * float(config.training.warmup_ratio)))
    elif config.training.warmup_steps > 0:
        warmup_steps = int(config.training.warmup_steps)
    else:
        # TOFU: warmup_steps=max(1, max_steps//cfg.num_epochs)
        warmup_steps = max(1, max_steps // num_epochs)

    ds_config_path = None
    if config.training.use_deepspeed:
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "training.use_deepspeed=true requires the `deepspeed` package. "
                "Install it in the current environment or set training.use_deepspeed=false."
            )
        if config.training.deepspeed_config:
            ds_config_path = str(Path(config.training.deepspeed_config).expanduser())
            if not os.path.isabs(ds_config_path):
                ds_config_path = str((Path(__file__).resolve().parents[4] / ds_config_path).resolve())
        else:
            default_ds = Path(__file__).resolve().parents[4] / "configs" / "ds_config.json"
            ds_config_path = str(default_ds)
        if not Path(ds_config_path).is_file():
            raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")

    bf16_available = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_available = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_available = False

    analysis_mode = bool(config.training.analysis_mode)
    if analysis_mode:
        save_steps = 5
    elif config.training.save_steps is not None:
        save_steps = max(1, min(int(config.training.save_steps), max_steps))
    else:
        # TOFU finetune.py: save_steps=max_steps（仅在训练结束时落盘）
        save_steps = max(1, max_steps)
    save_total_limit = None if analysis_mode else config.training.save_total_limit
    eval_strategy = "steps" if config.training.eval_while_train else "no"
    if analysis_mode:
        eval_steps = 5
    elif config.training.eval_steps is not None:
        eval_steps = max(1, int(config.training.eval_steps))
    else:
        eval_steps = steps_per_epoch

    ta_sig = signature(transformers.TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        num_train_epochs=effective_epochs,
        learning_rate=config.training.learning_rate,
        bf16=bf16_available,
        bf16_full_eval=bf16_available,
        logging_steps=max(
            1,
            int(config.training.logging_steps)
            if config.training.logging_steps and config.training.logging_steps > 0
            else max_steps // max(1, 20),
        ),
        logging_dir=str(Path(output_dir) / "logs"),
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if config.training.save_model and not config.training.eval_only else "no",
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        weight_decay=config.training.weight_decay,
        seed=config.training.seed,
        max_grad_norm=config.training.max_grad_norm,
        report_to=[],
        remove_unused_columns=False,
    )
    if "evaluation_strategy" in ta_sig:
        ta_kwargs["evaluation_strategy"] = eval_strategy
    if "eval_strategy" in ta_sig:
        ta_kwargs["eval_strategy"] = eval_strategy
    if save_total_limit is not None and ta_kwargs["save_strategy"] != "no":
        ta_kwargs["save_total_limit"] = int(save_total_limit)
    if ds_config_path and "deepspeed" in ta_sig:
        ta_kwargs["deepspeed"] = ds_config_path
    ta_filtered = {k: v for k, v in ta_kwargs.items() if k in ta_sig}
    return transformers.TrainingArguments(**ta_filtered)


class _EfficiencyLoggerCallback(transformers.TrainerCallback):
    def __init__(self, trainer, csv_path: Path):
        self.trainer = trainer
        self.csv_path = Path(csv_path)
        self._fp = None
        self._writer = None
        self._step_start = None
        self._total_seconds = 0.0
        self._paused = False

    def _ensure_writer(self):
        if self._writer is not None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists()
        self._fp = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if not file_exists:
            self._writer.writerow(
                [
                    "step",
                    "step_latency_seconds",
                    "cumulative_gpu_hours",
                    "forget_loss",
                    "retain_loss",
                    "rol_loss",
                    "rol_penalty",
                ]
            )
            self._fp.flush()

    def _pause(self):
        self._paused = True
        self._step_start = None

    def _resume(self):
        self._paused = False

    def _close(self):
        if self._fp is None:
            return
        try:
            self._fp.flush()
        finally:
            self._fp.close()
        self._fp = None
        self._writer = None

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.trainer.is_world_process_zero() or self._paused:
            self._step_start = None
            return control
        if self._step_start is None:
            self._step_start = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.trainer.is_world_process_zero() or self._paused or self._step_start is None:
            self._step_start = None
            return control
        step_seconds = max(0.0, time.perf_counter() - self._step_start)
        self._total_seconds += step_seconds
        self._ensure_writer()
        stats = getattr(self.trainer, "_pending_step_stats", {}) or {}
        self._writer.writerow(
            [
                int(state.global_step),
                step_seconds,
                self._total_seconds / 3600.0,
                stats.get("forget_loss"),
                stats.get("retain_loss"),
                stats.get("rol_loss"),
                stats.get("rol_penalty"),
            ]
        )
        self._fp.flush()
        self.trainer._pending_step_stats = None
        self._step_start = None
        return control

    def on_evaluate_begin(self, args, state, control, **kwargs):
        if self.trainer.is_world_process_zero():
            self._pause()
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer.is_world_process_zero():
            self._resume()
        return control

    def on_save_begin(self, args, state, control, **kwargs):
        if self.trainer.is_world_process_zero():
            self._pause()
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.trainer.is_world_process_zero():
            self._resume()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer.is_world_process_zero():
            self._close()
        return control


class RegLUUnlearnTrainer(Trainer):
    def __init__(
        self,
        *args,
        method_config,
        run_config=None,
        tokenizer=None,
        rol_basis_dict=None,
        **kwargs,
    ):
        self.method_config = method_config
        self.run_config = copy.deepcopy(run_config) if run_config is not None else None
        self.eval_tokenizer = tokenizer
        self.rol_basis_dict = rol_basis_dict
        self.rol_targets = method_config.rol_targets
        self.rol_lambda = float(method_config.rol_lambda)
        self.rol_rank = int(method_config.rol_rank)
        self.require_rila_cache = bool(getattr(method_config, "require_rila_cache", False))
        self.loss_type = method_config.variant.lower()
        super().__init__(*args, **kwargs)
        self._pending_step_stats = None
        self._rol_layers = []
        self._setup_rol_buffers()
        csv_path = Path(self.args.output_dir) / "efficiency_log.csv"
        self.add_callback(_EfficiencyLoggerCallback(self, csv_path))

    def _set_efficiency_paused(self, paused: bool) -> None:
        for callback in getattr(self.callback_handler, "callbacks", []):
            if paused and hasattr(callback, "_pause"):
                callback._pause()
            if not paused and hasattr(callback, "_resume"):
                callback._resume()

    def _scalarize(self, value):
        if value is None:
            return None
        try:
            return float(value.detach().float().item())
        except Exception:
            try:
                return float(value)
            except Exception:
                return None

    def _stash_step_stats(
        self,
        forget_loss=None,
        retain_loss=None,
        rol_loss=None,
        rol_penalty=None,
    ) -> None:
        if not getattr(self.model, "training", False):
            return
        stats = self._pending_step_stats or {}
        if forget_loss is not None:
            stats["forget_loss"] = self._scalarize(forget_loss)
        if retain_loss is not None:
            stats["retain_loss"] = self._scalarize(retain_loss)
        if rol_loss is not None:
            stats["rol_loss"] = self._scalarize(rol_loss)
        if rol_penalty is not None:
            stats["rol_penalty"] = self._scalarize(rol_penalty)
        self._pending_step_stats = stats

    def _target_matches(self, module_name: str) -> bool:
        if self.rol_targets == "all_lora":
            return True
        lowered = module_name.lower()
        return any(tag in lowered for tag in ("v_proj", "value_proj", "vproj"))

    def _random_rol_basis(self, dim: int, k: int, dtype, device):
        k_eff = max(0, min(k, dim))
        if k_eff == 0:
            return torch.zeros(dim, 0, dtype=dtype, device=device)
        base = torch.randn(dim, k_eff, dtype=torch.float32, device=device)
        q, _ = torch.linalg.qr(base, mode="reduced")
        return q[:, :k_eff].to(dtype=dtype)

    def _setup_rol_buffers(self):
        model = getattr(self.model, "module", self.model)
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer")):
                continue
            if not self._target_matches(name):
                continue
            adapter = "default"
            lora_b_weight = module.lora_B[adapter].weight
            out_features = lora_b_weight.shape[0]
            k_b = min(self.rol_rank, out_features)
            loaded = self.rol_basis_dict.get(name) if self.rol_basis_dict else None
            if self.require_rila_cache and not isinstance(loaded, torch.Tensor):
                raise ValueError(
                    f"ROL requires cached retain basis for layer '{name}', but none was provided."
                )
            q_retain = (
                loaded[:, :k_b].to(dtype=lora_b_weight.dtype, device=lora_b_weight.device)
                if isinstance(loaded, torch.Tensor) and loaded.shape[0] == out_features and loaded.shape[1] >= k_b
                else self._random_rol_basis(out_features, k_b, lora_b_weight.dtype, lora_b_weight.device)
            )
            module.register_buffer(f"rol_Q_{adapter}", q_retain, persistent=True)
            self._rol_layers.append(
                {
                    "module": module,
                    "adapter": adapter,
                    "q_attr": f"rol_Q_{adapter}",
                }
            )

    def _compute_ihl_losses(self, model, forget_inputs, retain_inputs):
        input_ids, labels, attention_mask = forget_inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous().squeeze().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().squeeze().view(-1)
        mask = shift_labels != -100
        forget_loss = multiclass_hinge_loss(
            shift_logits[mask, :],
            shift_labels[mask],
            shift_logits.size(-1),
        )
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        retain_outputs = model(
            retain_input_ids,
            labels=retain_labels,
            attention_mask=retain_attention_mask,
        )
        return forget_loss, retain_outputs.loss, outputs

    def _compute_gd_losses(self, model, forget_inputs, retain_inputs):
        input_ids, labels, attention_mask = forget_inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss = -outputs.loss
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        retain_outputs = model(
            retain_input_ids,
            labels=retain_labels,
            attention_mask=retain_attention_mask,
        )
        return forget_loss, retain_outputs.loss, outputs

    def _rol_penalty(self, reference: Tensor) -> Tensor:
        if not self._rol_layers or self.rol_rank <= 0:
            return reference.new_zeros(())
        total = torch.zeros((), device=reference.device, dtype=torch.float32)
        for entry in self._rol_layers:
            module = entry["module"]
            adapter = entry["adapter"]
            b_weight = module.lora_B[adapter].weight.to(dtype=torch.float32)
            q_retain = getattr(module, entry["q_attr"]).detach().to(dtype=torch.float32)
            total = total + torch.matmul(b_weight.transpose(0, 1), q_retain).pow(2).sum()
        return total

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs, retain_inputs = inputs
        if self.loss_type == "ihl":
            forget_loss, retain_loss, outputs = self._compute_ihl_losses(
                model,
                forget_inputs,
                retain_inputs,
            )
        elif self.loss_type == "gd":
            forget_loss, retain_loss, outputs = self._compute_gd_losses(
                model,
                forget_inputs,
                retain_inputs,
            )
        else:
            raise ValueError(f"Unsupported ReGLU variant '{self.loss_type}'.")
        loss = forget_loss + retain_loss
        rol_penalty = None
        if self.rol_lambda > 0:
            raw_penalty = self._rol_penalty(loss)
            rol_penalty = self.rol_lambda * raw_penalty
            loss = loss + rol_penalty
            if getattr(self.model, "training", False):
                self.log(
                    {
                        "train/forget_loss": float(forget_loss.detach().cpu()),
                        "train/retain_loss": float(retain_loss.detach().cpu()),
                        "train/rol_loss": float(raw_penalty.detach().cpu()),
                        "train/rol_penalty": float(rol_penalty.detach().cpu()),
                        "train/total_loss": float(loss.detach().cpu()),
                    }
                )
            self._stash_step_stats(
                forget_loss=forget_loss,
                retain_loss=retain_loss,
                rol_loss=raw_penalty,
                rol_penalty=rol_penalty,
            )
        else:
            self._stash_step_stats(
                forget_loss=forget_loss,
                retain_loss=retain_loss,
            )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        forget_inputs, _ = inputs
        input_ids, labels, attention_mask = forget_inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        return outputs.loss, outputs.logits, labels

    def _save_eval_checkpoint(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(str(model_dir))
        tokenizer = self.eval_tokenizer or getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(model_dir))

    def _build_eval_config(self, output_dir: Path, model_dir: Path):
        if self.run_config is None:
            return None
        eval_config = copy.deepcopy(self.run_config)
        eval_config.runtime.output_dir = str(output_dir)
        eval_config.runtime.dry_run = False
        eval_config.model.model_path = str(model_dir)
        eval_config.model.tokenizer_path = str(model_dir)
        if (model_dir / "adapter_config.json").is_file() and not eval_config.model.base_model_path:
            eval_config.model.base_model_path = self.run_config.model.model_path
        return eval_config

    def _flatten_eval_metrics(self, payload, prefix: str) -> dict[str, float]:
        metrics: dict[str, float] = {}

        def _walk(path: list[str], value) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    _walk([*path, str(key)], nested)
                return
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                metric_name = "_".join(
                    part.replace("/", "_").replace(".", "_") for part in path if part
                )
                if metric_name:
                    metrics[f"{prefix}_{metric_name}"] = float(value)

        _walk([], payload)
        return metrics

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        if self.run_config is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        step = int(getattr(self.state, "global_step", 0))
        eval_root = Path(self.args.output_dir) / f"checkpoint-{step}"
        model_dir = eval_root / "model"

        self._set_efficiency_paused(True)
        try:
            self._save_eval_checkpoint(model_dir)
            eval_config = self._build_eval_config(eval_root, model_dir)
            if eval_config is None:
                return {}
            from reglu.eval import run_eval

            summary = run_eval(eval_config)
        finally:
            self._set_efficiency_paused(False)

        metrics = self._flatten_eval_metrics(summary, metric_key_prefix)
        metrics[f"{metric_key_prefix}_step"] = float(step)
        if metrics:
            self.log(metrics)
        return metrics
