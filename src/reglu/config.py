from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from reglu.models import get_model_spec


SUPPORTED_TASKS = {"tofu", "wmdp"}
COMMAND_ALIASES = {
    "finetune": "finetune",
    "forget": "unlearn",
    "unlearn": "unlearn",
    "eval": "evaluate",
    "evaluate": "evaluate",
}
SUPPORTED_COMMANDS = set(COMMAND_ALIASES)
PUBLIC_COMMANDS = ("finetune", "forget", "eval")
SUPPORTED_METHOD_NAMES = {"reglu"}
SUPPORTED_VARIANTS = {"ihl", "gd"}
SUPPORTED_INIT_STRATEGIES = {"rila"}
SUPPORTED_ROL_TARGETS = {"all_lora", "vproj_only"}

# TOFU 任务在 v1 中支持的底座（与 reglu.models.registry 对齐）
_TOFU_MODEL_FAMILIES = frozenset({"llama2-7b", "phi", "phi-1.5"})


@dataclass
class DataConfig:
    path: str = ""
    split: str | list[str] = ""
    question_key: str = "question"
    answer_key: str = "answer"
    base_answer_key: str = "paraphrased_answer"
    perturbed_answer_key: str = "perturbed_answer"
    max_length: int = 512
    subset_indices_file: str | None = None
    unlearn_eval_subset_indices_file: str | None = None


@dataclass
class ModelConfig:
    model_path: str | None = None
    base_model_path: str | None = None
    tokenizer_path: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class LoraConfig:
    enabled: bool = True
    targets: str = "all"
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0


@dataclass
class MethodConfig:
    name: str = "reglu"
    variant: str = "ihl"
    init_strategy: str = "rila"
    rila_beta: float = 0.8
    rila_cov_shrink: float = 0.0001
    rol_lambda: float = 0.0
    rol_rank: int = 128
    rol_targets: str = "all_lora"
    rila_cache_path: str | None = None
    require_rila_cache: bool = False
    rila_samples_per_split: int = 256


@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float | None = None
    logging_steps: int = 10
    save_steps: int | None = None
    save_total_limit: int | None = None
    eval_steps: int | None = None
    max_grad_norm: float = 1.0
    save_model: bool = True
    eval_only: bool = False
    eval_while_train: bool = False
    analysis_mode: bool = False
    seed: int = 42
    # False: plain `python` / 单卡可跑。True 时需用 `deepspeed`（或等价）启动并完成分布式初始化。
    use_deepspeed: bool = False
    deepspeed_config: str | None = None


@dataclass
class EvalConfig:
    batch_size: int = 4
    ds_size: int | None = None
    tasks: list[str] = field(default_factory=list)
    dataset_split: str = "test"
    split_symbol: str | None = None
    max_length: int = 200
    max_new_tokens: int | None = None
    save_generated_text: bool = True
    overwrite: bool = True
    metrics: list[str] = field(default_factory=list)
    dataset_path: str | None = None
    retain_result: str | None = None
    model_mode: str = "standard"


@dataclass
class RuntimeConfig:
    output_dir: str = "./outputs/reglu-run"
    device: str = "auto"
    dry_run: bool = False


@dataclass
class RunConfig:
    task: str
    model_family: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key '{key}' in section '{type(instance).__name__}'.")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def canonicalize_command(command: str) -> str:
    try:
        return COMMAND_ALIASES[command]
    except KeyError as exc:
        raise ValueError(f"Unsupported command '{command}'.") from exc


def normalize_public_command(command: str) -> str:
    command = canonicalize_command(command)
    if command == "unlearn":
        return "forget"
    if command == "evaluate":
        return "eval"
    return command


def load_run_config(path: str | Path, command: str) -> RunConfig:
    command = canonicalize_command(command)
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "task" not in payload or "model_family" not in payload:
        raise ValueError("Config must define top-level 'task' and 'model_family'.")
    config = RunConfig(task=payload["task"], model_family=payload["model_family"])
    _merge_dataclass(config, payload)
    validate_run_config(config, command)
    return config


def validate_run_config(config: RunConfig, command: str) -> None:
    command = canonicalize_command(command)
    default_data = DataConfig()
    default_training = TrainingConfig()
    default_evaluation = EvalConfig()
    if config.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{config.task}'.")
    get_model_spec(config.model_family)
    if config.method.name not in SUPPORTED_METHOD_NAMES:
        raise ValueError(f"Unsupported method.name '{config.method.name}'.")
    if config.method.variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported method.variant '{config.method.variant}'.")
    if config.method.init_strategy not in SUPPORTED_INIT_STRATEGIES:
        raise ValueError(
            f"Unsupported method.init_strategy '{config.method.init_strategy}'."
        )
    if config.method.rol_targets not in SUPPORTED_ROL_TARGETS:
        raise ValueError(
            f"Unsupported method.rol_targets '{config.method.rol_targets}'."
        )
    if config.evaluation.model_mode not in {"standard", "rila"}:
        raise ValueError(
            f"Unsupported evaluation.model_mode '{config.evaluation.model_mode}'."
        )
    if config.evaluation.metrics:
        raise ValueError(
            "evaluation.metrics is not a supported public v1 setting; task summaries are fixed."
        )
    if config.runtime.device != "auto":
        raise ValueError("runtime.device only supports 'auto' in v1.")
    if command == "finetune" and config.task != "tofu":
        raise ValueError("v1 only supports `finetune` for task='tofu'.")
    if command == "finetune" and config.model_family not in _TOFU_MODEL_FAMILIES:
        allowed = ", ".join(sorted(_TOFU_MODEL_FAMILIES))
        raise ValueError(f"v1 TOFU finetune supports model_family in {{{allowed}}}.")
    if command == "finetune" and config.evaluation != default_evaluation:
        raise ValueError("finetune does not use evaluation.* settings in v1; remove that section from the config.")
    if config.task == "wmdp" and config.model_family != "zephyr-7b-beta":
        raise ValueError("v1 only supports WMDP with model_family='zephyr-7b-beta'.")
    if config.task == "tofu" and config.model_family not in _TOFU_MODEL_FAMILIES:
        allowed = ", ".join(sorted(_TOFU_MODEL_FAMILIES))
        raise ValueError(f"v1 TOFU supports model_family in {{{allowed}}}.")
    if config.training.batch_size <= 0 or config.training.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be > 0.")
    if config.method.rol_rank < 0:
        raise ValueError("method.rol_rank must be >= 0.")
    if config.method.rila_samples_per_split <= 0:
        raise ValueError("method.rila_samples_per_split must be > 0.")
    if config.lora.r < 0:
        raise ValueError("lora.r must be >= 0.")
    if config.training.max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be > 0.")
    if config.method.require_rila_cache and not config.method.rila_cache_path:
        raise ValueError(
            "method.require_rila_cache=true requires method.rila_cache_path."
        )
    if command == "evaluate":
        if config.training != default_training:
            raise ValueError("eval does not use training.* settings in v1; remove that section from the config.")
        if config.data.max_length != default_data.max_length:
            raise ValueError("eval uses evaluation.max_length in v1; data.max_length is not a supported eval setting.")
        if config.task == "tofu" and not config.data.path:
            raise ValueError("TOFU eval requires data.path.")
        if config.task == "tofu":
            if config.evaluation.tasks:
                raise ValueError("TOFU eval uses a fixed official task set in v1; evaluation.tasks is not supported.")
            if config.evaluation.dataset_path is not None:
                raise ValueError("TOFU eval reads from data.path and split names directly; evaluation.dataset_path is not supported in v1.")
            if config.evaluation.dataset_split != default_evaluation.dataset_split:
                raise ValueError("TOFU eval does not use evaluation.dataset_split in v1.")
            if config.data.subset_indices_file is not None:
                raise ValueError("TOFU eval only supports data.unlearn_eval_subset_indices_file for the forget split in v1.")
        if config.task == "wmdp":
            if config.data.path:
                raise ValueError("WMDP eval uses lm_eval task definitions directly; data.path is not supported in v1.")
            if config.data.subset_indices_file is not None or config.data.unlearn_eval_subset_indices_file is not None:
                raise ValueError("WMDP eval does not use subset index files in v1.")
            if (
                config.data.question_key != default_data.question_key
                or config.data.answer_key != default_data.answer_key
                or config.data.base_answer_key != default_data.base_answer_key
                or config.data.perturbed_answer_key != default_data.perturbed_answer_key
            ):
                raise ValueError("WMDP eval does not use TOFU-style data.* key overrides in v1.")
            if config.evaluation.dataset_path is not None:
                raise ValueError(
                    "WMDP eval uses lm_eval task definitions directly; evaluation.dataset_path is not supported in v1."
                )
            if config.evaluation.ds_size is not None:
                raise ValueError(
                    "WMDP eval uses lm_eval task definitions directly; evaluation.ds_size is not supported in v1."
                )
            if config.evaluation.retain_result is not None:
                raise ValueError("evaluation.retain_result is only supported for TOFU eval.")
            if config.evaluation.split_symbol is not None:
                raise ValueError("evaluation.split_symbol is only supported for TOFU eval.")
            if config.evaluation.max_new_tokens is not None:
                raise ValueError("evaluation.max_new_tokens is not used by WMDP lm_eval in v1.")
            if config.evaluation.dataset_split != "test":
                raise ValueError("evaluation.dataset_split is not configurable for WMDP lm_eval in v1.")
            if config.evaluation.save_generated_text != default_evaluation.save_generated_text:
                raise ValueError("evaluation.save_generated_text is not used by WMDP lm_eval in v1.")


def resolve_model_path(config: RunConfig) -> str:
    spec = get_model_spec(config.model_family)
    return config.model.model_path or spec.hf_key
