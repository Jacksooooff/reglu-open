from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    family: str
    hf_key: str
    question_start_tag: str
    question_end_tag: str
    answer_tag: str
    gradient_checkpointing: bool
    default_lora_targets: dict[str, list[str]]


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llama2-7b": ModelSpec(
        family="llama2-7b",
        hf_key="meta-llama/Llama-2-7b-hf",
        question_start_tag="[INST] ",
        question_end_tag=" [/INST]",
        answer_tag="",
        gradient_checkpointing=True,
        default_lora_targets={
            "all": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        },
    ),
    "zephyr-7b-beta": ModelSpec(
        family="zephyr-7b-beta",
        hf_key="HuggingFaceH4/zephyr-7b-beta",
        question_start_tag="<|user|>\n",
        question_end_tag="</s>",
        answer_tag="<|assistant|>\n",
        gradient_checkpointing=False,
        default_lora_targets={
            "all": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        },
    ),
    "phi-1.5": ModelSpec(
        family="phi-1.5",
        hf_key="microsoft/phi-1_5",
        question_start_tag="Question: ",
        question_end_tag="\n",
        answer_tag="Answer: ",
        gradient_checkpointing=False,
        default_lora_targets={
            "all": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "self_attn": ["q_proj", "k_proj", "v_proj", "dense"],
            "mlp": ["fc1", "fc2"],
        },
    ),
}

# TOFU 旧配置里常用 `phi`，与 `phi-1.5` 同义
_MODEL_FAMILY_ALIASES: dict[str, str] = {
    "phi": "phi-1.5",
}


def get_model_spec(model_family: str) -> ModelSpec:
    model_family = _MODEL_FAMILY_ALIASES.get(model_family, model_family)
    try:
        return _MODEL_REGISTRY[model_family]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_family='{model_family}'. Supported: {supported}"
        ) from exc


def list_supported_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)
