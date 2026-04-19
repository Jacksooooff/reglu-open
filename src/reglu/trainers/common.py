from __future__ import annotations

import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from reglu.config import RunConfig, resolve_model_path
from reglu.models import get_model_spec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(name: str):
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def load_tokenizer(config: RunConfig):
    from transformers import AutoTokenizer

    model_path = config.model.tokenizer_path or resolve_model_path(config)
    if config.model.local_files_only and not Path(model_path).exists():
        raise FileNotFoundError(f"Tokenizer path does not exist locally: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _ensure_local_model_dir(model_path: str) -> None:
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Model path {model_path} does not exist or is not a directory. Expected a local HuggingFace checkpoint directory."
        )
    has_weights = False
    for file in os.listdir(path):
        if re.search(r"pytorch.*\\.bin", file):
            has_weights = True
            break
        if re.search(r"model-.*\\.safetensors", file) or file.endswith(".safetensors"):
            has_weights = True
            break
    if not has_weights:
        raise FileNotFoundError(
            f"No local weight files (.bin or .safetensors) found under {model_path}."
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in local model directory: {model_path}")


def is_peft_adapter_dir(model_path: str | None) -> bool:
    return bool(model_path) and (Path(model_path) / "adapter_config.json").is_file()


def load_model(config: RunConfig, model_path: str | None = None):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = model_path or resolve_model_path(config)
    torch_dtype = resolve_torch_dtype(config.model.torch_dtype)
    if is_peft_adapter_dir(model_path):
        from peft import PeftModel

        if not config.model.base_model_path:
            raise ValueError(
                f"Model path {model_path} is a PEFT adapter directory; model.base_model_path must also be set."
            )
        base_model = load_model(config, model_path=config.model.base_model_path)
        return PeftModel.from_pretrained(base_model, model_path)
    if config.model.local_files_only:
        _ensure_local_model_dir(model_path)
    model_config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
    )
    load_kw = dict(
        config=model_config,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            **load_kw,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **load_kw,
        )
    spec = get_model_spec(config.model_family)
    if spec.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.generation_config.do_sample = True
    return model


def load_base_model(config: RunConfig):
    if not config.model.base_model_path:
        raise ValueError("model.base_model_path must be set to load a separate base model.")
    return load_model(config, model_path=config.model.base_model_path)
