from __future__ import annotations

from .cache import load_rol_bases

__all__ = [
    "RegLUUnlearnTrainer",
    "build_lora_model",
    "build_training_arguments",
    "load_rol_bases",
    "initialize_rila",
    "maybe_apply_rila_cache",
]


def __getattr__(name: str):
    if name == "initialize_rila":
        from .rila import initialize_rila

        return initialize_rila
    if name in {
        "RegLUUnlearnTrainer",
        "build_lora_model",
        "build_training_arguments",
        "maybe_apply_rila_cache",
    }:
        from .core import (
            RegLUUnlearnTrainer,
            build_lora_model,
            build_training_arguments,
            maybe_apply_rila_cache,
        )

        mapping = {
            "RegLUUnlearnTrainer": RegLUUnlearnTrainer,
            "build_lora_model": build_lora_model,
            "build_training_arguments": build_training_arguments,
            "maybe_apply_rila_cache": maybe_apply_rila_cache,
        }
        return mapping[name]
    raise AttributeError(name)
