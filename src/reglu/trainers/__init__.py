from __future__ import annotations


def run_finetune(config):
    from .finetune import run_finetune as _run_finetune

    return _run_finetune(config)


def run_unlearn(config):
    from .unlearn import run_unlearn as _run_unlearn

    return _run_unlearn(config)


__all__ = ["run_finetune", "run_unlearn"]
