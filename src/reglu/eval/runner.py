from __future__ import annotations

from reglu.config import RunConfig


def run_eval(config: RunConfig) -> dict:
    if config.task == "tofu":
        from reglu.eval.tofu import run_tofu_eval

        return run_tofu_eval(config)
    if config.task == "wmdp":
        from reglu.eval.wmdp import run_wmdp_eval

        return run_wmdp_eval(config)
    raise ValueError(f"Unsupported task '{config.task}'.")
