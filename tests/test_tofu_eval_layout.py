from __future__ import annotations

import importlib.machinery
import json
import sys
import types

import pytest


def test_tofu_eval_writes_legacy_root_level_logs(tmp_path, monkeypatch):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyPeftModel:
        @classmethod
        def from_pretrained(cls, model, model_path):
            return model

    peft_mod.PeftModel = DummyPeftModel
    peft_mod.LoraConfig = lambda *args, **kwargs: object()
    peft_mod.get_peft_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    reglu_methods_mod = types.ModuleType("reglu.methods.reglu")
    reglu_methods_mod.__spec__ = importlib.machinery.ModuleSpec("reglu.methods.reglu", loader=None)
    reglu_methods_mod.build_lora_model = lambda model, config: model
    reglu_methods_mod.maybe_apply_rila_cache = lambda model, cache_path, mode="w_only": False
    monkeypatch.setitem(sys.modules, "reglu.methods.reglu", reglu_methods_mod)

    from reglu.config import RunConfig
    from reglu.eval.tofu import run_tofu_eval

    config = RunConfig(task="tofu", model_family="llama2-7b")
    config.data.path = "locuslab/TOFU"
    config.data.split = "forget10"
    config.runtime.output_dir = str(tmp_path / "tofu-eval")

    class DummyModel:
        device = "cpu"

        def eval(self):
            return self

    class DummyTokenizer:
        pass

    monkeypatch.setattr("reglu.eval.tofu.load_tokenizer", lambda _: DummyTokenizer())
    monkeypatch.setattr("reglu.eval.tofu.load_model", lambda _: DummyModel())
    monkeypatch.setattr("reglu.eval.tofu._build_loader", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "reglu.eval.tofu._get_all_evals",
        lambda *args, **kwargs: {
            "avg_gt_loss": {0: 1.0},
            "rougeL_recall": {0: 0.4},
            "rouge1_recall": {0: 0.5},
            "avg_paraphrased_loss": {0: 1.0},
            "average_perturb_loss": {0: [2.0, 2.1]},
            "truth_ratio": {0: 0.3},
        },
    )

    summary = run_tofu_eval(config)

    output_dir = tmp_path / "tofu-eval"
    assert (output_dir / "eval_log.json").is_file()
    assert (output_dir / "eval_real_author_wo_options.json").is_file()
    assert (output_dir / "eval_real_world_wo_options.json").is_file()
    assert (output_dir / "eval_log_forget.json").is_file()
    assert (output_dir / "eval_log_aggregated.json").is_file()
    assert (output_dir / "aggregate_stat.csv").is_file()
    assert summary["aggregated_eval_log"] == str(output_dir / "eval_log_aggregated.json")


def test_tofu_eval_reuses_existing_task_logs_when_overwrite_disabled(tmp_path, monkeypatch):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyPeftModel:
        @classmethod
        def from_pretrained(cls, model, model_path):
            return model

    peft_mod.PeftModel = DummyPeftModel
    peft_mod.LoraConfig = lambda *args, **kwargs: object()
    peft_mod.get_peft_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    reglu_methods_mod = types.ModuleType("reglu.methods.reglu")
    reglu_methods_mod.__spec__ = importlib.machinery.ModuleSpec("reglu.methods.reglu", loader=None)
    reglu_methods_mod.build_lora_model = lambda model, config: model
    reglu_methods_mod.maybe_apply_rila_cache = lambda model, cache_path, mode="w_only": False
    monkeypatch.setitem(sys.modules, "reglu.methods.reglu", reglu_methods_mod)

    from reglu.config import RunConfig
    from reglu.eval.tofu import run_tofu_eval

    config = RunConfig(task="tofu", model_family="llama2-7b")
    config.data.path = "locuslab/TOFU"
    config.data.split = "forget10"
    config.evaluation.overwrite = False
    config.runtime.output_dir = str(tmp_path / "tofu-eval")

    output_dir = tmp_path / "tofu-eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_payload = {
        "avg_gt_loss": {"0": 1.0},
        "rougeL_recall": {"0": 0.4},
        "rouge1_recall": {"0": 0.5},
        "avg_paraphrased_loss": {"0": 1.0},
        "average_perturb_loss": {"0": [2.0, 2.1]},
        "truth_ratio": {"0": 0.3},
    }
    for filename in (
        "eval_log.json",
        "eval_real_author_wo_options.json",
        "eval_real_world_wo_options.json",
        "eval_log_forget.json",
    ):
        (output_dir / filename).write_text(json.dumps(task_payload), encoding="utf-8")

    class DummyModel:
        device = "cpu"

        def eval(self):
            return self

    class DummyTokenizer:
        pass

    monkeypatch.setattr("reglu.eval.tofu.load_tokenizer", lambda _: DummyTokenizer())
    monkeypatch.setattr("reglu.eval.tofu.load_model", lambda _: DummyModel())
    monkeypatch.setattr(
        "reglu.eval.tofu._build_loader",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached TOFU eval should not rebuild loaders")),
    )
    monkeypatch.setattr(
        "reglu.eval.tofu._get_all_evals",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached TOFU eval should not rerun model evaluation")),
    )

    summary = run_tofu_eval(config)

    assert summary["model_utility"] >= 0.0
    assert (output_dir / "eval_log_aggregated.json").is_file()
    assert (output_dir / "aggregate_stat.csv").is_file()
