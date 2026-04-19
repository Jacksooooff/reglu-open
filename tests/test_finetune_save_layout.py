from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

pytest.importorskip("torch")


def test_finetune_saves_root_and_checkpoint_last(tmp_path, monkeypatch):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyPeftModel:
        def merge_and_unload(self):
            return self

    peft_mod.PeftModel = DummyPeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class DummyTrainer:
        def __init__(self, model=None, **kwargs):
            self.model = model

        def train(self):
            return types.SimpleNamespace(training_loss=0.25)

        def evaluate(self):
            return {}

    transformers_mod.Trainer = DummyTrainer
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    reglu_methods_mod = types.ModuleType("reglu.methods.reglu")
    reglu_methods_mod.__spec__ = importlib.machinery.ModuleSpec("reglu.methods.reglu", loader=None)
    reglu_methods_mod.build_lora_model = lambda model, config: model
    reglu_methods_mod.build_training_arguments = lambda config, output_dir, train_size: object()
    monkeypatch.setitem(sys.modules, "reglu.methods.reglu", reglu_methods_mod)

    from reglu.config import RunConfig
    from reglu.trainers.finetune import run_finetune

    config = RunConfig(task="tofu", model_family="llama2-7b")
    config.data.path = "locuslab/TOFU"
    config.data.split = "full"
    config.runtime.output_dir = str(tmp_path / "finetune")

    class DummyTokenizer:
        def save_pretrained(self, path: str):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class DummyModel:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=True)
            self.generation_config = types.SimpleNamespace(do_sample=True)

        def enable_input_require_grads(self):
            return None

        def gradient_checkpointing_enable(self, **kwargs):
            return None

        def save_pretrained(self, path: str):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("reglu.trainers.finetune.load_tokenizer", lambda _: DummyTokenizer())
    monkeypatch.setattr("reglu.trainers.finetune.load_model", lambda _: DummyModel())
    monkeypatch.setattr("reglu.trainers.finetune.TofuTextDataset", lambda **kwargs: [1, 2, 3])

    summary = run_finetune(config)

    output_dir = tmp_path / "finetune"
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "tokenizer.json").is_file()
    assert (output_dir / "checkpoints" / "checkpoint-last" / "config.json").is_file()
    assert (output_dir / "checkpoints" / "checkpoint-last" / "tokenizer.json").is_file()
    assert summary["checkpoint_dir"] == str(output_dir / "checkpoints" / "checkpoint-last")
