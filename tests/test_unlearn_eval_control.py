from __future__ import annotations

import importlib.machinery
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

pytest.importorskip("torch")

def test_unlearn_trainer_evaluate_exports_checkpoint_and_runs_public_eval(tmp_path, monkeypatch):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyPeftModel:
        @classmethod
        def from_pretrained(cls, model, model_path):
            return model

    class DummyLoraConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    peft_mod.PeftModel = DummyPeftModel
    peft_mod.LoraConfig = DummyLoraConfig
    peft_mod.get_peft_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

    class DummyTrainerCallback:
        pass

    class DummyTrainingArguments:
        def __init__(self, *args, **kwargs):
            pass

    transformers_mod.Trainer = DummyTrainer
    transformers_mod.TrainerCallback = DummyTrainerCallback
    transformers_mod.TrainingArguments = DummyTrainingArguments
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    torchmetrics_mod = types.ModuleType("torchmetrics")
    torchmetrics_mod.__spec__ = importlib.machinery.ModuleSpec("torchmetrics", loader=None)
    tm_functional = types.ModuleType("torchmetrics.functional")
    tm_functional.__spec__ = importlib.machinery.ModuleSpec("torchmetrics.functional", loader=None)
    tm_classification = types.ModuleType("torchmetrics.functional.classification")
    tm_classification.__spec__ = importlib.machinery.ModuleSpec(
        "torchmetrics.functional.classification", loader=None
    )
    tm_confusion = types.ModuleType("torchmetrics.functional.classification.confusion_matrix")
    tm_confusion.__spec__ = importlib.machinery.ModuleSpec(
        "torchmetrics.functional.classification.confusion_matrix", loader=None
    )
    tm_hinge = types.ModuleType("torchmetrics.functional.classification.hinge")
    tm_hinge.__spec__ = importlib.machinery.ModuleSpec(
        "torchmetrics.functional.classification.hinge", loader=None
    )
    tm_utilities = types.ModuleType("torchmetrics.utilities")
    tm_utilities.__spec__ = importlib.machinery.ModuleSpec("torchmetrics.utilities", loader=None)
    tm_utilities_data = types.ModuleType("torchmetrics.utilities.data")
    tm_utilities_data.__spec__ = importlib.machinery.ModuleSpec(
        "torchmetrics.utilities.data", loader=None
    )

    tm_confusion._multiclass_confusion_matrix_format = lambda preds, target, *args, **kwargs: (
        preds,
        target,
    )
    tm_hinge._hinge_loss_compute = lambda measures, total: measures.sum() / total
    tm_hinge._multiclass_hinge_loss_arg_validation = lambda *args, **kwargs: None
    tm_hinge._multiclass_hinge_loss_tensor_validation = lambda *args, **kwargs: None
    tm_utilities_data.to_onehot = lambda target, num_classes: target

    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics_mod)
    monkeypatch.setitem(sys.modules, "torchmetrics.functional", tm_functional)
    monkeypatch.setitem(sys.modules, "torchmetrics.functional.classification", tm_classification)
    monkeypatch.setitem(
        sys.modules,
        "torchmetrics.functional.classification.confusion_matrix",
        tm_confusion,
    )
    monkeypatch.setitem(
        sys.modules,
        "torchmetrics.functional.classification.hinge",
        tm_hinge,
    )
    monkeypatch.setitem(sys.modules, "torchmetrics.utilities", tm_utilities)
    monkeypatch.setitem(sys.modules, "torchmetrics.utilities.data", tm_utilities_data)

    import reglu.eval
    from reglu.config import RunConfig
    from reglu.methods.reglu.core import RegLUUnlearnTrainer

    pauses: list[str] = []

    class PauseCallback:
        def _pause(self):
            pauses.append("pause")

        def _resume(self):
            pauses.append("resume")

    trainer = RegLUUnlearnTrainer.__new__(RegLUUnlearnTrainer)
    trainer.callback_handler = SimpleNamespace(callbacks=[PauseCallback()])
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / "forget-run"))
    trainer.state = SimpleNamespace(global_step=25)
    trainer.eval_tokenizer = None
    trainer.tokenizer = None
    trainer.run_config = RunConfig(task="wmdp", model_family="zephyr-7b-beta")
    trainer.run_config.model.model_path = "/models/base-zephyr"
    trainer.run_config.runtime.output_dir = str(tmp_path / "forget-run")

    class DummyTokenizer:
        def save_pretrained(self, path: str):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    trainer.eval_tokenizer = DummyTokenizer()

    def fake_save_model(path: str):
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    logged = {}
    captured = {}

    trainer.save_model = fake_save_model
    trainer.log = lambda metrics: logged.update(metrics)

    def fake_run_eval(config):
        captured["config"] = config
        return {
            "command": "eval",
            "task": "wmdp",
            "status": "ok",
            "wmdp_bio/acc": 0.75,
        }

    monkeypatch.setattr(reglu.eval, "run_eval", fake_run_eval)

    metrics = trainer.evaluate(metric_key_prefix="eval")

    eval_root = tmp_path / "forget-run" / "checkpoint-25"
    model_dir = eval_root / "model"
    assert pauses == ["pause", "resume"]
    assert model_dir.is_dir()
    assert (model_dir / "adapter_config.json").is_file()
    assert (model_dir / "tokenizer.json").is_file()
    assert captured["config"].runtime.output_dir == str(eval_root)
    assert captured["config"].model.model_path == str(model_dir)
    assert captured["config"].model.tokenizer_path == str(model_dir)
    assert captured["config"].model.base_model_path == "/models/base-zephyr"
    assert metrics["eval_wmdp_bio_acc"] == pytest.approx(0.75)
    assert logged["eval_wmdp_bio_acc"] == pytest.approx(0.75)
    assert metrics["eval_step"] == pytest.approx(25.0)
