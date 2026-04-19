from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

pytest.importorskip("torch")


def test_build_training_arguments_respects_explicit_eval_steps(monkeypatch, tmp_path):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyLoraConfig:
        def __init__(self, *args, **kwargs):
            pass

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
        def __init__(
            self,
            per_device_train_batch_size=None,
            per_device_eval_batch_size=None,
            gradient_accumulation_steps=None,
            warmup_steps=None,
            max_steps=None,
            num_train_epochs=None,
            learning_rate=None,
            bf16=None,
            bf16_full_eval=None,
            logging_steps=None,
            logging_dir=None,
            output_dir=None,
            optim=None,
            save_strategy=None,
            save_steps=None,
            save_only_model=None,
            ddp_find_unused_parameters=None,
            evaluation_strategy=None,
            eval_strategy=None,
            eval_steps=None,
            weight_decay=None,
            seed=None,
            max_grad_norm=None,
            report_to=None,
            remove_unused_columns=None,
            save_total_limit=None,
            deepspeed=None,
        ):
            self.kwargs = {k: v for k, v in locals().items() if k != "self"}

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

    from reglu.config import RunConfig
    from reglu.methods.reglu.core import build_training_arguments

    config = RunConfig(task="tofu", model_family="llama2-7b")
    config.training.eval_while_train = True
    config.training.eval_steps = 7
    config.training.save_steps = 11
    config.training.use_deepspeed = False

    args = build_training_arguments(config, str(tmp_path / "run"), train_size=4096)
    assert args.kwargs["eval_steps"] == 7
    assert args.kwargs["save_steps"] == 11
    assert args.kwargs["evaluation_strategy"] == "steps"


def test_build_training_arguments_uses_repo_local_ds_config(monkeypatch, tmp_path):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyLoraConfig:
        def __init__(self, *args, **kwargs):
            pass

    peft_mod.LoraConfig = DummyLoraConfig
    peft_mod.get_peft_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    deepspeed_mod = types.ModuleType("deepspeed")
    deepspeed_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

    class DummyTrainerCallback:
        pass

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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

    from reglu.config import RunConfig
    from reglu.methods.reglu.core import build_training_arguments

    config = RunConfig(task="tofu", model_family="llama2-7b")
    config.training.use_deepspeed = True
    args = build_training_arguments(config, str(tmp_path / "run"), train_size=128)
    assert args.kwargs["deepspeed"].endswith("configs/ds_config.json")
    # TOFU 同款整除：1 * 128 // (默认 batch=2 * grad=16) == 4；warmup == max_steps//epochs
    assert args.kwargs["max_steps"] == 4
    assert args.kwargs["warmup_steps"] == 4
