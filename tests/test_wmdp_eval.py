from __future__ import annotations

import json
import importlib.machinery
import sys
import types

import pytest


def test_wmdp_eval_uses_lm_eval_and_writes_expected_files(tmp_path, monkeypatch):
    peft_mod = types.ModuleType("peft")
    peft_mod.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class DummyPeftModel:
        @classmethod
        def from_pretrained(cls, model, model_path):
            return model

    peft_mod.PeftModel = DummyPeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_mod)

    from reglu.config import RunConfig
    from reglu.eval.wmdp import run_wmdp_eval

    config = RunConfig(task="wmdp", model_family="zephyr-7b-beta")
    config.data.split = "bio"
    config.runtime.output_dir = str(tmp_path / "wmdp-eval")

    class DummyModel:
        def eval(self):
            return self

    class DummyTokenizer:
        pass

    monkeypatch.setattr("reglu.eval.wmdp.load_tokenizer", lambda _: DummyTokenizer())
    monkeypatch.setattr("reglu.eval.wmdp._load_eval_model", lambda _: DummyModel())

    lm_eval_mod = types.ModuleType("lm_eval")
    tasks_mod = types.ModuleType("lm_eval.tasks")
    hf_mod = types.ModuleType("lm_eval.models.huggingface")

    def simple_evaluate(**kwargs):
        assert kwargs["tasks"] in (["wmdp_bio"], ["mmlu"])
        task_name = kwargs["tasks"][0]
        return {
            "results": {
                task_name: {
                    "acc,none": 0.25 if task_name == "wmdp_bio" else 0.5,
                    "acc_stderr,none": 0.01 if task_name == "wmdp_bio" else 0.02,
                    "alias": task_name,
                }
            },
            "samples": {
                task_name: {
                    task_name: [
                        {
                            "arguments": [["question", "A", "B"]],
                            "resps": [["A", 0.0]],
                        }
                    ]
                }
            },
        }

    class DummyTaskManager:
        def __init__(self):
            self.all_groups = set()

    class DummyHFLM:
        def __init__(self, pretrained=None, tokenizer=None, batch_size=None, max_length=None):
            self.pretrained = pretrained
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_length = max_length

    lm_eval_mod.simple_evaluate = simple_evaluate
    tasks_mod.TaskManager = DummyTaskManager
    hf_mod.HFLM = DummyHFLM

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_mod)

    summary = run_wmdp_eval(config)

    eval_log = tmp_path / "wmdp-eval" / "LMEval_EVAL.json"
    eval_summary = tmp_path / "wmdp-eval" / "LMEval_SUMMARY.json"
    assert eval_log.is_file()
    assert eval_summary.is_file()

    payload = json.loads(eval_log.read_text(encoding="utf-8"))
    aggregate = json.loads(eval_summary.read_text(encoding="utf-8"))
    assert "wmdp_bio" in payload
    assert "mmlu" in payload
    assert aggregate["wmdp_bio/acc"] == pytest.approx(0.25)
    assert aggregate["wmdp_bio/acc_stderr"] == pytest.approx(0.01)
    assert aggregate["mmlu/acc"] == pytest.approx(0.5)
    assert summary["wmdp_bio/acc"] == pytest.approx(0.25)
    assert summary["tasks"] == ["wmdp_bio", "mmlu"]
