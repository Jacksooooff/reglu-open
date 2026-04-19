from __future__ import annotations

import pytest
import yaml

from reglu.config import load_run_config


def test_load_config_and_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    config = load_run_config(config_path, "forget")
    assert config.task == "tofu"
    assert config.model_family == "llama2-7b"
    assert config.method.name == "reglu"
    assert config.method.variant == "ihl"
    assert config.method.init_strategy == "rila"
    assert config.method.rol_lambda == 0.0
    assert config.method.rol_rank == 128
    assert config.runtime.dry_run is True


def test_invalid_wmdp_finetune_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"path": "dummy", "split": "bio"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_run_config(config_path, "finetune")
    except ValueError as exc:
        assert "only supports `finetune`" in str(exc)
    else:
        raise AssertionError("Expected config validation failure for WMDP finetune.")


def test_finetune_rejects_evaluation_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "full"},
                "evaluation": {"batch_size": 8},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finetune does not use evaluation"):
        load_run_config(config_path, "finetune")


def test_finetune_accepts_phi_model_family(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "phi-1.5",
                "data": {"path": "locuslab/TOFU", "split": "full"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    config = load_run_config(config_path, "finetune")
    assert config.model_family == "phi-1.5"


def test_unknown_legacy_method_keys_are_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10"},
                "method": {
                    "legacy_method_key": 1,
                },
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_run_config(config_path, "forget")
    except ValueError as exc:
        assert "Unknown config key" in str(exc)
    else:
        raise AssertionError("Expected legacy method-key rejection.")


def test_wmdp_eval_without_dataset_path_is_allowed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    config = load_run_config(config_path, "eval")
    assert config.task == "wmdp"
    assert config.evaluation.dataset_path is None


def test_eval_rejects_training_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "training": {"batch_size": 8},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="eval does not use training"):
        load_run_config(config_path, "eval")


def test_wmdp_eval_dataset_path_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "evaluation": {"dataset_path": str(tmp_path / "eval.jsonl")},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_path"):
        load_run_config(config_path, "eval")


def test_wmdp_eval_ds_size_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "evaluation": {"ds_size": 16},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation.ds_size"):
        load_run_config(config_path, "eval")


def test_nonempty_evaluation_metrics_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10"},
                "evaluation": {"metrics": ["rouge"]},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation.metrics"):
        load_run_config(config_path, "eval")


def test_runtime_device_non_auto_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True, "device": "cuda"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime.device"):
        load_run_config(config_path, "eval")


def test_tofu_eval_tasks_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10"},
                "evaluation": {"tasks": ["retain"]},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation.tasks"):
        load_run_config(config_path, "eval")


def test_tofu_eval_dataset_path_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10"},
                "evaluation": {"dataset_path": str(tmp_path / "eval.json")},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_path"):
        load_run_config(config_path, "eval")


def test_eval_rejects_data_max_length(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "forget10", "max_length": 500},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.max_length"):
        load_run_config(config_path, "eval")


def test_wmdp_eval_data_path_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"path": "./data/wmdp-corpora", "split": "bio"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.path"):
        load_run_config(config_path, "eval")


def test_tofu_eval_subset_indices_file_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {
                    "path": "locuslab/TOFU",
                    "split": "forget10",
                    "subset_indices_file": str(tmp_path / "subset.txt"),
                },
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unlearn_eval_subset_indices_file"):
        load_run_config(config_path, "eval")


def test_wmdp_eval_question_key_override_is_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio", "question_key": "prompt"},
                "runtime": {"output_dir": str(tmp_path / "out"), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="key overrides"):
        load_run_config(config_path, "eval")
