from __future__ import annotations

import json
import sys

import pytest
import yaml


pytest.importorskip("torch")

from reglu.cli import main


def test_cli_dry_run_finetune_writes_summary(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "tofu",
                "model_family": "llama2-7b",
                "data": {"path": "locuslab/TOFU", "split": "full"},
                "runtime": {"output_dir": str(output_dir), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["reglu", "finetune", "--config", str(config_path)])
    main()
    summary = json.loads((output_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    out = capsys.readouterr().out
    assert '"command": "finetune"' in out


def test_cli_dry_run_wmdp_eval_writes_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": "wmdp",
                "model_family": "zephyr-7b-beta",
                "data": {"split": "bio"},
                "runtime": {"output_dir": str(output_dir), "dry_run": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["reglu", "eval", "--config", str(config_path)])
    main()
    summary = json.loads((output_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "wmdp"
    assert summary["status"] == "dry_run"
    assert summary["command"] == "eval"
