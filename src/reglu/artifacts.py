from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from reglu.config import normalize_public_command


def ensure_run_layout(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    artifacts = root / "artifacts"
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    for path in (root, artifacts, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "artifacts": artifacts,
        "checkpoints": checkpoints,
        "logs": logs,
        "config": root / "config.yaml",
        "metrics": root / "metrics.jsonl",
        "summary": root / "metrics.summary.json",
        "rila_cache": artifacts / "rila_cache.pt",
    }


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(v) for v in value]
    return value


def write_config_snapshot(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain_data(payload), handle, sort_keys=False, allow_unicode=True)


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    normalized_payload = _to_plain_data(payload)
    if isinstance(normalized_payload, dict) and isinstance(
        normalized_payload.get("command"), str
    ):
        normalized_payload["command"] = normalize_public_command(
            normalized_payload["command"]
        )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(normalized_payload, handle, indent=2, ensure_ascii=False)
