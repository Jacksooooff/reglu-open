from __future__ import annotations

from pathlib import Path

import torch


def load_rol_bases(
    cache_path: str | Path,
    rol_rank: int,
) -> dict[str, torch.Tensor]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"RILA cache not found: {cache_path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    layers = payload.get("layers", {})
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"Invalid RILA cache format: {cache_path}")
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"Invalid RILA cache metadata format: {cache_path}")
    result: dict[str, torch.Tensor] = {}
    for name, entry in layers.items():
        q_retain_b = entry.get("Qr_retain")
        if q_retain_b is None:
            raise ValueError(f"Layer '{name}' is missing Qr_retain.")
        if q_retain_b.shape[1] < rol_rank:
            raise ValueError(
                f"Layer '{name}' has Qr_retain dim {q_retain_b.shape[1]} < rol_rank={rol_rank}."
            )
        result[name] = q_retain_b[:, :rol_rank].contiguous()
    return result
