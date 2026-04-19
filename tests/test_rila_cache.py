from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reglu.methods.reglu.cache import load_rol_bases


def test_load_rol_bases_reads_b_side_basis(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "layers": {
                "layer0": {
                    "Qr_retain": torch.randn(8, 4),
                }
            }
        },
        cache_path,
    )
    payload = load_rol_bases(cache_path, rol_rank=2)
    assert "layer0" in payload
    assert payload["layer0"].shape == (8, 2)


def test_load_rol_bases_requires_enough_basis_dimension(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "layers": {
                "layer0": {
                    "Qr_retain": torch.randn(8, 4),
                }
            }
        },
        cache_path,
    )
    with pytest.raises(ValueError, match="rol_rank=5"):
        load_rol_bases(cache_path, rol_rank=5)


def test_load_rol_bases_rejects_invalid_metadata(tmp_path):
    cache_path = tmp_path / "cache.pt"
    torch.save(
        {
            "metadata": "invalid",
            "layers": {
                "layer0": {
                    "Qr_retain": torch.randn(8, 4),
                }
            },
        },
        cache_path,
    )
    with pytest.raises(ValueError, match="metadata"):
        load_rol_bases(cache_path, rol_rank=2)
