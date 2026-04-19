from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from reglu.config import RunConfig
from reglu.data.common import custom_data_collator_unlearn
from reglu.methods.reglu.core import maybe_apply_rila_cache
from reglu.models import get_model_spec


def _get_target_lora_modules(model, target_tags: list[str]) -> dict[str, torch.nn.Module]:
    target_modules: dict[str, torch.nn.Module] = {}
    for mod_name, mod in model.named_modules():
        has_lora = hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base_layer")
        if not has_lora:
            continue
        if not any(tag in mod_name for tag in target_tags):
            continue
        try:
            _ = mod.lora_A["default"].weight
            _ = mod.lora_B["default"].weight
        except Exception:
            continue
        target_modules[mod_name] = mod
    return target_modules


def _resolve_rila_cache_path(config: RunConfig, output_dir: str, target_tags: list[str]) -> Path:
    if config.method.rila_cache_path:
        return Path(config.method.rila_cache_path).expanduser()
    alpha = int(config.lora.alpha)
    rank = int(config.lora.r)
    scaling = float(alpha) / float(rank) if rank > 0 else 1.0
    metadata = {
        "version": 1,
        "model_family": config.model_family,
        "model_path": str(Path(config.model.model_path or get_model_spec(config.model_family).hf_key).expanduser()),
        "split": str(config.data.split),
        "targets": sorted(target_tags),
        "rank": rank,
        "alpha": alpha,
        "scaling_s": scaling,
        "beta": float(config.method.rila_beta),
        "cov_shrink": float(config.method.rila_cov_shrink),
        "samples": int(config.method.rila_samples_per_split),
    }
    cache_key = hashlib.md5(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    readable = (
        f"rila_{config.model_family}"
        f"_tgt-{','.join(sorted(target_tags))}"
        f"_r{rank}_a{alpha}_s{int(round(scaling, 0))}"
        f"_beta{config.method.rila_beta}_n{config.method.rila_samples_per_split}"
        f"_sh{config.method.rila_cov_shrink}"
        f"_{config.data.split}_{cache_key}.pt"
    ).replace("/", "-").replace(" ", "")
    cache_dir = Path(output_dir) / "artifacts" / "rila_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / readable


def _cache_metadata(config: RunConfig, target_tags: list[str]) -> dict[str, object]:
    alpha = int(config.lora.alpha)
    rank = int(config.lora.r)
    scaling = float(alpha) / float(rank) if rank > 0 else 1.0
    return {
        "version": 1,
        "model_family": config.model_family,
        "model_path": str(Path(config.model.model_path or get_model_spec(config.model_family).hf_key).expanduser()),
        "split": str(config.data.split),
        "targets": sorted(target_tags),
        "rank": rank,
        "alpha": alpha,
        "scaling_s": scaling,
        "beta": float(config.method.rila_beta),
        "cov_shrink": float(config.method.rila_cov_shrink),
        "samples": int(config.method.rila_samples_per_split),
    }


def initialize_rila(
    model,
    config: RunConfig,
    dataset,
    output_dir: str,
    strict_cache: bool,
) -> tuple[bool, dict[str, torch.Tensor] | None, Path | None]:
    model_spec = get_model_spec(config.model_family)
    target_tags = model_spec.default_lora_targets[config.lora.targets]
    target_modules = _get_target_lora_modules(model, target_tags)
    if not target_modules:
        return False, None, None

    cache_path = _resolve_rila_cache_path(config, output_dir, target_tags)
    cache_meta = _cache_metadata(config, target_tags)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") == cache_meta:
            applied = maybe_apply_rila_cache(model, str(cache_path), mode="all")
            if applied:
                rol_bases: dict[str, torch.Tensor] = {}
                for name, entry in payload.get("layers", {}).items():
                    q_retain = entry.get("Qr_retain")
                    if isinstance(q_retain, torch.Tensor) and q_retain.shape[1] >= config.method.rol_rank:
                        rol_bases[name] = q_retain[:, : config.method.rol_rank].contiguous()
                return True, rol_bases or None, cache_path
        elif strict_cache:
            raise ValueError(f"Existing RILA cache metadata mismatch: {cache_path}")
    elif strict_cache and config.method.rila_cache_path:
        raise FileNotFoundError(f"Required RILA cache not found: {cache_path}")

    rank = int(config.lora.r)
    if rank <= 0:
        return False, None, cache_path

    import torch.distributed as dist

    is_dist = dist.is_available() and dist.is_initialized()
    world_rank = dist.get_rank() if is_dist else 0
    # 与 TOFU 旧仓初始化流程一致：若模型仍在 CPU，rank0 临时搬到 GPU 做采集前向，否则整段会在 CPU 上极慢。
    orig_device = next(model.parameters()).device
    init_device = orig_device
    if world_rank == 0 and torch.cuda.is_available():
        try:
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                init_device = torch.device(f"cuda:{int(local_rank_env)}")
            else:
                init_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            if init_device.type == "cuda" and orig_device.type != "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                model.to(init_device)
                torch.cuda.empty_cache()
                print(f"[RILA] moved model to {init_device} for init forward pass (was on {orig_device})")
        except Exception as exc:  # pragma: no cover - env-specific
            print(f"[RILA] failed to move model to GPU for init: {exc}")
            init_device = orig_device
    compute_device = init_device if getattr(init_device, "type", "cpu") == "cuda" else torch.device("cpu")

    samples_target = int(config.method.rila_samples_per_split)
    beta = float(config.method.rila_beta)
    eps = 1e-4

    forget_feats = {name: [] for name in target_modules}
    retain_feats = {name: [] for name in target_modules}
    sample_counts = {"forget": 0, "retain": 0}

    if world_rank == 0:
        dl_bs = max(1, min(8, int(config.training.batch_size)))
        loader = DataLoader(
            dataset,
            batch_size=dl_bs,
            shuffle=False,
            collate_fn=custom_data_collator_unlearn,
            pin_memory=torch.cuda.is_available(),
        )

        collection_mode = {"split": "forget"}

        def make_hook(key: str) -> Callable:
            def _hook(_mod, _inp, out):
                out_t = out[0] if isinstance(out, (tuple, list)) else out
                if out_t is None:
                    return
                if out_t.dim() == 3:
                    pooled = out_t.sum(dim=1)
                elif out_t.dim() == 2:
                    pooled = out_t
                else:
                    return
                feats = pooled.detach().to("cpu").to(torch.float64)
                if collection_mode["split"] == "forget":
                    forget_feats[key].append(feats)
                else:
                    retain_feats[key].append(feats)
            return _hook

        handles = [
            module.base_layer.register_forward_hook(make_hook(name))
            for name, module in target_modules.items()
        ]
        model_was_training = model.training
        model.eval()
        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:  # pragma: no cover
            _tqdm = None
        # 按样本数显示进度：目标 = samples_target（forget/retain 各自独立凑到 256）
        # 每个 step 会同时对 forget + retain 各跑一次前向；显示的是 min(forget_done, retain_done)。
        pbar = (
            _tqdm(total=samples_target, desc="RILA init", unit="sample", leave=True)
            if _tqdm
            else None
        )
        use_amp = init_device.type == "cuda"
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            for forget_batch, retain_batch in loader:
                f_ids, _, f_attn = forget_batch
                r_ids, _, r_attn = retain_batch
                if sample_counts["forget"] < samples_target:
                    collection_mode["split"] = "forget"
                    with amp_ctx:
                        _ = model(
                            f_ids.to(init_device, non_blocking=True),
                            attention_mask=f_attn.to(init_device, non_blocking=True),
                        )
                    sample_counts["forget"] += int(f_ids.shape[0])
                if sample_counts["retain"] < samples_target:
                    collection_mode["split"] = "retain"
                    with amp_ctx:
                        _ = model(
                            r_ids.to(init_device, non_blocking=True),
                            attention_mask=r_attn.to(init_device, non_blocking=True),
                        )
                    sample_counts["retain"] += int(r_ids.shape[0])
                if pbar is not None:
                    done = min(sample_counts["forget"], sample_counts["retain"])
                    pbar.n = min(done, samples_target)
                    pbar.refresh()
                if sample_counts["forget"] >= samples_target and sample_counts["retain"] >= samples_target:
                    break
        if pbar is not None:
            pbar.close()
        for handle in handles:
            handle.remove()
        if model_was_training:
            model.train()

    def _stack_first_k(chunks: list[torch.Tensor]) -> torch.Tensor | None:
        if not chunks:
            return None
        cat = torch.cat(chunks, dim=0)
        if cat.shape[0] > samples_target:
            cat = cat[:samples_target]
        return cat

    any_failure = False
    saved_layers: dict[str, dict[str, torch.Tensor]] = {}
    rol_bases: dict[str, torch.Tensor] = {}

    for name, module in target_modules.items():
        if world_rank != 0:
            continue
        h_forget = _stack_first_k(forget_feats[name])
        h_retain = _stack_first_k(retain_feats[name])
        if h_forget is None or h_retain is None or h_forget.numel() == 0 or h_retain.numel() == 0:
            any_failure = True
            continue
        h_forget = h_forget.to(device=compute_device)
        h_retain = h_retain.to(device=compute_device)
        nf, d_out = h_forget.shape
        nr, d_out2 = h_retain.shape
        if d_out != d_out2:
            any_failure = True
            continue
        cf = (h_forget.T @ h_forget) / max(1, nf)
        cr = (h_retain.T @ h_retain) / max(1, nr)
        eye = torch.eye(d_out, dtype=torch.float64, device=compute_device)
        cf = cf + eps * eye
        cr = cr + eps * eye
        delta = (1.0 - beta) * cf - beta * cr
        evals, evecs = torch.linalg.eigh(delta)
        qr = evecs[:, -rank:]

        k_basis = min(int(config.method.rol_rank), d_out)
        _, cr_evecs = torch.linalg.eigh(cr)
        q_retain = cr_evecs[:, -k_basis:]
        q_retain_cpu = q_retain.detach().cpu().to(torch.float32)

        w0 = module.base_layer.weight.detach().to(torch.float64).to(device=compute_device)
        if qr.shape[0] != w0.shape[0]:
            any_failure = True
            continue
        a_init = qr.T @ w0
        b_init = qr
        dtype = module.base_layer.weight.dtype
        device = module.base_layer.weight.device
        scaling = float(getattr(module, "scaling", {}).get("default", 1.0))
        updated_w = w0 - scaling * (b_init @ a_init)
        module.base_layer.weight.data = updated_w.to(dtype=dtype, device=device).contiguous()
        module.lora_A["default"].weight.data = a_init.to(dtype=dtype, device=device).contiguous()
        module.lora_B["default"].weight.data = b_init.to(dtype=dtype, device=device).contiguous()
        rol_bases[name] = q_retain_cpu
        saved_layers[name] = {
            "A": a_init.to(torch.float32).cpu(),
            "B": b_init.to(torch.float32).cpu(),
            "W": updated_w.to(torch.float32).cpu(),
            "Qr_retain": q_retain_cpu,
            "top_eigenvalues": evals[-min(rank, 8) :].detach().cpu().to(torch.float32),
        }

    if is_dist:
        for module in target_modules.values():
            dist.broadcast(module.lora_A["default"].weight.data, src=0)
            dist.broadcast(module.lora_B["default"].weight.data, src=0)

    if world_rank == 0 and saved_layers and not any_failure:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": cache_meta, "layers": saved_layers}, cache_path)

    if any_failure:
        return False, None, cache_path
    return True, rol_bases, cache_path
