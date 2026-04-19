# RegLU Method

RegLU v1 keeps a single public method name, `RegLU`.

The public config surface is:

- `method.name = reglu`
- `method.variant = ihl | gd`
- `method.init_strategy = rila`
- `method.rol_lambda`
- `method.rol_rank`
- `method.rol_targets = all_lora | vproj_only`
- `method.rila_cache_path`

Internally, v1 implements:

- paired forget/retain batches
- IHL or GD variant
- optional `ROL` regularization over LoRA `B` matrices
- `RILA` cache-backed initialization and `ROL` basis loading

Paper-aligned scope:

- `ROL` is implemented only on the LoRA `B` side
- non-paper branches such as `AB`-side regularization are intentionally removed from the public interface

The main goal is a clean, reproducible public API rather than preserving every historical experimental branch.
