# Migration Notes

Old entrypoints are intentionally not preserved as public interfaces.

## Old -> New

- `TOFU/run_finetune.sh` -> `reglu finetune --config configs/tofu_finetune.yaml`
- 旧 TOFU 遗忘训练脚本 -> `reglu forget --config configs/tofu_forget.yaml`
- 旧 WMDP 遗忘训练脚本 -> `reglu forget --config configs/wmdp_forget.yaml`
- 旧 TOFU 评测脚本 -> `reglu eval --config configs/tofu_eval.yaml`
- 旧 WMDP 评测脚本 -> `reglu eval --config configs/wmdp_eval.yaml`

## Intentional removals

- sweep scripts
- path-hardcoded launchers
- external evaluation repo coupling
- historical experiment directories and aggregation scripts
