# State

`reglu-open` 已经完成当前阶段的代码层面收口，公开接口和主线行为如下：

## 已完成

- 公开 CLI 统一为 `reglu finetune / reglu forget / reglu eval`
- 公开命名统一为论文命名：`RegLU / RILA / ROL`
- WMDP 评测统一走 `lm_eval`，并写出 `LMEval_EVAL.json` / `LMEval_SUMMARY.json`
- forget trainer 的 `eval_only` / `eval_while_train` 控制流已经接到公开 `reglu eval` 路径
- TOFU 评测会写出 legacy 风格的 `eval_log_aggregated.json` 和 `aggregate_stat.csv`
- `training.eval_steps` 已经接入训练参数构造，不再是无效配置
- 非公共或未实现的配置项不再静默忽略：
  - `evaluation.metrics`
  - `runtime.device != auto`
  - TOFU eval 的 `evaluation.tasks` / `evaluation.dataset_path` / `evaluation.dataset_split`
  - WMDP eval 的 `data.path` / `evaluation.dataset_path` / `evaluation.ds_size`

## 当前保证

- 代码路径已经通过仓库内的静态与单元测试
- 评测输出文件名、公开命名、主线配置和脚本已经收敛到开源仓库形态
- 不再依赖旧 repo 的私有脚本、路径约定或外部评测仓库耦合

## 仍需外部验证

以下部分已经无法仅靠本地代码审查完成，必须在有依赖和 GPU 的环境中验证：

- 旧版与新版训练 loss 轨迹是否逐步贴合
- 同配置下 TOFU summary 是否数值一致
- 同配置下 WMDP `lm_eval` summary 是否数值一致
- 保存出的 checkpoint 在真实环境中的可恢复性和复现性

换句话说，代码层面的迁移和接口收口已经完成；剩下的是环境依赖齐全后的结果对比验证。
