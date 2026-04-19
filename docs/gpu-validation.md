# GPU Validation Guide

这份文档的目标不是“跑通一次”，而是把 `reglu-open` 和旧仓库放在同一套 GPU 环境下，做一轮可以留档、可以复查、可以机械执行的等价性验证。

适用范围：

- TOFU / `llama2-7b`: `finetune -> forget -> eval`
- WMDP / `zephyr-7b-beta`: `forget -> eval`

## 验收原则

必须同时满足下面三层：

1. 环境一致
2. 配置语义一致
3. 产物与结果一致

如果只满足第 3 层但前两层不一致，这次验证不算通过，因为你无法证明差异来自实现，而不是环境或配置漂移。

## 一、GPU 环境要求

建议旧仓库和 `reglu-open` 在同一台机器、同一组 GPU 上完成验证。

最少记录以下信息：

- `nvidia-smi`
- `python --version`
- `torch.__version__`
- `transformers.__version__`
- `peft.__version__`
- `datasets.__version__`
- `lm_eval.__version__`
- `deepspeed.__version__`
- `CUDA_VISIBLE_DEVICES`
- `WORLD_SIZE`

建议把这些信息保存到一个文本文件，例如：

```bash
nvidia-smi > /path/to/validation/env.txt
python - <<'PY' >> /path/to/validation/env.txt
import torch, transformers, peft, datasets
print("python")
import sys; print(sys.version)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
try:
    import lm_eval
    print("lm_eval", getattr(lm_eval, "__version__", "unknown"))
except Exception:
    print("lm_eval", "not-importable")
try:
    import deepspeed
    print("deepspeed", deepspeed.__version__)
except Exception:
    print("deepspeed", "not-importable")
PY
```

## 二、运行前固定项

旧版和新版必须固定这些条件：

- 同一 `model_family`
- 同一底座模型或同一 finetune checkpoint
- 同一数据 split
- 同一 seed
- 同一 GPU 数量
- 同一 `batch_size`
- 同一 `gradient_accumulation_steps`
- 同一 `learning_rate`
- 同一 `num_epochs` / `max_steps`
- 同一 LoRA rank / alpha / dropout
- 同一遗忘损失变体
- 同一 eval 数据规模，例如 TOFU 的 `ds_size=300`

推荐验证目录布局：

```text
/abs/path/to/gpu-validation/
  env.txt
  legacy/
    tofu-finetune/
    tofu-forget/
    tofu-eval/
    wmdp-forget/
    wmdp-eval/
  reglu/
    tofu-finetune/
    tofu-forget/
    tofu-eval/
    wmdp-forget/
    wmdp-eval/
  reports/
    gpu-validation-report.json
```

## 三、推荐验证矩阵

最低要求做 4 组：

1. TOFU finetune
2. TOFU forget
3. TOFU eval
4. WMDP eval

如果要做完整闭环，建议做 5 组：

1. TOFU finetune
2. TOFU forget
3. TOFU eval
4. WMDP forget
5. WMDP eval

## 四、每一组要检查什么

### 1. TOFU finetune

通过标准：

- 旧版和新版都成功结束，没有 OOM、没有 fallback 到错误模型
- 训练配置语义一致
- 新版必须产出：
  - `config.yaml`
  - `metrics.summary.json`
  - `checkpoints/checkpoint-last`
  - 根目录 merged model
- 如果你能导出旧版 loss trace，则要求：
  - step 数一致
  - final train loss 绝对误差 `<= 1e-3`
  - loss trace 的主要趋势一致

说明：

- finetune 阶段最重要的是确认它确实生成了后续 forget/eval 要使用的底座。
- 如果旧版没有标准化 summary 文件，不强制和 `metrics.summary.json` 做逐字段比对，但至少要记录 checkpoint 路径和训练日志。

### 2. TOFU forget

通过标准：

- 旧版和新版都成功结束
- LoRA / RILA / ROL 相关超参一一对应
- 新版必须产出：
  - `config.yaml`
  - `metrics.summary.json`
  - `checkpoints/checkpoint-last`
  - `efficiency_log.csv`
- 如果能导出旧版 forget loss trace，要求：
  - step 数一致
  - final forget loss 绝对误差 `<= 1e-3`
  - retain / total loss 没有量级漂移

### 3. TOFU eval

这是最关键的数值验收点。

通过标准：

- 两边目录都必须有：
  - `eval_log.json`
  - `eval_real_author_wo_options.json`
  - `eval_real_world_wo_options.json`
  - `eval_log_forget.json`
  - `eval_log_aggregated.json`
  - `aggregate_stat.csv`
- `aggregate_stat.csv` 的数值键集合一致
- 下列指标逐项满足：
  - 绝对误差 `<= 5e-4`
  - 相对误差 `<= 1e-3`

重点指标：

- `Model Utility`
- `Forget Quality`
- `KS Test Forget`
- `KS Test PVal Forget`
- `Retain Probability`
- `Retain ROUGE`
- `Retain Truth Ratio`
- `Real Authors Probability`
- `Real World Probability`
- `Forget Truth Ratio`

### 4. WMDP forget

通过标准：

- 旧版和新版都成功结束
- 新版必须产出：
  - `config.yaml`
  - `metrics.summary.json`
  - `checkpoints/checkpoint-last`
  - `efficiency_log.csv`

WMDP forget 的最终数值验收以随后的 `lm_eval` 结果为准，不建议只看训练 loss。

### 5. WMDP eval

通过标准：

- 两边目录都必须有：
  - `LMEval_EVAL.json`
  - `LMEval_SUMMARY.json`
- `LMEval_EVAL.json` 的任务集合一致
- `LMEval_SUMMARY.json` 的数值键集合一致
- 每个数值键满足：
  - 绝对误差 `<= 5e-4`
  - 相对误差 `<= 1e-3`

重点关注：

- `wmdp_bio/acc` 或 `wmdp_cyber/acc`
- `wmdp_bio/acc_stderr` 或 `wmdp_cyber/acc_stderr`
- `mmlu/acc`
- `mmlu/acc_stderr`

## 五、推荐执行顺序

### A. 跑旧版

按你已经确认过的旧版主线命令，分别把产物落到：

- `/abs/path/to/gpu-validation/legacy/tofu-finetune`
- `/abs/path/to/gpu-validation/legacy/tofu-forget`
- `/abs/path/to/gpu-validation/legacy/tofu-eval`
- `/abs/path/to/gpu-validation/legacy/wmdp-forget`
- `/abs/path/to/gpu-validation/legacy/wmdp-eval`

如果旧版脚本不能直接改输出目录，至少把最终产物拷贝到这些固定目录。

### B. 跑 `reglu-open`

在 `reglu-open` 下把示例配置复制一份到临时目录，再把 `runtime.output_dir` 改到验证目录。

示例：

```bash
cd /path/to/reglu-open_v3

PYTHONPATH=src python -m reglu.cli finetune --config configs/tofu_finetune.yaml
PYTHONPATH=src python -m reglu.cli forget --config configs/tofu_forget.yaml
PYTHONPATH=src python -m reglu.cli eval --config configs/tofu_eval.yaml

PYTHONPATH=src python -m reglu.cli forget --config configs/wmdp_forget.yaml
PYTHONPATH=src python -m reglu.cli eval --config configs/wmdp_eval.yaml
```

推荐把 `runtime.output_dir` 改成：

- `/abs/path/to/gpu-validation/reglu/tofu-finetune`
- `/abs/path/to/gpu-validation/reglu/tofu-forget`
- `/abs/path/to/gpu-validation/reglu/tofu-eval`
- `/abs/path/to/gpu-validation/reglu/wmdp-forget`
- `/abs/path/to/gpu-validation/reglu/wmdp-eval`

## 六、机械校验脚本

仓库里已经提供脚本：

- [validate_gpu_runs.py](scripts/validate_gpu_runs.py)
- [gpu-validation.example.yaml](docs/examples/gpu-validation.example.yaml)

脚本支持的检查类型：

- `run_dir`
- `tofu_eval`
- `wmdp_eval`
- `json`
- `csv_trace`

### 1. 准备 manifest

复制示例文件并改成你自己的绝对路径：

```bash
cp docs/examples/gpu-validation.example.yaml /abs/path/to/gpu-validation/manifest.yaml
```

### 2. 运行校验

```bash
cd /path/to/reglu-open_v3
PYTHONPATH=src python scripts/validate_gpu_runs.py \
  --manifest /abs/path/to/gpu-validation/manifest.yaml \
  --report /abs/path/to/gpu-validation/reports/gpu-validation-report.json
```

也可以直接用模块入口：

```bash
cd /path/to/reglu-open_v3
PYTHONPATH=src python -m reglu.validation \
  --manifest /abs/path/to/gpu-validation/manifest.yaml \
  --report /abs/path/to/gpu-validation/reports/gpu-validation-report.json
```

### 3. 脚本返回码

- 返回 `0`: 所有检查通过
- 返回 `1`: 至少一项失败

## 七、manifest 说明

### `tofu_eval`

用于对比 TOFU 最终评测目录。

示例：

```yaml
- name: tofu_eval
  kind: tofu_eval
  baseline_dir: /abs/path/to/legacy/tofu_eval_dir
  candidate_dir: /abs/path/to/reglu-open/outputs/tofu-eval
```

脚本会自动比较：

- 必要文件是否齐全
- 每个任务文件的均值摘要
- `aggregate_stat.csv`

### `wmdp_eval`

用于对比 WMDP 的 `lm_eval` 结果目录。

示例：

```yaml
- name: wmdp_eval
  kind: wmdp_eval
  baseline_dir: /abs/path/to/legacy/wmdp_eval_dir
  candidate_dir: /abs/path/to/reglu-open/outputs/wmdp-eval
```

脚本会自动比较：

- `LMEval_EVAL.json` 的任务集合
- `LMEval_SUMMARY.json` 的数值键和值

### `run_dir`

用于做布局检查，必要时也能加 summary / trace 比对。

示例：

```yaml
- name: tofu_forget_layout
  kind: run_dir
  baseline_dir: /abs/path/to/legacy/tofu_forget_run
  candidate_dir: /abs/path/to/reglu-open/outputs/tofu-forget
  required_files_candidate:
    - config.yaml
    - metrics.summary.json
    - checkpoints/checkpoint-last
    - efficiency_log.csv
```

### `csv_trace`

用于 loss trace 对比。前提是你能把旧版 trace 导出成 CSV。

示例：

```yaml
- name: optional_forget_loss_trace
  kind: csv_trace
  baseline: /abs/path/to/legacy/forget_loss_trace.csv
  candidate: /abs/path/to/reglu-open/outputs/tofu-forget/efficiency_log.csv
  step_key: step
  value_key: forget_loss
  atol: 1.0e-3
  rtol: 5.0e-3
```

## 八、推荐阈值

默认阈值：

- `atol = 5e-4`
- `rtol = 1e-3`

建议：

- TOFU / WMDP 最终评测继续用默认阈值
- 训练曲线可以放宽到：
  - `atol = 1e-3`
  - `rtol = 5e-3`

如果你发现只在最后几位浮动，而趋势和最终 summary 都一致，不要急着判定失败；先回看环境记录和 seed 是否真的完全一致。

## 九、失败时怎么排查

先看失败属于哪一类：

1. 缺文件
2. 键集合不一致
3. 数值超阈值

排查顺序必须固定：

1. 先核对环境记录
2. 再核对 config 实参
3. 再核对底座模型路径和 tokenizer 路径
4. 再核对数据 split 和 subset
5. 最后才看实现差异

常见原因：

- 旧版和新版实际读到的不是同一个底座
- TOFU eval 的 retain_result 用错了
- WMDP 评测没有同时包含 `mmlu`
- batch size / gradient accumulation 不一致
- 训练步数一致，但 warmup 语义不一致
- 旧版或新版使用了不同的 checkpoint 做评测

## 十、最低交付标准

如果你要把这轮 GPU 验证作为正式验收，至少需要存档这些文件：

- `env.txt`
- 旧版和新版的全部最终产物目录
- manifest 文件
- `gpu-validation-report.json`
- 一份人工结论，明确写出：
  - 哪些 check 通过
  - 哪些 check 失败
  - 失败是否可解释
  - 是否可以宣称“行为等价”

只有当 `tofu_eval` 和 `wmdp_eval` 都通过，并且训练阶段没有发现控制流或布局异常，才建议把 `reglu-open` 标记为 GPU 侧验收通过。
