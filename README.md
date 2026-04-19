# ReGLU: Representation-Guided Low-rank Unlearning

> Official implementation of **"Representation-Guided Parameter-Efficient LLM Unlearning"** (**Findings of ACL 2026**).
>
> Zeguan Xiao\*, Lang Mo\*, Yun Chen, Lei Yang, Jiehui Zhao, Lili Yang†, Guanhua Chen†
> (\*Equal contribution; †Corresponding authors)
> Shanghai University of Finance and Economics · Southern University of Science and Technology · Deepexi Technology Co. Ltd.

ReGLU is a lightweight, LoRA-based LLM unlearning framework that exploits the geometry of the representation space. It combines:

- **RILA** (*Representation-guided Initialization for LoRA* — called the "representation-guided LoRA initialization" in the paper) — initialize LoRA `A`, `B` from the top eigenvectors of a `forget` / `retain` feature-covariance contrast, so optimization starts inside a subspace that is maximally forget-informative while keeping retain-direction leakage small.
- **ROL** (*Representation-oriented Orthogonal Loss*) — a regularizer that constrains the LoRA `B` outputs to lie in the orthogonal complement of the retain-set representation subspace, minimizing interference with retain-set performance.
- Two forget objectives: **IHL** (inverse-hinge) and **GD** (gradient difference).

Benchmarks supported out of the box:

| Task  | Model family      | Script                         |
|-------|-------------------|--------------------------------|
| TOFU  | `phi-1.5`, `llama2-7b` | `scripts/tofu_{finetune,forget,eval}.sh` |
| WMDP  | `zephyr-7b-beta`  | `scripts/wmdp_{forget,eval}.sh` |

---

## Repository layout

```
reglu-open/
├── configs/            # YAML configs (TOFU & WMDP; finetune / forget / eval)
├── scripts/            # thin shell wrappers around `reglu <cmd> --config ...`
├── src/reglu/          # Python package (data / methods / trainers / eval)
├── tests/              # unit & smoke tests
├── docs/               # method and migration notes
├── pyproject.toml
└── README.md
```

## Installation

Tested with Python 3.10+, PyTorch ≥ 2.1, CUDA 12.x.

```bash
git clone https://github.com/Jacksooooff/reglu-open.git
cd reglu-open

conda create -n reglu python=3.10 -y
conda activate reglu

pip install -e .
# Optional: multi-GPU with DeepSpeed
pip install -e ".[train]"
# Optional: dev/test deps
pip install -e ".[dev]"
```

WMDP evaluation uses the `lm-eval` harness, which is declared as a dependency in `pyproject.toml`.

## Quick start

### TOFU (`phi-1.5` by default)

```bash
# 1. Finetune on TOFU `full` split (produces checkpoint-last used by forget)
./scripts/tofu_finetune.sh

# 2. Unlearn on forget10 with RegLU (RILA init + ROL regularizer + IHL loss)
./scripts/tofu_forget.sh

# 3. Official TOFU evaluation (MU / FQ + retain / author / world metrics)
./scripts/tofu_eval.sh
```

To switch model family, edit `configs/tofu_*.yaml` and set `model_family: llama2-7b`.

### WMDP (`zephyr-7b-beta`)

```bash
# 1. Unlearn on the `bio` split
./scripts/wmdp_forget.sh

# 2. Evaluate with lm-eval (wmdp_bio + mmlu)
./scripts/wmdp_eval.sh
```

### Multi-GPU

The finetune script also supports `torchrun`:

```bash
TORCHRUN_NPROC_PER_NODE=4 ./scripts/tofu_finetune.sh
```

## Reproducing the paper

All YAMLs in `configs/` ship the paper-reported settings.

| Setting                  | TOFU (forget10)            | WMDP (bio)              |
|--------------------------|----------------------------|-------------------------|
| LoRA `r` / `alpha`       | 32 / 64                    | 32 / 64                 |
| `batch × grad_accum`     | 4 × 8                      | 2 × 16                  |
| `num_epochs`             | 5                          | 5                       |
| `max_steps`              | auto (formula below)       | 100                     |
| `learning_rate`          | 1e-4                       | 1e-5                    |
| `max_length`             | 500                        | 1024                    |
| RILA samples / split     | 256                        | 256                     |
| RILA β                   | 0.7                        | 0.5                     |
| ROL λ / rank             | 0.5 / 128                  | 0.5 / 128               |
| ROL targets / side       | `all_lora` / `B`           | `all_lora` / `B`        |

TOFU training steps follow the standard formula

```
max_steps = num_epochs × |train| / (batch_size × grad_accum × world_size)
```

e.g. for TOFU forget10: `5 × 400 / (4 × 8 × 1) = 62` steps.

## Configuration surface (public v1)

`method`:

- `name`            = `reglu`
- `variant`         = `ihl | gd`
- `init_strategy`   = `default | rila`
- `rila_beta`, `rila_cov_shrink`, `rila_samples_per_split`, `rila_cache_path`, `require_rila_cache`
- `rol_lambda`, `rol_rank`, `rol_targets` (`all_lora | vproj_only`)

`training`: batch / grad-accum / epochs / max_steps / lr / wd / warmup / save_* / seed.

`evaluation`: batch size, max_length, overwrite, ds_size (TOFU).

ReGLU v1 intentionally keeps the public interface small. ROL is implemented only on the LoRA `B` side — historical AB-side branches are not exposed.

## Artifacts per run

Each `reglu finetune | forget | eval` run writes under `runtime.output_dir`:

```
<output_dir>/
├── config.yaml
├── metrics.jsonl
├── metrics.summary.json
├── checkpoints/checkpoint-last/
└── artifacts/rila_cache/…  (forget only, when init_strategy=rila)
```

TOFU eval additionally writes `eval_log.json`, `eval_real_author_wo_options.json`, `eval_real_world_wo_options.json`, `eval_log_forget.json`, `eval_log_aggregated.json`, `aggregate_stat.csv`.
WMDP eval additionally writes `LMEval_EVAL.json`, `LMEval_SUMMARY.json`.

## Tests

```bash
pytest -q
```

Tests that require `torch` will be skipped automatically if PyTorch is not installed.

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{xiao2026reglu,
  title     = {Representation-Guided Parameter-Efficient {LLM} Unlearning},
  author    = {Xiao, Zeguan and Mo, Lang and Chen, Yun and Yang, Lei and Zhao, Jiehui and Yang, Lili and Chen, Guanhua},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```

> The BibTeX will be updated with the final ACL Anthology key (pages / DOI) once the proceedings are published.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Acknowledgements

We build on the TOFU benchmark, WMDP, `lm-evaluation-harness`, `peft`, and `transformers`.
