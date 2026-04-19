#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
CFG="${ROOT_DIR}/configs/tofu_eval.yaml"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON_BIN:-python}" -m reglu.cli eval --config "${CFG}"
