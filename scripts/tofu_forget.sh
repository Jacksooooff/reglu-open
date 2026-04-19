#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
CFG="${ROOT_DIR}/configs/tofu_forget.yaml"
N="${TORCHRUN_NPROC_PER_NODE:-1}"
MP="${MASTER_PORT:-28766}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ "${N}" -gt 1 ] 2>/dev/null; then
  exec torchrun --nproc_per_node="${N}" --master_port="${MP}" -m reglu.cli forget --config "${CFG}"
fi
exec "${PYTHON_BIN:-python}" -m reglu.cli forget --config "${CFG}"
