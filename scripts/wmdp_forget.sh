#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" -m reglu.cli forget --config "${ROOT_DIR}/configs/wmdp_forget.yaml"
