#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ $# -ne 1 ]]; then
  echo "Usage: bash run.sh PORT" >&2
  exit 2
fi
export PYTHONIOENCODING=utf-8
exec "${PYTHON_BIN:-python3}" -u main3.py "$1"
