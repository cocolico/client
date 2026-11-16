#!/usr/bin/env bash
# real 本地推理服务；请使用独立的 FastWAM Python 环境。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.local/share/agibot_inference/venvs/fastwam/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "缺少 FastWAM 环境。请设置 PYTHON_BIN=/path/to/fastwam/python；依赖见 pyproject.toml。" >&2
  exit 1
fi
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH=/data/agibot_inference/models/fastwam/assets
export DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# 19000 避开现有通往 9zy 的 18999 隧道；可以通过 --port 覆盖。
exec "$PYTHON_BIN" experiments/agibot/run_ours_server.py --host 127.0.0.1 --port 19000 "$@"
