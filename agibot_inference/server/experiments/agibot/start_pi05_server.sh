#!/usr/bin/env bash
# Local Pi0.5 inference server and checkpoint links.
set -euo pipefail
AGIBOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$AGIBOT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.local/share/agibot_inference/venvs/pi05/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec "$PYTHON_BIN" "$AGIBOT_ROOT/server/experiments/agibot/run_pi05_server.py" --host 127.0.0.1 --port 19000 "$@"
