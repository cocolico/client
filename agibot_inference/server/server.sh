#!/usr/bin/env bash
# 一次加载一个任务；切换任务/模型请用 restart --task ... --model ...。
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION=agibot-inference
LOG="/data/agibot_inference/logs/server.log"
FASTWAM_PY="$HOME/.local/share/agibot_inference/venvs/fastwam/bin/python"
ACTION="${1:-status}"
if (($#)); then shift; fi

stop_server() {
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then return; fi
  tmux send-keys -t "$SESSION" C-c
  for ((i=0; i<100; i++)); do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then return; fi
    sleep .2
  done
  echo "服务未退出，请检查日志；未启动第二个模型。" >&2
  return 1
}

case "$ACTION" in
  restart) stop_server ;;
  stop) stop_server; echo "本地服务已停止。"; exit 0 ;;
  status) exec "$FASTWAM_PY" "$ROOT/healthcheck.py" ;;
  logs) exec tail -n 80 -f "$LOG" ;;
  start) ;;
  *) echo "用法：bash server.sh {start|restart|stop|status|logs} [--task clothes|guizi|shuiping|dengpao] [--model ours|fastwam|pi05]" >&2; exit 2 ;;
esac
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "已有服务；切换任务/模型请使用 restart。" >&2
  "$FASTWAM_PY" "$ROOT/healthcheck.py"
  if (($#)); then exit 2; fi
  exit 0
fi
mkdir -p /data/agibot_inference/logs
MODEL=fastwam
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --model)
      ((i+=1))
      MODEL="${ARGS[i]:?--model 缺少值}"
      ;;
    --model=*) MODEL="${ARGS[i]#--model=}" ;;
  esac
done
case "$MODEL" in
  ours|fastwam) START_SCRIPT="$ROOT/experiments/agibot/start_ours_server.sh" ;;
  pi05) START_SCRIPT="$ROOT/experiments/agibot/start_pi05_server.sh" ;;
  *) echo "未知模型：$MODEL；可选 ours|fastwam|pi05" >&2; exit 2 ;;
esac
# 对每个参数进行 shell 转义后传给 tmux，保留用户的参数边界。
printf -v COMMAND '%q ' bash "$START_SCRIPT" "$@"
printf -v LOG_PATH '%q' "$LOG"
tmux new-session -d -s "$SESSION" "exec $COMMAND >> $LOG_PATH 2>&1"
echo "正在加载；bash $ROOT/server.sh status 查看任务和真实训练步数。日志：$LOG"
# 等模型实际就绪再返回，随后可直接运行 client。
for ((i=0; i<90; i++)); do
  if "$FASTWAM_PY" "$ROOT/healthcheck.py" > "/data/agibot_inference/logs/startup-status.json" 2>/dev/null; then
    cat "/data/agibot_inference/logs/startup-status.json"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "模型加载失败：" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  sleep 2
done
echo "模型启动超时，请检查 $LOG" >&2
exit 1
