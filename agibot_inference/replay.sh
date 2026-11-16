#!/usr/bin/env bash
# real / Agibot only. Selection changes; existing replay execution is reused.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK=''
EPISODE=''
DRY_RUN=0
LIST=0
usage() {
  cat <<'HELP'
用法: bash replay.sh --task yifu|diguan|dengpao|drink --episode 编号或录制名 [--dry-run]
列表: bash replay.sh --task diguan --list
默认执行真机，先恢复所选数据首帧；--dry-run 仅离线预检。
drink 同步回放腰部旋转，其他任务不开放腰部运动。
编号映射固定在 configs/replay_tasks.json。yifu 0 保留原71维样本。
其他条目回放本地原始录制反馈，不是云端裁剪/删帧后的78维训练数据。
HELP
}
fail() { echo "错误: $*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    --task|--episode)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "$1 缺少值"
      if [[ "$1" == --task ]]; then
        [[ -z "$TASK" ]] || fail '重复 --task'; TASK="$2"
      else
        [[ -z "$EPISODE" ]] || fail '重复 --episode'; EPISODE="$2"
      fi
      shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "不支持的参数: $1" ;;
  esac
done
if ((!LIST)); then [[ -n "$TASK" ]] || fail '必须指定 --task'; fi
if ((!LIST)); then [[ -n "$EPISODE" ]] || fail '必须指定 --episode'; fi
# Resolve user-supplied relative paths before env.sh changes the working directory.
if [[ "$EPISODE" == */* && -d "$EPISODE" ]]; then
  EPISODE="$(cd -- "$EPISODE" && pwd)"
fi
ARGS=()
if [[ -n "$TASK" ]]; then ARGS+=(--task "$TASK"); fi
if ((LIST)); then
  ARGS+=(--list)
elif ((DRY_RUN)); then
  ARGS+=(--episode "$EPISODE" --dry-run)
else
  ARGS+=(--episode "$EPISODE" --execute)
  echo '恢复所选录制首帧后完整回放；Ctrl+C停止后续发送。'
fi
export PYTHONDONTWRITEBYTECODE=1
source "$ROOT/env.sh" >/dev/null
exec python "$ROOT/scripts/replay_tasks.py" "${ARGS[@]}"
