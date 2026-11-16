#!/usr/bin/env bash
# Source this file to load the GDK client Python and enter this repository.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source /home/descfly/code/agibot_inference/env.sh' >&2
  exit 1
fi
_agibot_load_env() {
  local project_dir app_dir conda_init
  local SCRIPT_DIR DIR SO_DIRS NEW_LIB_PATH BIN_PATH local_ip
  project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  app_dir="$HOME/.cache/agibot/app"
  conda_init="$HOME/miniconda3/etc/profile.d/conda.sh"
  [[ -f "$conda_init" && -f "$app_dir/env.sh" ]] || {
    echo 'Missing existing Conda or GDK installation' >&2
    return 1
  }
  source "$conda_init" || return
  conda activate gdk-py310 || return
  source "$app_dir/env.sh" "$app_dir" || return
  cd "$project_dir" || return
}
_agibot_load_env
