#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/descfly/code/agibot_inference
DEST="$ROOT/checkpoints/diguan_three_camera"
REMOTE='114.67.229.27'
SSH='ssh -l haoran_liao:litchi_data_017 -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i /home/descfly/.ssh/jd_inference_mac_authorized -p 2222'
REPO=/root/haoran/project/egoav/RoboTwin-AV-agibot
POLICY="$REPO/policy/pi05"
RUN=diguan_three_fast50k_n23_20261012_isolated
CKPT="$POLICY/checkpoints_sharpa61/pi05_base_agibot_sharpa61_full/${RUN}_bs128/50000"
mkdir -p "$DEST/50000" "$DEST/support/dataset/meta"
exec 9>"$DEST/.transfer.lock"
flock -n 9 || exit 2
trap 'rc=$?; if ((rc != 0)); then echo "TRANSFER_FAILED exit=$rc $(date -Is)"; fi' EXIT
echo "TRANSFER_STARTED $(date -Is)"
support=(
 "src/openpi/training/config.py"
 "docs/sharpa61_training.md"
 "scripts/sharpa_training_run.py"
 "scripts/sharpa_camera_test.py"
 "assets_sharpa61/pi05_base_agibot_sharpa61_full/diguan/"
 "training_runs_sharpa61/$RUN/current.json"
 "training_runs_sharpa61/$RUN/status.json"
 "wandb/run-20261012_225901-x55lnmop/files/config.yaml"
 "wandb/run-20261012_225901-x55lnmop/files/requirements.txt"
 "wandb/run-20261012_225901-x55lnmop/files/wandb-summary.json"
)
sources=()
for path in "${support[@]}"; do sources+=("$REMOTE:$POLICY/./$path"); done
rsync -aR -e "$SSH" "${sources[@]}" "$DEST/support/"
meta=(info.json tasks.jsonl conversion.json validation.json robot.urdf)
sources_meta=()
for path in "${meta[@]}"; do sources_meta+=("$REMOTE:$REPO/data_lerobot/diguan/meta/$path"); done
rsync -a -e "$SSH" "${sources_meta[@]}" "$DEST/support/dataset/meta/"
echo "CONFIG_COPIED_HASHING_SOURCE $(date -Is)"
$SSH "$REMOTE" "cd '$CKPT' && find params assets _CHECKPOINT_METADATA -type f -print0 | sort -z | xargs -0 sha256sum" > "$DEST/50000/SHA256SUMS.source.pending"
mv "$DEST/50000/SHA256SUMS.source.pending" "$DEST/50000/SHA256SUMS.source"
rsync -a --partial --inplace --info=progress2 --stats -e "$SSH" --exclude='/train_state/' "$REMOTE:$CKPT/" "$DEST/50000/"
echo "TRANSFER_DONE_VERIFYING $(date -Is)"
(cd "$DEST/50000" && sha256sum -c SHA256SUMS.source)
rsync -rcnR --out-format='%i %n%L' -e "$SSH" "${sources[@]}" "$DEST/support/" > "$DEST/support-verification.diff"
rsync -rcn --out-format='%i %n%L' -e "$SSH" "${sources_meta[@]}" "$DEST/support/dataset/meta/" > "$DEST/dataset-verification.diff"
test ! -s "$DEST/support-verification.diff"
test ! -s "$DEST/dataset-verification.diff"
echo "VERIFIED_COMPLETE $(date -Is)"
