# Deployment and verification

This checkout runs on `ssh real` at `/home/descfly/code/agibot_inference`. The original dated delivery at `/home/descfly/code/agibot_code_20260924` remains available for rollback. The model files stay outside Git under `/data/real8_*` and are exposed through stable links under `/data/agibot_inference/models/`.

## Runtime requirements

- AgiBot GDK 3.3.8, compatible robot firmware, and the existing `gdk-py310` Conda environment.
- RTX 5090, FastWAM Python 3.10 / PyTorch 2.7.1+cu128, Pi0.5 Python 3.11 / JAX 0.6.2 with CUDA 13 `ptxas`.
- FastWAM environment: `~/.local/share/agibot_inference/venvs/fastwam`.
- Pi0.5 environment: `~/.local/share/agibot_inference/venvs/pi05`.
- Offline tokenizer: `~/.cache/openpi/big_vision/paligemma_tokenizer.model` (source copy under `assets/openpi/`).

Load the client environment with `source ~/env_agibot.sh`. Do not run `uv sync` against the Pi0.5 lock without restoring the RTX 5090 compatible JAX 0.6.2 packages afterward.

## Verify, without robot motion

```bash
source ~/env_agibot.sh
bin/agibot server status
bin/agibot test --task guizi --model pi05
bin/agibot replay data/raw_data/guizi/0914柜子_20260914174235
```

To switch models, use `bin/agibot server restart --task guizi --model fastwam` or `--model ours`, then run the matching `bin/agibot test`. Check `server status` for model, task, step, 20 Hz, 19-joint order, and the checkpoint path. The expected observation is RGB `(480,640,3)` plus state `(19,)`; expected actions are `(32,19)`.

The current robot has no old one-DOF grippers. Legacy model actions and live 19D observations are incompatible with its two Sharpa Wave SE hands. `once`, `interactive`, and `replay --execute` are disabled; use `test` and replay validation only. See `SHARPA_POSITION.md` for independent hand read/move commands and the physical feedback test status. Do not infer new 22D hand actions from an old gripper scalar.

Logs: `/data/agibot_inference/logs/server.log`; text features: `/data/agibot_inference/text/`. Status: `bin/agibot server status`. Model links: `/data/agibot_inference/models/{fastwam,ours,pi05}`. The original checkpoints and source handoff are untouched.
