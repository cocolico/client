from pathlib import Path

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI05Sharpa61:
    """Inference-only reconstruction of pi05_base_agibot_sharpa61_full.

    Uses the original training adapter and completed checkpoint, not a base
    weight loader or the legacy Aloha/active-only transforms.
    """
    def __init__(self, train_config_name: str, checkpoint_dir: str, action_chunk: int = 50):
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.sharpa_policy import SharpaInputs, SharpaOutputs
        from openpi import transforms

        if train_config_name != 'pi05_base_agibot_sharpa61_full' or action_chunk != 50:
            raise ValueError('Sharpa checkpoint requires pi05_base_agibot_sharpa61_full / 50 frames')
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        config = _config.TrainConfig(
            name=train_config_name,
            model=Pi0Config(pi05=True, action_dim=61, action_horizon=50,
                            max_token_len=320, paligemma_variant='gemma_2b',
                            action_expert_variant='gemma_300m'),
            data=_config.SimpleDataConfig(
                repo_id='yifu',
                assets=_config.AssetsConfig(assets_dir=str(self.checkpoint_dir / 'assets'), asset_id='yifu'),
                data_transforms=lambda _: transforms.Group(
                    inputs=[SharpaInputs(camera_mode='three')], outputs=[SharpaOutputs()]),
            ),
        )
        self.config = config
        self.policy = _policy_config.create_trained_policy(config, self.checkpoint_dir, robotwin_repo_id='yifu')

    def infer(self, payload: dict) -> np.ndarray:
        actions = np.asarray(self.policy.infer(payload)['actions'], dtype=np.float32)
        if actions.shape != (50, 61):
            raise ValueError(f'Sharpa checkpoint returned {actions.shape}, expected (50, 61)')
        return actions


def select_action_chunk(actions: np.ndarray, chunk_size: int | None) -> np.ndarray:
    if chunk_size is None:
        return actions
    return actions[:chunk_size]


class PI05ActiveOnly:
    def __init__(self, train_config_name: str, checkpoint_dir: str, action_chunk: int | None):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        config = _config.get_config(train_config_name)

        assets_dir = self.checkpoint_dir / "assets"
        norm_stats_paths = list(assets_dir.rglob("norm_stats.json"))
        if len(norm_stats_paths) != 1:
            raise ValueError(
                "checkpoint must contain exactly one norm_stats.json, "
                f"found {len(norm_stats_paths)} under {assets_dir}"
            )

        assets_id = norm_stats_paths[0].parent.relative_to(assets_dir).as_posix()
        self.policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_dir,
            robotwin_repo_id=assets_id,
        )
        self.action_chunk = action_chunk
        self.instruction: str | None = None
        self.observation_window: dict | None = None
        print("Pi0.5 active-only model loaded successfully")

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def update_observation_window(self, img_arr: list[np.ndarray], state: np.ndarray) -> None:
        img_front = np.transpose(img_arr[0], (2, 0, 1))
        self.observation_window = {
            "state": state,
            "images": {
                "cam_active": img_front,
            },
            "prompt": self.instruction,
        }

    def get_action(self) -> np.ndarray:
        if self.observation_window is None:
            raise RuntimeError("update_observation_window must run before inference")
        return self.policy.infer(self.observation_window)["actions"]

    def infer(self, payload: dict) -> np.ndarray:
        self.set_language(payload["prompt"])
        self.update_observation_window(payload["images"], payload["state"])
        return select_action_chunk(self.get_action(), self.action_chunk)

    def reset_model(self) -> None:
        self.instruction = None
        self.observation_window = None
