"""EgoStandard video dataset with versioned 23D first-camera targets."""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_f

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from fastwam.datasets.egostandard_geometry import (
    FIRST_CAMERA_FRAME, first_camera_targets, validate_representation,
)


def normalize_task_label(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", task.casefold())


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-12):
        raise ValueError("Quaternion must be finite and have non-zero norm")
    normalized = quaternion / norm
    return np.where(normalized[..., :1] < 0.0, -normalized, normalized)


def _slerp(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    left = _normalize_quaternion(left)
    right = _normalize_quaternion(right)
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = sin_theta < 1e-6
    weight_left = np.array(1.0 - alpha, copy=True)
    weight_right = np.array(alpha, copy=True)
    np.divide(
        np.sin((1.0 - alpha) * theta),
        sin_theta,
        out=weight_left,
        where=~linear,
    )
    np.divide(
        np.sin(alpha * theta),
        sin_theta,
        out=weight_right,
        where=~linear,
    )
    return _normalize_quaternion(weight_left * left + weight_right * right)


def _interpolation_indices(source_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(source_ns, query_ns, side="left")
    upper = np.clip(upper, 1, len(source_ns) - 1)
    lower = upper - 1
    denominator = (source_ns[upper] - source_ns[lower]).astype(np.float64)
    alpha = ((query_ns - source_ns[lower]) / denominator).reshape(-1, 1)
    return lower, upper, alpha


def _interpolate_pose(
    source_ns: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    query_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower, upper, alpha = _interpolation_indices(source_ns, query_ns)
    interpolated_position = (1.0 - alpha) * position[lower] + alpha * position[upper]
    interpolated_quaternion = _slerp(quaternion[lower], quaternion[upper], alpha)
    inside = (query_ns >= source_ns[0]) & (query_ns <= source_ns[-1])
    return interpolated_position, interpolated_quaternion, lower, upper


class EgoStandardCachedDataset(torch.utils.data.Dataset):
    """Return FastWAM-compatible samples from immutable EgoStandard episode caches.

    Left/right ``xyz+qwxyz+gripper`` (8D each) and camera ``xyz+qwxyz`` (7D)
    default to the fixed camera frame at the chunk's conditioning timestamp.
    World-frame output is an explicit legacy/debug option, not the default.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        num_video_frames: int = 9,
        video_interval_s: float = 0.08,
        action_horizon: int = 32,
        action_frequency_hz: float = 50.0,
        sample_stride_s: float = 0.32,
        video_size: list[int] | tuple[int, int] = (224, 224),
        video_backend: str = "pyav",
        normalize_video: bool = True,
        cache_root_override: str | None = None,
        processor: Any = None,
        action_representation: str = FIRST_CAMERA_FRAME,
    ) -> None:
        del processor  # Compatibility namespace used by the existing model config.
        self.action_representation = validate_representation(action_representation)
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be train, val, or all; got {split}")
        if num_video_frames < 2 or num_video_frames % 4 != 1:
            raise ValueError("num_video_frames must be >1 and satisfy T % 4 == 1")
        if action_horizon % (num_video_frames - 1) != 0:
            raise ValueError("action_horizon must be divisible by video transitions")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = split
        self.num_video_frames = int(num_video_frames)
        self.video_interval_s = float(video_interval_s)
        self.action_horizon = int(action_horizon)
        self.action_frequency_hz = float(action_frequency_hz)
        self.sample_stride_s = float(sample_stride_s)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.video_backend = str(video_backend)
        self.normalize_video = bool(normalize_video)
        self.cache_root_override = (
            None if cache_root_override is None else Path(cache_root_override).expanduser().resolve()
        )
        self.records = self._load_manifest()
        self._window_counts = [self._window_count(record) for record in self.records]
        self._cumulative: list[int] = []
        running = 0
        groups: dict[str, int] = {}
        group_ids: list[int] = []
        for record, count in zip(self.records, self._window_counts):
            running += count
            self._cumulative.append(running)
            alias = normalize_task_label(str(record["task_directory"]))
            group_ids.extend([groups.setdefault(alias, len(groups))] * count)
        if running == 0:
            raise ValueError(f"Manifest contains no usable windows: {self.manifest_path}")
        self.sampling_group_ids = tuple(group_ids)
        self.sampling_group_count = len(groups)

    @property
    def slide_duration_s(self) -> float:
        return max(
            (self.num_video_frames - 1) * self.video_interval_s,
            (self.action_horizon - 1) / self.action_frequency_hz,
        )

    def _load_manifest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if self.split != "all" and record["split"] != self.split:
                    continue
                if self.cache_root_override is not None:
                    episode_parts = Path(record["episode_root"]).parts
                    try:
                        episodes_index = episode_parts.index("episodes")
                    except ValueError as exc:
                        raise ValueError(
                            f"Manifest line {line_number} has no episodes/ path: "
                            f"{record['episode_root']}"
                        ) from exc
                    episode_relative = Path(*episode_parts[episodes_index:])
                    local_episode = self.cache_root_override / episode_relative
                    record["episode_root"] = str(local_episode)
                    record["video_path"] = str(local_episode / "head_left.mp4")
                    record["pose_path"] = str(local_episode / "poses.npz")
                for key in ("video_path", "pose_path"):
                    if not Path(record[key]).is_file():
                        raise FileNotFoundError(f"Manifest line {line_number}: {record[key]}")
                records.append(record)
        if not records:
            raise ValueError(f"No records for split={self.split} in {self.manifest_path}")
        return records

    def _window_count(self, record: dict[str, Any]) -> int:
        video_duration = (int(record["num_video_frames"]) - 1) / float(record["fps"])
        available = video_duration - self.slide_duration_s
        return 0 if available < 0 else int(math.floor(available / self.sample_stride_s)) + 1

    def __len__(self) -> int:
        return self._cumulative[-1]

    def _resolve_index(self, index: int) -> tuple[int, float]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect_right(self._cumulative, index)
        previous = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        return episode_index, (index - previous) * self.sample_stride_s

    @staticmethod
    def _hand_targets(
        pose: Any,
        side: str,
        query_ns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion, lower, upper = _interpolate_pose(
            pose["pose_time_ns"],
            pose[f"hand_{side}_world_position"][:, 0],
            pose[f"hand_{side}_world_quat_wxyz"][:, 0],
            query_ns,
        )
        aperture = np.interp(query_ns, pose["pose_time_ns"], pose[f"{side}_aperture_m"])
        palm_width = np.interp(query_ns, pose["pose_time_ns"], pose[f"{side}_palm_width_m"])
        gripper = np.clip(aperture / np.maximum(palm_width, 1e-6), 0.0, 2.0)
        source_valid = pose[f"{side}_valid"].astype(bool)
        valid = source_valid[lower] & source_valid[upper]
        action = np.concatenate([position, quaternion, gripper[:, None]], axis=1)
        valid &= np.isfinite(action).all(axis=1)
        valid &= valid[0]
        return action.astype(np.float32), action[0].astype(np.float32), valid

    @staticmethod
    def _head_targets(
        pose: Any,
        query_ns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion, _, _ = _interpolate_pose(
            pose["pose_time_ns"],
            pose["head_left_world_position"],
            pose["head_left_world_quat_wxyz"],
            query_ns,
        )
        action = np.concatenate([position, quaternion], axis=1).astype(np.float32)
        return action, np.isfinite(action).all(axis=1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, start_s = self._resolve_index(index)
        record = self.records[episode_index]
        with np.load(record["pose_path"], allow_pickle=False) as pose:
            video_origin_ns = int(pose["video_time_ns"][0])
            start_ns = video_origin_ns + int(round(start_s * 1e9))
            action_ns = start_ns + np.rint(
                np.arange(self.action_horizon, dtype=np.float64) * 1e9 / self.action_frequency_hz
            ).astype(np.int64)
            left_action, left_initial, left_valid = self._hand_targets(
                pose, "left", action_ns
            )
            right_action, right_initial, right_valid = self._hand_targets(
                pose, "right", action_ns
            )
            head_action, head_valid = self._head_targets(pose, action_ns)

        video_timestamps = [start_s + step * self.video_interval_s for step in range(self.num_video_frames)]
        video = decode_video_frames(
            Path(record["video_path"]),
            video_timestamps,
            tolerance_s=0.5 / float(record["fps"]) + 1e-3,
            backend=self.video_backend,
        )
        video = transforms_f.resize(
            video,
            list(self.video_size),
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        ).to(torch.float32)
        if self.normalize_video:
            video = video.mul(2.0).sub(1.0)

        action = np.concatenate([left_action, right_action, head_action], axis=1)
        valid_mask = np.concatenate(
            [
                np.repeat(left_valid[:, None], 8, axis=1),
                np.repeat(right_valid[:, None], 8, axis=1),
                np.repeat(head_valid[:, None], 7, axis=1),
            ],
            axis=1,
        )
        # Invalid annotations must never introduce NaNs into the denoising
        # trajectory: masking a NaN loss after MSE is too late because NaN * 0
        # remains NaN. Values are neutralized here and excluded by the mask.
        action = np.where(valid_mask, action, 0.0).astype(np.float32)
        if not left_valid[0]:
            left_initial = np.zeros_like(left_initial)
        if not right_valid[0]:
            right_initial = np.zeros_like(right_initial)
        head_initial = head_action[0] if head_valid[0] else np.zeros(7, np.float32)
        left_initial = np.nan_to_num(left_initial)
        right_initial = np.nan_to_num(right_initial)
        proprio_0 = np.concatenate(
            [left_initial, right_initial, np.nan_to_num(head_initial)]
        ).astype(np.float32)
        proprio = np.repeat(proprio_0[None], self.action_horizon, axis=0)
        if self.action_representation == FIRST_CAMERA_FRAME:
            action, valid_mask, proprio = first_camera_targets(action, valid_mask, proprio)
        prompt = f"A video recorded from a person's point of view executing: {record['description']}"
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "action": torch.from_numpy(action),
            "action_valid_mask": torch.from_numpy(valid_mask),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "proprio": torch.from_numpy(proprio),
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "image_is_pad": torch.zeros(self.num_video_frames, dtype=torch.bool),
            "prompt": prompt,
            "task": str(record["task_directory"]),
            "episode_id": str(record["episode_id"]),
            "slide_start_s": torch.tensor(start_s, dtype=torch.float32),
        }
