#!/usr/bin/env python3
"""Compare direct MCAP window decoding with the existing immutable pilot cache."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from fastwam.datasets.egostandard_geometry import FIRST_CAMERA_FRAME, first_camera_targets

from fastwam.datasets.egostandard_cached_dataset import (
    EgoStandardCachedDataset,
    _interpolate_pose,
)


VIDEO_TOPIC = "/sensor/camera/head_left/video"
LEFT_HAND_TOPIC = "/pose/left_hand"
RIGHT_HAND_TOPIC = "/pose/right_hand"
HEADCAM_TOPIC = "/pose/headcam"
BAD_HAND_TOPIC = "/annotation/bad_frame/pose/hand"
WINDOW_TOPICS = (VIDEO_TOPIC, LEFT_HAND_TOPIC, RIGHT_HAND_TOPIC, HEADCAM_TOPIC, BAD_HAND_TOPIC)


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return None
    return vector / norm


def gripper_from_points(points: np.ndarray) -> tuple[float, float, bool]:
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return float("nan"), float("nan"), False
    forward = normalize(np.mean(points[[5, 9, 17]], axis=0) - points[0])
    if forward is None:
        return float("nan"), float("nan"), False
    across_raw = points[5] - points[17]
    across = normalize(across_raw - np.dot(across_raw, forward) * forward)
    if across is None or normalize(np.cross(forward, across)) is None:
        return float("nan"), float("nan"), False
    aperture = float(np.linalg.norm(points[4] - points[8]))
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    valid = bool(0.025 <= palm_width <= 0.16 and 0.0 <= aperture <= 0.20)
    return aperture, palm_width, valid


def transform_array(transforms: Any) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        [[item.pos.x, item.pos.y, item.pos.z] for item in transforms], dtype=np.float32
    )
    quaternions = np.asarray(
        [[item.quat.w, item.quat.x, item.quat.y, item.quat.z] for item in transforms],
        dtype=np.float32,
    )
    return positions, quaternions


def has_idr(data: bytes) -> bool:
    index = 0
    while index + 4 < len(data):
        if data[index : index + 4] == b"\0\0\0\1":
            nal_index = index + 4
        elif data[index : index + 3] == b"\0\0\1":
            nal_index = index + 3
        else:
            index += 1
            continue
        if nal_index < len(data) and data[nal_index] & 31 == 5:
            return True
        index = nal_index + 1
    return False


def first_video_clock(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, message, decoded in reader.iter_decoded_messages(topics=[VIDEO_TOPIC]):
            timestamp_ns = int(decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos)
            return message.log_time, timestamp_ns
    raise ValueError(f"No left video messages: {path}")


def read_window_messages(
    path: Path,
    origin_log_ns: int,
    origin_video_ns: int,
    start_s: float,
    duration_s: float,
) -> tuple[list[tuple[int, bytes]], dict[int, Any], dict[int, Any], dict[int, Any], dict[int, bool]]:
    lookback_s = min(start_s, 1.25)
    for _ in range(5):
        lower_log_ns = origin_log_ns + int(round((start_s - lookback_s) * 1e9))
        upper_log_ns = origin_log_ns + int(round((start_s + duration_s + 0.20) * 1e9))
        video: list[tuple[int, bytes]] = []
        left: dict[int, Any] = {}
        right: dict[int, Any] = {}
        head: dict[int, Any] = {}
        bad: dict[int, bool] = {}
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, _, decoded in reader.iter_decoded_messages(
                topics=WINDOW_TOPICS,
                start_time=lower_log_ns,
                end_time=upper_log_ns,
            ):
                if channel.topic == VIDEO_TOPIC:
                    timestamp_ns = int(
                        decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
                    )
                    video.append((timestamp_ns, bytes(decoded.data)))
                elif channel.topic == LEFT_HAND_TOPIC:
                    left[int(decoded.header.ts)] = transform_array(decoded.transforms)
                elif channel.topic == RIGHT_HAND_TOPIC:
                    right[int(decoded.header.ts)] = transform_array(decoded.transforms)
                elif channel.topic == HEADCAM_TOPIC:
                    head[int(decoded.header.ts)] = transform_array(decoded.transforms[:1])
                elif channel.topic == BAD_HAND_TOPIC:
                    bad[int(decoded.header.ts)] = bool(decoded.is_bad)
        target_ns = origin_video_ns + int(round(start_s * 1e9))
        idr_before_target = [index for index, (ts, data) in enumerate(video) if ts <= target_ns and has_idr(data)]
        if idr_before_target:
            return video[idr_before_target[-1] :], left, right, head, bad
        if start_s - lookback_s <= 0:
            break
        lookback_s = min(start_s, lookback_s * 2)
    raise ValueError(f"No preceding IDR for {path} at {start_s=}")


def decode_video(
    video_messages: list[tuple[int, bytes]],
    video_time_ns: np.ndarray | None,
    average_rate: Fraction,
    time_base: Fraction,
    timestamps_s: list[float],
    video_size: tuple[int, int],
    first_global_index: int | None = None,
) -> torch.Tensor:
    first_timestamp_ns = video_messages[0][0]
    if first_global_index is None:
        if video_time_ns is None:
            raise ValueError("video_time_ns or first_global_index is required")
        first_global_index = int(np.searchsorted(video_time_ns, first_timestamp_ns, side="left"))
        if (
            first_global_index >= len(video_time_ns)
            or int(video_time_ns[first_global_index]) != first_timestamp_ns
        ):
            raise ValueError(f"Video timestamp is absent from ordinal index: {first_timestamp_ns}")
    target_local_indices: list[int] = []
    for timestamp_s in timestamps_s:
        center = int(math.floor(timestamp_s * float(average_rate)))
        candidate_indices = torch.arange(max(0, center - 2), center + 4, dtype=torch.int64)
        query = torch.tensor(timestamp_s, dtype=torch.float32)
        candidate_timestamps = torch.tensor(
            [
                float(round(Fraction(int(index), 1) / average_rate / time_base) * time_base)
                for index in candidate_indices
            ],
            dtype=torch.float32,
        )
        target_index = int(candidate_indices[torch.argmin(torch.abs(candidate_timestamps - query))])
        local_index = target_index - first_global_index
        if local_index < 0:
            raise IndexError(f"Decoded range starts after target: {target_index=} {first_global_index=}")
        target_local_indices.append(local_index)

    # H264 inter-frame dependencies still have to be decoded from the preceding
    # IDR, but RGB conversion is needed only for the nine frames returned to the
    # model. Avoid materializing every dependency frame as a full-resolution
    # RGB tensor.
    needed_indices = set(target_local_indices)
    selected_by_index: dict[int, torch.Tensor] = {}
    decoded_count = 0

    def collect(frame: av.VideoFrame) -> None:
        nonlocal decoded_count
        if decoded_count in needed_indices:
            selected_by_index[decoded_count] = torch.from_numpy(
                frame.to_ndarray(format="rgb24")
            ).permute(2, 0, 1)
        decoded_count += 1

    codec = av.CodecContext.create("h264", "r")
    for _, data in video_messages:
        for packet in codec.parse(data):
            for frame in codec.decode(packet):
                collect(frame)
    for packet in codec.parse(b""):
        for frame in codec.decode(packet):
            collect(frame)
    for frame in codec.decode(None):
        collect(frame)
    missing = sorted(needed_indices - selected_by_index.keys())
    if missing:
        raise IndexError(
            f"Decoded range misses local targets: missing={missing} frames={decoded_count}"
        )
    selected = [selected_by_index[index] for index in target_local_indices]
    video = torch.stack(selected).to(torch.float32).div(255.0)
    return transforms_f.resize(
        video,
        list(video_size),
        interpolation=transforms_f.InterpolationMode.BILINEAR,
        antialias=True,
    ).mul(2.0).sub(1.0)


def build_pose_arrays(
    left: dict[int, Any],
    right: dict[int, Any],
    head: dict[int, Any],
    bad: dict[int, bool],
) -> dict[str, np.ndarray]:
    common = np.asarray(sorted(set(left) & set(right) & set(head)), dtype=np.int64)
    if common.size < 2:
        raise ValueError("Fewer than two synchronized pose frames")
    left_position = np.stack([left[int(ts)][0] for ts in common])
    left_quat = np.stack([left[int(ts)][1] for ts in common])
    right_position = np.stack([right[int(ts)][0] for ts in common])
    right_quat = np.stack([right[int(ts)][1] for ts in common])
    head_position = np.stack([head[int(ts)][0][0] for ts in common])
    head_quat = np.stack([head[int(ts)][1][0] for ts in common])
    left_gripper = [gripper_from_points(points) for points in left_position]
    right_gripper = [gripper_from_points(points) for points in right_position]
    annotation_valid = np.asarray([not bad.get(int(ts), False) for ts in common])
    return {
        "pose_time_ns": common,
        "hand_left_world_position": left_position,
        "hand_left_world_quat_wxyz": left_quat,
        "hand_right_world_position": right_position,
        "hand_right_world_quat_wxyz": right_quat,
        "head_left_world_position": head_position,
        "head_left_world_quat_wxyz": head_quat,
        "left_aperture_m": np.asarray([item[0] for item in left_gripper], dtype=np.float32),
        "left_palm_width_m": np.asarray([item[1] for item in left_gripper], dtype=np.float32),
        "left_valid": annotation_valid & np.asarray([item[2] for item in left_gripper]),
        "right_aperture_m": np.asarray([item[0] for item in right_gripper], dtype=np.float32),
        "right_palm_width_m": np.asarray([item[1] for item in right_gripper], dtype=np.float32),
        "right_valid": annotation_valid & np.asarray([item[2] for item in right_gripper]),
    }


def direct_targets(
    dataset: EgoStandardCachedDataset,
    pose: dict[str, np.ndarray],
    origin_video_ns: int,
    start_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_ns = origin_video_ns + int(round(start_s * 1e9))
    action_ns = start_ns + np.rint(
        np.arange(dataset.action_horizon, dtype=np.float64)
        * 1e9
        / dataset.action_frequency_hz
    ).astype(np.int64)
    left_action, left_initial, left_valid = dataset._hand_targets(
        pose, "left", action_ns
    )
    right_action, right_initial, right_valid = dataset._hand_targets(
        pose, "right", action_ns
    )
    head_action, head_valid = dataset._head_targets(pose, action_ns)
    action = np.concatenate([left_action, right_action, head_action], axis=1)
    valid = np.concatenate(
        [
            np.repeat(left_valid[:, None], 8, axis=1),
            np.repeat(right_valid[:, None], 8, axis=1),
            np.repeat(head_valid[:, None], 7, axis=1),
        ],
        axis=1,
    )
    action = np.where(valid, action, 0.0).astype(np.float32)
    if not left_valid[0]:
        left_initial = np.zeros_like(left_initial)
    if not right_valid[0]:
        right_initial = np.zeros_like(right_initial)
    head_initial = head_action[0] if head_valid[0] else np.zeros(7, np.float32)
    proprio_0 = np.concatenate(
        [
            np.nan_to_num(left_initial),
            np.nan_to_num(right_initial),
            np.nan_to_num(head_initial),
        ]
    )
    proprio = np.repeat(proprio_0[None], dataset.action_horizon, axis=0)
    if dataset.action_representation == FIRST_CAMERA_FRAME:
        action, valid, proprio = first_camera_targets(action, valid, proprio)
    return torch.from_numpy(action), torch.from_numpy(valid), torch.from_numpy(proprio)


class EgoStandardMcapDataset(EgoStandardCachedDataset):
    """Read one exact training window directly from indexed MCAP chunks.

    The pilot schema reuses the immutable cache manifest only for episode metadata,
    the video timestamp ordinal table, and the remux PTS rational. RGB and Cartesian
    targets are reconstructed from the source MCAP and have a zero-difference parity
    gate against ``EgoStandardCachedDataset``.
    """

    def __init__(self, *args: Any, index_cache_size: int = 256, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.index_cache_size = max(0, int(index_cache_size))
        self._video_index_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _load_manifest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if self.split != "all" and record["split"] != self.split:
                    continue
                if "index_path" in record:
                    for key in ("source_mcap", "index_path"):
                        if not Path(record[key]).is_file():
                            raise FileNotFoundError(f"Manifest line {line_number}: {record[key]}")
                else:
                    for key in ("video_path", "pose_path"):
                        if not Path(record[key]).is_file():
                            raise FileNotFoundError(f"Manifest line {line_number}: {record[key]}")
                records.append(record)
        if not records:
            raise ValueError(f"No records for split={self.split} in {self.manifest_path}")
        return records

    def _load_video_index(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._video_index_cache.pop(path, None)
        if cached is not None:
            self._video_index_cache[path] = cached
            return cached
        with np.load(path, allow_pickle=False) as payload:
            result = (
                np.array(payload["keyframe_time_ns"], dtype=np.int64, copy=True),
                np.array(payload["keyframe_ordinal"], dtype=np.int64, copy=True),
            )
        if self.index_cache_size > 0:
            self._video_index_cache[path] = result
            while len(self._video_index_cache) > self.index_cache_size:
                self._video_index_cache.popitem(last=False)
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, start_s = self._resolve_index(index)
        record = self.records[episode_index]
        source = Path(record["source_mcap"])
        origin_log_ns = int(record.get("origin_log_ns", 0))
        origin_video_ns = int(record.get("origin_video_ns", 0))
        if origin_log_ns == 0 or origin_video_ns == 0:
            origin_log_ns, origin_video_ns = first_video_clock(source)
        video_messages, left, right, head, bad = read_window_messages(
            source,
            origin_log_ns,
            origin_video_ns,
            start_s,
            self.slide_duration_s,
        )
        first_global_index: int | None = None
        video_time_ns: np.ndarray | None = None
        if "index_path" in record:
            keyframe_times, keyframe_ordinals = self._load_video_index(record["index_path"])
            keyframe_position = int(np.searchsorted(keyframe_times, video_messages[0][0]))
            if (
                keyframe_position >= len(keyframe_times)
                or int(keyframe_times[keyframe_position]) != video_messages[0][0]
            ):
                raise ValueError(
                    f"Decoded IDR is absent from thin index: {source} timestamp={video_messages[0][0]}"
                )
            first_global_index = int(keyframe_ordinals[keyframe_position])
            average_rate = Fraction(
                int(record["average_rate_numerator"]), int(record["average_rate_denominator"])
            )
            time_base = Fraction(
                int(record["time_base_numerator"]), int(record["time_base_denominator"])
            )
        else:
            with np.load(record["pose_path"], allow_pickle=False) as pose_index:
                video_time_ns = np.array(pose_index["video_time_ns"], copy=True)
            with av.open(record["video_path"]) as reference_container:
                reference_stream = reference_container.streams.video[0]
                average_rate = Fraction(reference_stream.average_rate)
                time_base = Fraction(reference_stream.time_base)
        video = decode_video(
            video_messages,
            video_time_ns,
            average_rate,
            time_base,
            [start_s + step * self.video_interval_s for step in range(self.num_video_frames)],
            self.video_size,
            first_global_index=first_global_index,
        )
        action, valid_mask, proprio = direct_targets(
            self,
            build_pose_arrays(left, right, head, bad),
            origin_video_ns,
            start_s,
        )
        prompt = f"A video recorded from a person's point of view executing: {record['description']}"
        return {
            "video": video.permute(1, 0, 2, 3).contiguous(),
            "action": action,
            "action_valid_mask": valid_mask,
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "proprio": proprio,
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "image_is_pad": torch.zeros(self.num_video_frames, dtype=torch.bool),
            "prompt": prompt,
            "task": str(record["task_directory"]),
            "episode_id": str(record["episode_id"]),
            "slide_start_s": torch.tensor(start_s, dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EgoStandardCachedDataset(str(args.manifest), split="train", video_backend="pyav")
    direct_dataset = (
        None
        if args.direct_manifest is None
        else EgoStandardMcapDataset(str(args.direct_manifest), split="train", video_backend="pyav")
    )
    rng = random.Random(args.seed)
    sample_dataset = dataset if direct_dataset is None else direct_dataset
    indices = rng.sample(range(len(sample_dataset)), args.samples)
    reference_episode_indices = {
        str(record["episode_id"]): episode_index
        for episode_index, record in enumerate(dataset.records)
    }
    maxima = {"video": 0.0, "action": 0.0, "mask": 0.0, "proprio": 0.0}
    started = time.perf_counter()
    for ordinal, index in enumerate(indices, start=1):
        if direct_dataset is not None:
            episode_index, start_s = direct_dataset._resolve_index(index)
            record = direct_dataset.records[episode_index]
            reference_episode_index = reference_episode_indices[str(record["episode_id"])]
            previous = (
                0
                if reference_episode_index == 0
                else dataset._cumulative[reference_episode_index - 1]
            )
            reference_index = previous + int(round(start_s / dataset.sample_stride_s))
            reference = dataset[reference_index]
            direct = direct_dataset[index]
            pairs = {
                "video": (reference["video"], direct["video"]),
                "action": (reference["action"], direct["action"]),
                "mask": (reference["action_valid_mask"], direct["action_valid_mask"]),
                "proprio": (reference["proprio"], direct["proprio"]),
            }
            for name, (expected, actual) in pairs.items():
                difference = float(
                    (expected.to(torch.float32) - actual.to(torch.float32)).abs().max()
                )
                maxima[name] = max(maxima[name], difference)
            print(
                json.dumps(
                    {
                        "processed": ordinal,
                        "index": index,
                        "episode_id": record["episode_id"],
                        "start_s": start_s,
                        "max_abs_diff": maxima,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        episode_index, start_s = dataset._resolve_index(index)
        record = dataset.records[episode_index]
        reference = dataset[index]
        source = Path(record["source_mcap"])
        origin_log_ns, origin_video_ns = first_video_clock(source)
        video_messages, left, right, head, bad = read_window_messages(
            source, origin_log_ns, origin_video_ns, start_s, dataset.slide_duration_s
        )
        with np.load(record["pose_path"], allow_pickle=False) as pose_index:
            video_time_ns = np.array(pose_index["video_time_ns"], copy=True)
        with av.open(record["video_path"]) as reference_container:
            reference_stream = reference_container.streams.video[0]
            average_rate = Fraction(reference_stream.average_rate)
            time_base = Fraction(reference_stream.time_base)
        direct_video = decode_video(
            video_messages,
            video_time_ns,
            average_rate,
            time_base,
            [start_s + step * dataset.video_interval_s for step in range(dataset.num_video_frames)],
            dataset.video_size,
        ).permute(1, 0, 2, 3).contiguous()
        direct_action, direct_mask, direct_proprio = direct_targets(
            dataset, build_pose_arrays(left, right, head, bad), origin_video_ns, start_s
        )
        pairs = {
            "video": (reference["video"], direct_video),
            "action": (reference["action"], direct_action),
            "mask": (reference["action_valid_mask"], direct_mask),
            "proprio": (reference["proprio"], direct_proprio),
        }
        for name, (expected, actual) in pairs.items():
            difference = float((expected.to(torch.float32) - actual.to(torch.float32)).abs().max())
            maxima[name] = max(maxima[name], difference)
        print(
            json.dumps(
                {
                    "processed": ordinal,
                    "index": index,
                    "episode_id": record["episode_id"],
                    "start_s": start_s,
                    "max_abs_diff": maxima,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "samples": args.samples,
        "elapsed_seconds": time.perf_counter() - started,
        "max_abs_diff": maxima,
        "ok": all(value == 0.0 for value in maxima.values()),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
