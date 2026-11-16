#!/usr/bin/env python3
"""Extract non-overlapping 20 Hz yifu observations from one raw G2 episode.

Run on real from the agibot_inference repository.  This is read-only with
respect to the raw episode and never initializes GDK or either hand device.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from egoav78_contract import g2_to78
from g2_chunk_camera import G2ChunkCamera, pose_columns
from replay_tasks import prepare_raw
from scipy.spatial.transform import Rotation


DEFAULT_EPISODE = Path(
    "/data/sharpa/pose_studio/data/raw_data/yifu/"
    "1004晾衣服_20261004114719"
)
POLICY_HZ = 20
RAW_HZ = 100
CHUNK = 32
HISTORY_OFFSETS_S = np.asarray([-.8, -.6, -.4, -.2, 0.], dtype=float)


def camera_timestamps(path):
    values = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        values.append(int(fields[0]))
    result = np.asarray(values, dtype=np.int64)
    if not len(result) or np.any(np.diff(result) <= 0):
        raise ValueError("head camera timestamps must be nonempty and increasing")
    return result


def native61(plan, raw_index):
    left = plan["paths"]["left"][raw_index]
    right = plan["paths"]["right"][raw_index]
    hands = np.asarray(plan["hand_path"][raw_index], dtype=np.float64)
    head = np.asarray(plan["head_path"][raw_index], dtype=np.float64)
    value = np.r_[left[1], left[2], hands[0], right[1], right[2], hands[1], head]
    if value.shape != (61,) or not np.isfinite(value).all():
        raise ValueError("raw observation did not produce a finite native61 state")
    return value.astype(np.float32), int(round(left[0] * 1_000_000_000))


def base_state78(value, adapter):
    """Pack raw G2 feedback in av-pipeline's declared source/base_link frame."""
    def pose9(xyz, quat):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
        transform[:3, 3] = xyz
        return pose_columns(transform[None])[0]
    head = adapter.fk.head_pose(adapter.body_target, value[58:61])
    state71 = np.r_[pose9(value[0:3], value[3:7]), value[7:29],
                    pose9(value[29:32], value[32:36]), value[36:58],
                    pose_columns(head[None])[0]]
    result = g2_to78(state71)
    if result.shape != (78,) or not np.isfinite(result).all():
        raise ValueError("base-link state packing failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "input/raw_observations.npz")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.relative_to((ROOT / "tests").resolve())
    output.parent.mkdir(parents=True, exist_ok=True)

    plan = prepare_raw(args.episode.resolve(), "yifu")
    raw_count = len(plan["head_path"])
    if raw_count != len(plan["paths"]["left"]) or raw_count != len(plan["hand_path"]):
        raise ValueError("aligned raw streams disagree on frame count")
    stride = RAW_HZ // POLICY_HZ
    if stride * POLICY_HZ != RAW_HZ:
        raise ValueError("raw/policy rates must have an integral ratio")
    policy_count = (raw_count - 1) // stride + 1
    camera_file = args.episode / "camera/head_color/head_color.txt"
    camera_ns = camera_timestamps(camera_file)
    policy_raw_rows = np.arange(policy_count, dtype=np.int64) * stride
    policy_sample_ns = np.asarray([
        round(plan["paths"]["left"][int(row)][0] * 1_000_000_000)
        for row in policy_raw_rows
    ], dtype=np.int64)
    valid = np.flatnonzero((policy_sample_ns >= camera_ns[0])
                           & (policy_sample_ns <= camera_ns[-1]))
    if not len(valid):
        raise ValueError("head RGB and measured state have no common policy frame")
    first_source_policy_row = int(valid[0])
    source_policy_rows = np.arange(first_source_policy_row, int(valid[-1]) + 1,
                                   CHUNK, dtype=np.int64)
    policy_rows = np.arange(len(source_policy_rows), dtype=np.int64) * CHUNK
    raw_rows = source_policy_rows * stride
    if not np.array_equal(policy_rows, np.arange(len(policy_rows)) * CHUNK):
        raise AssertionError("policy observations must start at 0,32,64,...")

    adapter = G2ChunkCamera(ROOT / "assets/robot/agibot.urdf", plan["body_q"])
    states, chunk_states, native, sample_ns, anchors = [], [], [], [], []
    for row in raw_rows:
        value, stamp = native61(plan, int(row))
        context = adapter.begin(value, sample_ns=stamp)
        states.append(base_state78(value, adapter))
        chunk_states.append(g2_to78(context.state71))
        native.append(value)
        sample_ns.append(stamp)
        anchors.append(context.base_from_camera)

    image_indices, image_ns, image_pad = [], [], []
    for stamp in sample_ns:
        targets = stamp + np.rint(HISTORY_OFFSETS_S * 1e9).astype(np.int64)
        indices = np.searchsorted(camera_ns, targets, side="right") - 1
        padding = indices < 0
        indices = np.maximum(indices, 0)
        image_indices.append(indices)
        image_ns.append(camera_ns[indices])
        image_pad.append(padding)

    np.savez_compressed(
        output,
        state78=np.asarray(states, dtype=np.float32),
        state78_chunk_camera=np.asarray(chunk_states, dtype=np.float32),
        native61=np.asarray(native, dtype=np.float32),
        sample_ns=np.asarray(sample_ns, dtype=np.int64),
        base_from_camera=np.asarray(anchors, dtype=np.float64),
        policy_frame_index=policy_rows,
        source_policy_frame_index=source_policy_rows,
        raw_frame_index=raw_rows,
        image_frame_index=np.asarray(image_indices, dtype=np.int64),
        image_timestamp_ns=np.asarray(image_ns, dtype=np.int64),
        image_padding_mask=np.asarray(image_pad, dtype=bool),
        body_target=np.asarray(plan["body_q"], dtype=np.float64),
    )
    manifest = {
        "status": "complete",
        "task": "yifu",
        "source_episode": str(args.episode.resolve()),
        "source_format": "raw_measured_feedback",
        "request_state_frame": "source/base_link (av-pipeline contract)",
        "raw_frames": raw_count,
        "raw_hz": RAW_HZ,
        "policy_hz": POLICY_HZ,
        "policy_frames": policy_count,
        "first_aligned_source_policy_frame": first_source_policy_row,
        "chunk_frames": CHUNK,
        "observation_policy_frames": policy_rows.tolist(),
        "observation_source_policy_frames": source_policy_rows.tolist(),
        "observation_raw_frames": raw_rows.tolist(),
        "chunks": len(policy_rows),
        "last_chunk_extends_past_source_frames": int(
            source_policy_rows[-1] + CHUNK - (int(valid[-1]) + 1)),
        "robot_io": False,
        "raw_report": plan["report"],
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
