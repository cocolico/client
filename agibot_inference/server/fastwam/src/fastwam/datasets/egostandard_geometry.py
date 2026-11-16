"""Versioned EgoStandard 23D geometry; no video or robot-data dependencies."""
from __future__ import annotations

import numpy as np

WORLD_FRAME = "world_absolute_xyz_quaternion_wxyz_gripper_proxy"
FIRST_CAMERA_FRAME = "chunk_first_camera_xyz_quaternion_wxyz_gripper_proxy_v1"


def validate_representation(value: str) -> str:
    if value not in (WORLD_FRAME, FIRST_CAMERA_FRAME):
        raise ValueError(f"Unknown EgoStandard action representation: {value}")
    return value


def _unit(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-12):
        raise ValueError("Valid pose requires a finite nonzero wxyz quaternion")
    return q / norm


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), axis=-1)


def _rotate(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    cross = 2.0 * np.cross(q[..., 1:], p)
    return p + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def first_camera_targets(
    action: np.ndarray, valid: np.ndarray, proprio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebase world targets once, using the *observed* camera in proprio[0].

    Input/output layout is L[xyz,wxyz,grip], R[xyz,wxyz,grip], C[xyz,wxyz].
    Masked blocks stay zero, grippers stay unchanged, no input is mutated.
    State must repeat the conditioning observation (never future states).
    """
    action = np.asarray(action)
    valid = np.asarray(valid, dtype=bool)
    proprio = np.asarray(proprio)
    if action.ndim != 2 or action.shape[1] != 23 or not len(action):
        raise ValueError("Expected nonempty [H,23] EgoStandard action")
    if valid.shape != action.shape or proprio.shape != action.shape:
        raise ValueError("Action, validity and proprio must have equal [H,23] shapes")
    if not np.array_equal(proprio, np.broadcast_to(proprio[0], proprio.shape)):
        raise ValueError("Proprio must repeat only the chunk's conditioning state")
    if not valid[0, 16:23].all() or not np.isfinite(proprio[0, 16:23]).all():
        raise ValueError("Invalid first camera: cannot anchor the action chunk")
    origin = proprio[0, 16:19].astype(np.float64)
    inverse_q = _unit(proprio[0, 19:23].astype(np.float64)).copy()
    inverse_q[1:] *= -1
    out = np.zeros(action.shape, dtype=np.float32)
    state = np.zeros(23, dtype=np.float32)
    for start, width in ((0, 8), (8, 8), (16, 7)):
        block_mask = valid[:, start:start + width]
        if np.any(block_mask != block_mask[:, :1]):
            raise ValueError("EgoStandard validity must cover entire pose/gripper blocks")
        rows = block_mask[:, 0]
        if start < 16 and not rows[0] and rows.any():
            raise ValueError("Hand chunk must be masked when its initial state is invalid")

        def transform(poses: np.ndarray) -> np.ndarray:
            poses = np.asarray(poses, dtype=np.float64)
            if not np.isfinite(poses).all():
                raise ValueError("Nonfinite unmasked EgoStandard pose")
            result = poses.copy()
            result[..., :3] = _rotate(inverse_q, poses[..., :3] - origin)
            q = _unit(_multiply(inverse_q, _unit(poses[..., 3:7])))
            # Deterministic sign at the start; temporal continuity thereafter.
            for i in range(len(q)):
                if (i == 0 and q[i, np.argmax(np.abs(q[i]))] < 0) or (
                    i > 0 and np.dot(q[i - 1], q[i]) < 0
                ):
                    q[i] *= -1
            result[..., 3:7] = q
            return result.astype(np.float32)

        out[rows, start:start + width] = transform(action[rows, start:start + width])
        if rows[0]:
            state[start:start + width] = transform(proprio[:1, start:start + width])[0]
    # Define the observed camera identity exactly, avoiding roundoff in state.
    state[16:23] = (0, 0, 0, 1, 0, 0, 0)
    return out, valid.copy(), np.repeat(state[None], len(action), axis=0)
