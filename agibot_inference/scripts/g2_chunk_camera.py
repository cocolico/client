"""G2 coordinate-only adapter; no transport, hardware initialization or executor.

Internal physical71 order: left EEF9, left qpos22, right EEF9, right qpos22,
head EEF9. Rotation6D concatenates the first two matrix columns. This is NOT
a replacement wire contract for the old absolute Mixed76 v4 service.
"""
from dataclasses import dataclass

import numpy as np

from mixed76_contract import encode_state, decode_actions


def _frozen(values, dtype=np.float64):
    x = np.asarray(values, dtype=dtype)
    # Immutable byte backing also prevents callers re-enabling writes.
    return np.frombuffer(x.tobytes(), dtype=dtype).reshape(x.shape)


def pose_matrices(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 9 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty Nx9 XYZ + rotation columns')
    a, b = x[:,3:6].copy(), x[:,6:9].copy()
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate predicted rotation first column')
    a /= norm
    b -= np.sum(a*b, axis=1, keepdims=True)*a
    norm = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate predicted rotation second column')
    b /= norm
    t = np.tile(np.eye(4), (len(x),1,1))
    t[:,:3,:3] = np.stack((a,b,np.cross(a,b)), axis=-1)
    t[:,:3,3] = x[:,:3]
    return t


def pose_columns(matrices):
    t = np.asarray(matrices, dtype=np.float64)
    return np.concatenate((t[:,:3,3], t[:,:3,0], t[:,:3,1]), axis=1)


def transform71(values, target_from_source):
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(target_from_source, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 71 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty Nx71 physical actions')
    if (t.shape != (4,4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0,0,0,1], atol=1e-9, rtol=0)
            or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(t[:3,:3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Expected a rigid coordinate transform')
    result = x.copy()
    for start in (0,31,62):
        result[:,start:start+9] = pose_columns(t[None] @ pose_matrices(x[:,start:start+9]))
    return result


@dataclass(frozen=True)
class ChunkFrame:
    sample_ns: int
    base_from_camera: np.ndarray
    body_target: np.ndarray
    state71: np.ndarray


class G2ChunkCamera:
    """Reuse replay FK/inverse and the existing native61 boundary conversion."""
    def __init__(self, urdf, body_target):
        # No robot class is created. Replay imports protobuf definitions only.
        from replay_public_eef import HeadKinematics
        from replay import BODY_NAMES
        body = np.asarray(body_target, dtype=np.float64)
        if body.shape != (5,) or not np.isfinite(body).all():
            raise ValueError('Explicit fixed body control target (5 radians) required')
        self.fk = HeadKinematics(urdf)
        self.body_names = BODY_NAMES
        for name, value in zip(BODY_NAMES, body):
            low, high = self.fk.nodes[name][5]
            if not low-1e-3 <= value <= high+1e-3:
                raise ValueError('Fixed body control target outside URDF limits')
        self.body_target = _frozen(body)

    def begin(self, native61, *, sample_ns):
        x = np.asarray(native61, dtype=np.float64)
        # Reuse the existing finite/shape/quaternion validation and ordering.
        packed = encode_state(x, 'g2')
        if not isinstance(sample_ns, (int, np.integer)) or sample_ns < 0:
            raise ValueError('Expected nonnegative observation sample_ns')
        anchor = self.fk.head_pose(self.body_target, x[58:61])
        state = np.concatenate((packed[12:21], x[7:29], packed[21:30],
                                x[36:58], pose_columns(anchor[None])[0]))
        relative = transform71(state[None], np.linalg.inv(anchor))[0]
        return ChunkFrame(int(sample_ns), _frozen(anchor), self.body_target,
                          _frozen(relative, np.float32))

    def decode(self, chunk, actions71):
        if not isinstance(chunk, ChunkFrame) or not np.array_equal(chunk.body_target, self.body_target):
            raise ValueError('Chunk context/fixed body target mismatch')
        base = transform71(actions71, chunk.base_from_camera)
        # Project predicted head angles into URDF limits. The shared replay
        # inverse remains strict by default; report any projection explicitly.
        head = self.fk.solve(pose_matrices(base[:,62:71]), self.body_target,
                             position_diagnostic_only=True, clip_joint_limits=True)
        packed = np.zeros((len(base),76), dtype=np.float32)
        packed[:,12:21], packed[:,21:30] = base[:,:9], base[:,31:40]
        packed[:,30:52], packed[:,53:75] = base[:,9:31], base[:,40:62]
        packed[:,9:12] = head
        # Same xyz/xyzw/native61 mapping and quaternion continuity as before.
        native = decode_actions(packed, 'g2')
        report = dict(self.fk.report, sample_ns=chunk.sample_ns,
                      base_from_camera=chunk.base_from_camera.tolist())
        return native, report


def validate_recorded(sample, urdf, *, chunk_size=32):
    """Recorded targets simulate the model output; no server or hardware I/O.

    Uses the independently saved public71 actions and native61 reference, not
    a model prediction. This verifies conversion, NOT model/server behavior.
    """
    import hashlib
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from run_mixed76_client import prepare_g2_observation, restore_g2_actions
    from replay import BODY_NAMES
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 50:
        raise ValueError('Expected chunk size 1..50')
    with np.load(sample, allow_pickle=False) as data:
        states = np.array(data['roundtrip_native_state'])
        expected = np.array(data['roundtrip_native_action'])
        public = np.array(data['public_action'])
        stamps = np.array(data['timestamp_ns'])
        names = list(data['joint_names'])
        body = np.array(data['target_joints_rad'])[:,[names.index(n) for n in BODY_NAMES]]
    count = len(states)
    if (not count or states.shape != (count,61) or expected.shape != (count,61)
            or public.shape != (count,71) or body.shape != (count,5)
            or stamps.shape != (count,) or len(names) != len(set(names))
            or not all(np.isfinite(a).all() for a in (states,expected,public,body,stamps))
            or np.any(np.diff(stamps) <= 0)):
        raise ValueError('Invalid recorded numeric mirror')
    if not np.allclose(body, body[0], atol=1e-6, rtol=0):
        raise ValueError('Recording body control target is not fixed')
    adapter = G2ChunkCamera(urdf, body[0])
    decoded, reports = [], []
    state_error = 0.
    state_rotation_error = 0.
    for start in range(0,count,chunk_size):
        state, context = prepare_g2_observation({'state':states[start]},
            sample_ns=int(stamps[start]), adapter=adapter)
        restored, _ = restore_g2_actions(state[None], context=context, adapter=adapter)
        # q and -q represent the same arm orientation; use angular distance.
        scalar_indices = list(range(3))+list(range(7,32))+list(range(36,61))
        state_error = max(state_error,float(np.max(np.abs(restored[0,scalar_indices]-states[start,scalar_indices]))))
        for i in (0,29):
            angular_error = (Rotation.from_quat(states[start,i+3:i+7]).inv()
                             * Rotation.from_quat(restored[0,i+3:i+7])).magnitude()
            state_rotation_error = max(state_rotation_error,float(angular_error))
        relative = transform71(public[start:start+chunk_size], np.linalg.inv(context.base_from_camera))
        native, report = restore_g2_actions(relative, context=context, adapter=adapter)
        decoded.append(native); reports.append(report)
    decoded = np.concatenate(decoded)
    pos_error = max(float(np.max(np.linalg.norm(decoded[:,i:i+3]-expected[:,i:i+3],axis=1)))
                    for i in (0,29))
    rot_error = max(float(np.max((Rotation.from_quat(expected[:,i+3:i+7]).inv()
                    * Rotation.from_quat(decoded[:,i+3:i+7])).magnitude())) for i in (0,29))
    head_error = float(np.max(np.abs(decoded[:,58:61]-expected[:,58:61])))
    hand_error = max(float(np.max(np.abs(decoded[:,i:i+22]-expected[:,i:i+22]))) for i in (7,36))
    if max(pos_error,rot_error,head_error,hand_error,state_error,state_rotation_error) > 1e-6:
        raise ValueError(f'Recorded conversion mismatch: {pos_error=}, {rot_error=}, '
                         f'{head_error=}, {hand_error=}, {state_error=}, {state_rotation_error=}')
    return dict(frames=count,chunks=len(reports),chunk_size=chunk_size,
        body_target_rad=body[0].tolist(),head_frame='head_link3',
        max_arm_position_error_m=pos_error,max_arm_rotation_error_rad=rot_error,
        max_head_qpos_error_rad=head_error,max_hand_qpos_error_rad=hand_error,
        max_observation_scalar_error=state_error,
        max_observation_rotation_error_rad=state_rotation_error,
        head_max_position_error_mm=max(r['head_max_position_error_mm'] for r in reports),
        sample_sha256=hashlib.sha256(Path(sample).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
        robot_io=False,server_verified=False,
        action_source='recorded public71; not model predictions')
