"""Offline feedback alignment and guarded replay helpers. No SDK initialization."""
from bisect import bisect_left
from contextlib import contextmanager, ExitStack
import math
import os
from pathlib import Path
import time

from agibot_tasks import SHARPA_JOINT_NAMES
from replay import read_pbdat
from replay_eef import FRAMES, normalized

MAX_HAND_GAP_NS = 100_000_000
EDGE_TOLERANCE_NS = 20_000_000
CONTROL_LOCKS = (
    Path('/home/descfly/code/egoav/pico/.all_teleop.lock'),
    Path('/home/descfly/code/egoav/sharpa/.hands_control.lock'),
)


def source_stamp(header):
    stamp = header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9:
        raise ValueError('Invalid source timestamp')
    result = stamp.sec * 10**9 + stamp.nanosec
    if result <= 0:
        raise ValueError('Missing source timestamp')
    return result


def load_full_paths(episode):
    """Read the complete absolute base_link EEF stream; never cap at 36 s."""
    from tf2_msgs_pb.msg.TFMessage_pb2 import TFMessage
    paths = {side: [] for side in FRAMES}
    for _, msg in read_pbdat(Path(episode)/'record/-tf.pbdat', TFMessage):
        for transform in msg.transforms:
            for side, frame in FRAMES.items():
                if transform.child_frame_id != frame or transform.header.frame_id != 'base_link':
                    continue
                stamp = source_stamp(transform.header)
                t = stamp / 1e9
                v, r = transform.transform.translation, transform.transform.rotation
                p = (v.x, v.y, v.z)
                if not all(math.isfinite(x) for x in p):
                    raise ValueError(f'{side}: nonfinite EEF position')
                q = normalized((r.x, r.y, r.z, r.w))
                if paths[side] and t <= paths[side][-1][0]:
                    raise ValueError(f'{side}: EEF source time not increasing')
                paths[side].append((t, p, q))
    if any(len(path) < 3 for path in paths.values()):
        raise ValueError('Insufficient dual-arm TF samples')
    return paths


def load_feedback(episode):
    """Use JointState.position, not motor_position or the target topics."""
    from genie_msgs_pb.msg.JointState_pb2 import JointState
    streams = {}
    for side in ('left', 'right'):
        rows = []
        for _, msg in read_pbdat(Path(episode)/f'record/-sharpa-{side}-state.pbdat', JointState):
            if (len(msg.name) != 22 or len(set(msg.name)) != 22 or
                    set(msg.name) != set(SHARPA_JOINT_NAMES) or len(msg.position) != 22):
                raise ValueError(f'{side}: Sharpa feedback requires the 22 named joints')
            values = dict(zip(msg.name, msg.position))
            qpos = tuple(values[name] for name in SHARPA_JOINT_NAMES)
            if not all(math.isfinite(x) for x in qpos):
                raise ValueError(f'{side}: nonfinite feedback qpos')
            stamp = source_stamp(msg.header)
            if rows:
                gap = stamp - rows[-1][0]
                if gap <= 0:
                    raise ValueError(f'{side}: feedback source time not increasing')
                if gap > MAX_HAND_GAP_NS:
                    raise ValueError(f'{side}: feedback gap {gap/1e6:.3f} ms at {stamp}; no replay')
            rows.append((stamp, qpos))
        if len(rows) < 2:
            raise ValueError(f'{side}: insufficient feedback samples')
        streams[side] = rows
    return streams


def align_feedback(streams, grid_ns):
    """Linear interpolation over short gaps; report small endpoint phase holds."""
    if len(grid_ns) < 2 or any(b <= a for a, b in zip(grid_ns, grid_ns[1:])):
        raise ValueError('Replay grid must be increasing')
    aligned, report = {}, {}
    for side in ('left', 'right'):
        rows = streams[side]
        stamps = [stamp for stamp, _ in rows]
        edge_hold = max(0, stamps[0]-grid_ns[0], grid_ns[-1]-stamps[-1])
        if edge_hold > EDGE_TOLERANCE_NS:
            raise ValueError(f'{side}: hand coverage missing {edge_hold/1e6:.3f} ms; no replay')
        values = []
        for stamp in grid_ns:
            i = bisect_left(stamps, stamp)
            if i == 0:
                qpos = rows[0][1]
            elif i == len(rows):
                qpos = rows[-1][1]
            else:
                alpha = (stamp-stamps[i-1])/(stamps[i]-stamps[i-1])
                qpos = tuple(a+(b-a)*alpha for a,b in zip(rows[i-1][1],rows[i][1]))
            if values and max(abs(a-b) for a,b in zip(values[-1],qpos)) > .35:
                raise ValueError(f'{side}: aligned hand qpos step exceeds 0.35 rad at {stamp}')
            values.append(qpos)
        aligned[side] = values
        report[side] = dict(samples=len(rows), duration_s=(stamps[-1]-stamps[0])/1e9,
                            hz=(len(rows)-1)*1e9/(stamps[-1]-stamps[0]),
                            max_gap_ms=max(b-a for a,b in zip(stamps,stamps[1:]))/1e6,
                            edge_hold_ms=edge_hold/1e6)
    return list(zip(aligned['left'],aligned['right'])), report


@contextmanager
def control_locks(paths=CONTROL_LOCKS):
    """Share existing launcher locks; never kill another owner or unlink a lock."""
    import fcntl
    with ExitStack() as stack:
        for path in paths:
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stream = stack.enter_context(os.fdopen(fd, 'a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Control is owned by teleop/replay: {path}; stop it first') from exc
        yield


def move_hands_to_first(hands, target, check):
    """Measured start, 100 Hz smoothstep, peak entry speed <= 0.3 rad/s."""
    start = hands.read_qpos()
    distance = max(abs(a-b) for x,y in zip(start,target) for a,b in zip(x,y))
    steps = math.ceil(max(3., 1.5*distance/.3)*100)
    print(f'Sharpa 双手平滑进入首帧：{steps/100:.2f}s，最大变化 {distance:.3f}rad',flush=True)
    begun = time.monotonic()
    for i in range(steps+101):
        delay = begun+i*.01-time.monotonic()
        if delay < -.1:
            raise RuntimeError('Hand startup deadline missed by >100 ms')
        time.sleep(max(0.,delay))
        if i % 10 == 0:check()
        u = min(1.,i/steps); weight = u*u*(3-2*u)
        q = [tuple(a+(b-a)*weight for a,b in zip(x,y)) for x,y in zip(start,target)]
        hands.send_qpos(*q,frame_index=-1,subframe_index=i)
    actual = hands.read_qpos()
    error = max(abs(a-b) for x,y in zip(actual,target) for a,b in zip(x,y))
    if error > .1:
        raise RuntimeError(f'Sharpa first-frame error {error:.3f} rad; no replay')
