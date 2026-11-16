"""Task-specific first measured hand pose. Import/loading never starts hardware."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from agibot_tasks import SHARPA_JOINT_NAMES

DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/'configs/hand_reset_episodes.json'


@dataclass(frozen=True)
class HandReset:
    task: str
    episode: str
    target: tuple
    source_stamps_ns: tuple
    first_message_sha256: tuple


def load_hand_reset(task, config_path=DEFAULT_CONFIG):
    """Validate both first feedback samples BEFORE any body/hand movement.

    Read position (actual SDK angle), never motor_position, target or prediction.
    Each side uses its own original first sample, without cropping/interpolation.
    """
    from robot_wrapper import read_pbdat
    from genie_msgs_pb.msg.JointState_pb2 import JointState
    config=json.loads(Path(config_path).read_text())
    if task not in config:
        raise ValueError(f'No approved hand reset recording for task {task!r}')
    episode=Path(config[task])
    if not episode.is_absolute():
        raise ValueError('Hand reset episode must be an absolute path')
    targets=[];stamps=[];hashes=[]
    for side in ('left','right'):
        path=episode/'record'/f'-sharpa-{side}-state.pbdat'
        reader=read_pbdat(path,JointState)
        try:
            _,msg=next(reader)
        finally:
            reader.close()
        if (len(msg.name)!=22 or len(set(msg.name))!=22 or len(msg.position)!=22
                or set(msg.name)!=set(SHARPA_JOINT_NAMES)):
            raise ValueError(f'{side}: first feedback must have 22 named Sharpa joints: {path}')
        values=dict(zip(msg.name,msg.position))
        target=tuple(float(values[n]) for n in SHARPA_JOINT_NAMES)
        stamp=int(msg.header.stamp.sec)*1_000_000_000+int(msg.header.stamp.nanosec)
        if not all(map(math.isfinite,target)) or stamp<=0:
            raise ValueError(f'{side}: invalid first hand feedback: {path}')
        targets.append(target);stamps.append(stamp)
        hashes.append(hashlib.sha256(msg.SerializeToString()).hexdigest())
    return HandReset(task,str(episode),tuple(targets),tuple(stamps),tuple(hashes))


def reset_hands_to_recording(hands, reset, *, check):
    """Reuse replay's >=3s / <=0.3rad/s ramp, 1s hold, 0.1rad arrival check."""
    from replay_feedback import move_hands_to_first
    if not isinstance(reset,HandReset):
        raise ValueError('Validated recorded hand reset required')
    report=dict(task=reset.task,episode=reset.episode,source='first JointState.position',
                source_stamps_ns=reset.source_stamps_ns,
                first_message_sha256=reset.first_message_sha256,
                target_qpos_rad=reset.target,tolerance_rad=.1)
    print('HAND_RESET_START '+json.dumps(report,ensure_ascii=False),flush=True)
    try:
        check()
        move_hands_to_first(hands,reset.target,check)
        check()
        actual=hands.read_qpos()
        if (len(actual)!=2 or any(len(q)!=22 for q in actual)
                or not all(math.isfinite(v) for q in actual for v in q)):
            raise RuntimeError('Invalid hand feedback after reset')
        errors={}
        for side,q,target in zip(('left','right'),actual,reset.target):
            residual=[abs(a-b) for a,b in zip(q,target)]
            worst=max(range(22),key=residual.__getitem__)
            errors[side]=dict(max_error_rad=residual[worst],joint=SHARPA_JOINT_NAMES[worst])
        if any(x['max_error_rad']>.1 for x in errors.values()):
            raise RuntimeError('Hand reset not reached: '+json.dumps(errors))
        report['feedback']=errors
        print('HAND_RESET_READY '+json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException as exc:
        print('HAND_RESET_FAILED '+json.dumps(dict(task=reset.task,episode=reset.episode,
                                                 error=str(exc)),ensure_ascii=False),flush=True)
        raise
    return report
