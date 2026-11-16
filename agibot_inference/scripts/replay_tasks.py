#!/usr/bin/env python3
"""Task/episode selection for real's G2 replay. Hardware only after --execute."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'configs/replay_tasks.json'
ALIASES = {'衣服': 'yifu', '晾衣服': 'yifu', '滴管': 'diguan', '滴管移液': 'diguan',
           '灯泡': 'dengpao', '拧灯泡': 'dengpao', '饮品': 'drink', '制作饮料': 'drink'}


def resolve(task, episode, registry):
    task = ALIASES.get(task, task)
    if task not in registry['tasks']:
        raise ValueError('任务必须为 yifu / diguan / dengpao / drink')
    entries = registry['tasks'][task]['episodes']
    if episode.isdecimal():
        entry = next((e for e in entries if e['index'] == int(episode)), None)
    else:
        entry = next((e for e in entries if e['name'] == episode or e['path'] == episode), None)
        if entry is None and ('/' in episode):
            supplied = Path(episode).expanduser().resolve()
            entry = next((e for e in entries if Path(e['path']).resolve() == supplied), None)
    if entry is None:
        raise ValueError(f'{task} 没有登记 episode {episode}；用 --task {task} --list 查看')
    path = Path(entry['path'])
    if not path.is_dir():
        raise ValueError(f'录制目录不存在: {path}')
    return task, entry


def joint_stream(episode):
    from replay import BODY_NAMES, JOINT_NAMES, read_pbdat
    from replay_feedback import source_stamp
    from genie_msgs_pb.msg.JointState_pb2 import JointState
    names = BODY_NAMES + JOINT_NAMES
    stamps, values = [], []
    for _, msg in read_pbdat(Path(episode) / 'record/-hal-joint_state.pbdat', JointState):
        if len(msg.name) != len(set(msg.name)) or len(msg.name) != len(msg.motor_position):
            raise ValueError('Body/head/arm feedback joint schema mismatch')
        by_name = dict(zip(msg.name, msg.motor_position))
        row = [by_name[n] for n in names]
        stamp = source_stamp(msg.header)
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite body/head/arm feedback')
        if stamps and not 0 < stamp - stamps[-1] <= 100_000_000:
            raise ValueError(f'Body/head/arm feedback gap or reversed clock at {stamp}')
        stamps.append(stamp)
        values.append(row)
    if len(stamps) < 2:
        raise ValueError('Insufficient body/head/arm feedback')
    return np.array(stamps, dtype=np.int64), np.array(values)


def align_joints(stamps, values, grid_ns):
    from replay_feedback import EDGE_TOLERANCE_NS
    edge = max(0, int(stamps[0]) - int(grid_ns[0]), int(grid_ns[-1]) - int(stamps[-1]))
    if edge > EDGE_TOLERANCE_NS:
        raise ValueError(f'Body/head/arm coverage missing {edge / 1e6:.3f} ms')
    # Subtract integer origin before converting to float: retain nanosecond precision.
    origin = int(grid_ns[0])
    source = (stamps - origin).astype(float) / 1e9
    grid = (np.asarray(grid_ns, dtype=np.int64) - origin).astype(float) / 1e9
    return np.column_stack([np.interp(grid, source, values[:, i]) for i in range(values.shape[1])]), edge


def prepare_raw(episode, task, urdf=ROOT / 'assets/robot/agibot.urdf'):
    from replay import head_target
    from replay_feedback import load_full_paths, load_feedback, align_feedback
    from replay_full_eef import preflight, validate_head_path, validate_body_path, validate_arm_base_paths
    from replay_public_eef import HeadKinematics, reexpress_arm_paths
    paths = load_full_paths(episode)
    preflight(paths)  # Keep existing TF timing and left/right skew checks.
    grid_ns = [round(point[0] * 1e9) for point in paths['left']]
    hands, hand_report = align_feedback(load_feedback(episode), grid_ns)
    stamps, values = joint_stream(episode)
    joints, edge = align_joints(stamps, values, grid_ns)
    body = joints[:, :5].copy()
    raw_head = joints[:, 5:8]
    head = np.array([head_target(tuple(row)) for row in raw_head])
    if np.max(np.abs(np.diff(head, axis=0))) > .35:
        raise ValueError('Head trajectory exceeds existing 0.35 rad step convention')
    # The existing drink executor locks body joints 1..4 and controls only yaw.
    locked = body[:, :4] if task == 'drink' else body
    locked_motion = float(np.max(np.abs(locked - locked[0])))
    if locked_motion > .02:
        raise ValueError(f'Recorded locked body joints move {locked_motion:.6f} rad; '
                         'cannot replay them as fixed joints')
    body[:, :4] = body[0, :4]
    if task != 'drink':
        body[:, 4] = body[0, 4]
    body_path = None
    arm_base = None
    fk = HeadKinematics(urdf)
    for i, name in enumerate(__import__('replay').BODY_NAMES):
        lo, hi = fk.nodes[name][5]
        if np.any(body[:, i] < lo - 1e-3) or np.any(body[:, i] > hi + 1e-3):
            raise ValueError('Recorded body path outside URDF bounds: ' + name)
    # The raw source already contains physical head qpos: do not reinterpret it as EEF.
    if np.any(head < fk.bounds[:, 0] - 1e-3) or np.any(head > fk.bounds[:, 1] + 1e-3):
        raise ValueError('Recorded head qpos outside URDF bounds')
    if task == 'drink':
        body_path = list(map(tuple, body))
        arm_base = reexpress_arm_paths(paths, body, fk)
    first_joints = tuple(np.r_[head[0], joints[0, 8:]])
    head_path = list(map(tuple, head))
    validate_head_path(paths, first_joints, head_path)
    validate_body_path(paths, tuple(body[0]), body_path)
    if body_path is not None:
        validate_arm_base_paths(paths, arm_base)
    duration = paths['left'][-1][0] - paths['left'][0][0]
    report = dict(source='raw_measured_feedback', source_directory=str(episode),
                  replay_frames=len(head), duration_s=duration,
                  replay_hz=(len(head) - 1) / duration,
                  head_range_deg=np.rad2deg(np.ptp(head, axis=0)).tolist(),
                  body_range_deg=np.rad2deg(np.ptp(body, axis=0)).tolist(),
                  waist_yaw_enabled=task == 'drink', locked_body_max_motion_rad=locked_motion,
                  joint_edge_hold_ms=edge / 1e6, hands=hand_report,
                  head_source='HAL motor_position; named qpos', arms_source='base_link TF',
                  hands_source='Sharpa JointState.position',
                  source_is_cleaned_training_episode=False, physical_replay_run=False)
    return dict(paths=paths, body_q=tuple(body[0]), joint_q=first_joints, hand_path=hands,
                head_path=head_path, body_path=body_path, arm_base_paths=arm_base, report=report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task')
    p.add_argument('--episode')
    p.add_argument('--list', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    registry = json.loads(REGISTRY.read_text())
    task = ALIASES.get(args.task, args.task)
    if args.list and task is None:
        for name, cfg in registry['tasks'].items():
            print(f"{name}\t{len(cfg['episodes'])} 条；bash replay.sh --task {name} --list 查看编号")
        return
    if task not in registry['tasks']:
        p.error('任务必须为 yifu / diguan / dengpao / drink')
    if args.list:
        for entry in registry['tasks'][task]['episodes']:
            print(f"{task}\t{entry['index']}\t{entry['format']}\t{entry['name']}\t{entry['path']}")
        return
    if args.episode is None:
        p.error('必须指定 --episode 编号或录制名')
    task, entry = resolve(task, args.episode, registry)
    print(f"task={task}, episode={entry['index']}, format={entry['format']}, path={entry['path']}", flush=True)
    if entry['format'] == 'public71':
        from replay_public_eef import prepare
        plan = prepare(entry['path'], source='state')
    else:
        plan = prepare_raw(Path(entry['path']), task)
    plan['report'].update(task=task, episode_index=entry['index'], source_directory=entry['path'])
    print(json.dumps(plan['report'], ensure_ascii=False, indent=2), flush=True)
    if args.execute:
        from replay_full_eef import execute, HAND_CONFIG
        from replay_feedback import control_locks
        with control_locks():
            execute(plan['paths'], plan['body_q'], plan['joint_q'], plan['hand_path'], HAND_CONFIG,
                    head_path=plan['head_path'], body_path=plan.get('body_path'),
                    arm_base_paths=plan.get('arm_base_paths'))
    else:
        print('离线检查通过；未初始化GDK、未连接双手、未启动真机。', flush=True)


if __name__ == '__main__':
    main()
