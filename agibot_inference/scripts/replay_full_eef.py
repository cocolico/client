#!/usr/bin/env python3
"""Replay complete dual-arm EEF and measured Sharpa qpos. Dry-run by default.

The original CLI only moves body/head to the first frame. The optional
head_path API, used by replay_public_eef.py, also replays the complete head
trajectory through the existing factory combined head/arm sender.
Old gripper targets and Sharpa command topics are never loaded or commanded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from genie_msgs_pb.msg.JointState_pb2 import JointState
from controller_diagnostics import ControllerFault
from replay import BODY_NAMES, JOINT_NAMES, head_target, move_to_first_frame, read_pbdat, read_positions
from replay_eef import FRAMES, degrees, live_pose
from replay_feedback import load_full_paths, load_feedback, align_feedback, control_locks, move_hands_to_first

DATA_ROOT = Path('/home/descfly/code/egoav/pico/data/raw_data/1015拧灯泡')
HAND_CONFIG = Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json'
HEAD_TRACKING_LIMIT_DEG = 15.0
HEAD_TRACKING_LIMIT_RAD = math.radians(HEAD_TRACKING_LIMIT_DEG)
ARM_TRACKING_POSITION_LIMIT_M = 0.10
ARM_TRACKING_ORIENTATION_LIMIT_DEG = 30.0


def resolve_episode(path, name, root=DATA_ROOT):
    if (path is None) == (name is None):
        raise ValueError('Supply either --episode NAME or a positional episode path, not both')
    if name is not None:
        if not name or Path(name).name != name or name in ('.','..'):
            raise ValueError('--episode must be one directory name under the data root')
        root = root.resolve()
        path = (root/name).resolve(strict=True)
        if path.parent != root:
            raise ValueError('--episode resolves outside the data root')
    else:
        path = Path(path).expanduser().resolve(strict=True)
    if not (path/'record').is_dir():
        raise ValueError(f'No record/ directory in {path}')
    return path


def first_joint_frame(episode: Path):
    record = episode / 'record/-hal-joint_state.pbdat'
    _, msg = next(read_pbdat(record, JointState))
    if len(msg.name)!=len(msg.motor_position) or len(set(msg.name))!=len(msg.name):
        raise ValueError('Invalid first-frame joint schema')
    state = dict(zip(msg.name, msg.motor_position))
    if not all(math.isfinite(state[n]) for n in BODY_NAMES+JOINT_NAMES):
        raise ValueError('Nonfinite first-frame body/head/arm feedback')
    return tuple(state[n] for n in BODY_NAMES), head_target(tuple(state[n] for n in JOINT_NAMES))


def preflight(paths):
    if len(paths['left']) != len(paths['right']):
        raise ValueError('Left and right TF lengths differ')
    for side, path in paths.items():
        intervals = [path[i][0]-path[i-1][0] for i in range(1,len(path))]
        steps = [math.dist(path[i][1],path[i-1][1]) for i in range(1,len(path))]
        rotations = [degrees(path[i][2],path[i-1][2]) for i in range(1,len(path))]
        if not 0.005 <= min(intervals) <= max(intervals) <= 0.03:
            raise ValueError(f'{side}: invalid replay clock')
        print(f'{side}: {len(path)} poses, {path[-1][0]-path[0][0]:.2f}s, '
              f'max step {max(steps):.4f}m/{max(rotations):.2f}deg', flush=True)
    skew = max(abs(a[0]-b[0]) for a,b in zip(paths['left'], paths['right']))
    if skew > 0.005:
        raise ValueError(f'Dual-arm timestamp skew {skew:.4f}s')


def assert_ready(robot):
    status = robot.get_motion_control_status()
    if status.error_code or status.collision_pairs_1 or status.collision_pairs_2:
        raise ControllerFault(status, robot)


def validate_arm_tracking(side, position_error_m, orientation_error_deg):
    if (position_error_m > ARM_TRACKING_POSITION_LIMIT_M
            or orientation_error_deg > ARM_TRACKING_ORIENTATION_LIMIT_DEG):
        raise RuntimeError(
            f'{side} tracking error {position_error_m:.3f}m/{orientation_error_deg:.1f}deg')


def set_pose(pose, point):
    _, p, q = point
    pose.position.x, pose.position.y, pose.position.z = p
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q


def validate_head_path(paths, joint_q, head_path):
    if head_path is None:
        return
    if (len(head_path) != len(paths['left']) or len(joint_q) != 17
            or any(len(q) != 3 or not all(map(math.isfinite,q)) for q in head_path)):
        raise ValueError('Head path must contain one finite 3-joint target per replay frame')
    if any(tuple(q) != head_target(tuple(q)) for q in head_path):
        raise ValueError('Head path must already use the existing head target bounds')
    if max(abs(a-b) for a,b in zip(head_path[0],joint_q[:3])) > 1e-6:
        raise ValueError('Head trajectory first frame differs from startup target')


def validate_body_path(paths, body_q, body_path):
    if body_path is None:return
    if (len(body_path)!=len(paths['left']) or len(body_q)!=5
            or any(len(q)!=5 or not all(map(math.isfinite,q)) for q in body_path)):
        raise ValueError('Body path requires five finite joints per replay frame')
    if max(abs(a-b) for a,b in zip(body_path[0],body_q))>1e-6:
        raise ValueError('Body trajectory first frame differs from startup target')
    if any(abs(a-b)>.35 for prev,row in zip(body_path,body_path[1:]) for a,b in zip(prev,row)):
        raise ValueError('Body trajectory step exceeds 0.35rad')


def validate_arm_base_paths(paths, arm_base_paths):
    if arm_base_paths is None:
        raise ValueError('Dynamic body replay requires arm_base_link EEF paths')
    for side in ('left','right'):
        if side not in arm_base_paths or len(arm_base_paths[side]) != len(paths[side]):
            raise ValueError('arm_base_link EEF paths must align with both arms')
        for source,target in zip(paths[side],arm_base_paths[side]):
            if (len(target) != 3 or target[0] != source[0] or len(target[1]) != 3
                    or len(target[2]) != 4 or not all(map(math.isfinite,target[1]+target[2]))):
                raise ValueError('Invalid arm_base_link EEF path')


def send_teleop_replay(publisher, gdk, left, right, head, waist, tick):
    """Publish one native-teleop-style sample through the priority-100 route."""
    from genie_msgs_pb.msg.ReactiveControl_pb2 import ReactiveControl
    from genie_msgs_pb.msg.RetargetInfo_pb2 import RetargetInfo
    if len(head) != 3 or len(waist) != 5 or not all(map(math.isfinite,tuple(head)+tuple(waist))):
        raise ValueError('Teleop replay needs finite head and five-joint waist targets')
    msg=ReactiveControl(lifetime=.2,control_period=.01)
    stamp=time.time_ns()
    msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(stamp,10**9)
    msg.header.frame_id='arm_base_link'
    for group,name,point in ((RetargetInfo.GROUP_LEFT_ARM,'arm_l_end_link',left),
                             (RetargetInfo.GROUP_RIGHT_ARM,'arm_r_end_link',right)):
        part=msg.retarget_groups.add(group_id=group,input_type=RetargetInfo.INPUT_VR,
            control_type=RetargetInfo.ABS_POSE,frame_id='arm_base_link')
        part.target_frame_names.append(name)
        set_pose(part.target_frame_poses.add(),point)
    for group,value in zip((RetargetInfo.GROUP_HEAD_YAW,RetargetInfo.GROUP_HEAD_ROLL,
                            RetargetInfo.GROUP_HEAD_PITCH),head):
        part=msg.retarget_groups.add(group_id=group,input_type=RetargetInfo.INPUT_VR,
            control_type=RetargetInfo.ABS_JOINT,frame_id='base_link')
        part.target_joint_positions.append(float(value))
    part=msg.retarget_groups.add(group_id=RetargetInfo.GROUP_WAIST,
        input_type=RetargetInfo.INPUT_VR,control_type=RetargetInfo.ABS_JOINT,
        frame_id='base_link')
    part.target_joint_positions.extend(map(float,waist))
    info=gdk.PubMsgInfo();info.source_ts_ns=stamp
    if not publisher.publish(msg,info):
        raise RuntimeError(f'Frame {tick}: teleop replay command failed')


def execute(paths, body_q, joint_q, hand_path, config, *, head_path=None, body_path=None,
            arm_base_paths=None):
    validate_head_path(paths,joint_q,head_path)
    validate_body_path(paths,body_q,body_path)
    if body_path is not None and head_path is None:
        raise ValueError('Dynamic body replay requires synchronized head_path')
    if body_path is not None:
        validate_arm_base_paths(paths,arm_base_paths)
    from sharpa_hands import SharpaHands
    hands = SharpaHands.from_config(config)
    try:
        hands.connect()
        if body_path is not None:
            execute_connected(paths,body_q,joint_q,hand_path,hands,head_path=head_path,
                              body_path=body_path,arm_base_paths=arm_base_paths)
        elif head_path is None:
            execute_connected(paths,body_q,joint_q,hand_path,hands)
        else:
            execute_connected(paths,body_q,joint_q,hand_path,hands,head_path=head_path)
    finally:
        hands.close()


def execute_connected(paths, body_q, joint_q, hand_path, hands, *, head_path=None, body_path=None,
                      arm_base_paths=None):
    validate_head_path(paths,joint_q,head_path)
    validate_body_path(paths,body_q,body_path)
    if body_path is not None and head_path is None:
        raise ValueError('Dynamic body replay requires synchronized head_path')
    if body_path is not None:
        validate_arm_base_paths(paths,arm_base_paths)
    import agibot_gdk as gdk
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError('GDK init failed')
    robot, tf = gdk.Robot(), gdk.TF()
    combined = None
    teleop = None
    if body_path is not None:
        from genie_msgs_pb.msg.ReactiveControl_pb2 import ReactiveControl
        teleop=gdk.Publisher('/egoav/teleop/retarget',ReactiveControl.DESCRIPTOR.full_name,gdk.QoS())
    elif head_path is not None:
        # Reuse the existing factory combined route. Separate head/arm requests
        # compete in the factory arbitrator. Legacy replay without head_path
        # continues to use its original EndEffectorPose channel below.
        from types import SimpleNamespace
        from robot_wrapper import AgiBotRobot, TrajectoryTrackingControl
        context=SimpleNamespace(gdk=gdk,_eef_head_publisher=gdk.Publisher(
            '/gdk/model_predict',TrajectoryTrackingControl.DESCRIPTOR.full_name,gdk.QoS()))
        def combined(left,right,head,index):
            target=list(left[1])+list(left[2])+list(right[1])+list(right[2])+list(head)
            waist_q = None
            AgiBotRobot._send_combined_eef_head(context,target,index,waist_q=waist_q)
    time.sleep(1)
    assert_ready(robot)
    current_body = read_positions(robot, BODY_NAMES)
    body_delta = max(abs(a-b) for a,b in zip(current_body,body_q))
    print(f'复位躯干，最大关节变化 {body_delta:.3f}rad', flush=True)
    if body_delta > 1.0:
        raise RuntimeError('Body reset exceeds 1 rad')
    req = gdk.JointControlReq()
    req.joint_names, req.joint_positions = list(BODY_NAMES), list(body_q)
    req.joint_velocities = [0.10] * len(BODY_NAMES)
    req.life_time = max(12.0, 2*body_delta/0.10+5)
    if robot.joint_control_request(req) != 0:
        raise RuntimeError('Body reset failed')
    if max(abs(a-b) for a,b in zip(read_positions(robot,BODY_NAMES),body_q)) > 0.05:
        raise RuntimeError('Body did not reach recorded first frame')
    waist_lock=tuple(read_positions(robot,BODY_NAMES)[:4])
    assert_ready(robot)
    move_to_first_frame(robot, joint_q)  # head + arms, no old grippers
    assert_ready(robot)
    for side in ('left','right'):
        p,q = live_pose(tf,FRAMES[side])
        target = paths[side][0]
        dp,da = math.dist(p,target[1]), degrees(q,target[2])
        print(f'{side} 起点 EEF 误差 {dp:.4f}m/{da:.2f}deg',flush=True)
        if dp > 0.03 or da > 12:
            raise RuntimeError(f'{side} first-frame EEF mismatch; no path replay')
    move_hands_to_first(hands,hand_path[0],lambda:assert_ready(robot))
    print('起点核对通过，开始双臂 EEF + 双手 qpos 同步回放'
          + (' + 完整头部轨迹（联合发送）' if head_path is not None else '')
          + (' + 仅腰部旋转关节轨迹' if body_path is not None else '')
          + '；Ctrl+C 停止发新目标',flush=True)
    t0 = paths['left'][0][0]
    started = time.monotonic()
    for i,(left,right) in enumerate(zip(paths['left'],paths['right'])):
        delay = started+(left[0]-t0)-time.monotonic()
        if delay < -0.1:
            raise RuntimeError(f'Frame {i}: deadline missed by {-delay:.3f}s')
        time.sleep(max(0,delay))
        if i % 10 == 0:
            try:
                assert_ready(robot)
            except RuntimeError as exc:
                raise RuntimeError(f'Frame {i}, t={left[0]-t0:.2f}s: {exc}') from exc
            for side,point in (('left',left),('right',right)):
                actual_p,actual_q = live_pose(tf,FRAMES[side])
                dp,da = math.dist(actual_p,point[1]),degrees(actual_q,point[2])
                try:
                    validate_arm_tracking(side,dp,da)
                except RuntimeError as exc:
                    raise RuntimeError(f'Frame {i} {exc}') from exc
            if head_path is not None:
                actual=read_positions(robot,JOINT_NAMES[:3])
                error=max(abs(a-b) for a,b in zip(actual,head_path[i]))
                if error > HEAD_TRACKING_LIMIT_RAD:
                    raise RuntimeError(
                        f'Frame {i}: head tracking error {error:.4f}rad exceeds '
                        f'{HEAD_TRACKING_LIMIT_DEG:.1f}deg')
            if i % 500 == 0:
                print(f'frame {i}/{len(paths["left"])}',flush=True)
        if teleop is not None:
            send_teleop_replay(teleop,gdk,arm_base_paths['left'][i],
                               arm_base_paths['right'][i],head_path[i],
                               waist_lock+(body_path[i][4],),i)
        elif combined is None:
            command = gdk.EndEffectorPose()
            command.group = gdk.EndEffectorControlGroup.kBothArms
            command.life_time = 0.05
            set_pose(command.left_end_effector_pose,left)
            set_pose(command.right_end_effector_pose,right)
            if robot.end_effector_pose_control(command) != 0:
                raise RuntimeError(f'Frame {i}: dual-arm EEF command failed')
        else:
            combined(left,right,head_path[i],i)
        # Same frame clock as EEF; no independent hand loop that could continue
        # after an arm failure. SDK exceptions terminate both streams.
        hands.send_qpos(*hand_path[i],frame_index=i)
    time.sleep(0.2)
    assert_ready(robot)
    if head_path is not None:
        actual=read_positions(robot,JOINT_NAMES[:3])
        error=max(abs(a-b) for a,b in zip(actual,head_path[-1]))
        print(f'头部终点关节误差 {error:.6f}rad',flush=True)
        if error > HEAD_TRACKING_LIMIT_RAD:
            raise RuntimeError(
                f'Head final tracking error {error:.4f}rad exceeds '
                f'{HEAD_TRACKING_LIMIT_DEG:.1f}deg')
    for side in ('left','right'):
        p,q=live_pose(tf,FRAMES[side])
        target=paths[side][-1]
        print(f'{side} 终点 EEF 误差 {math.dist(p,target[1]):.4f}m/{degrees(q,target[2]):.2f}deg',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?',type=Path,help='legacy full episode path')
    parser.add_argument('--episode',help=f'episode directory name under {DATA_ROOT}')
    parser.add_argument('--hands-config',type=Path,default=HAND_CONFIG)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    episode=resolve_episode(args.path,args.episode)
    print(f'数据: {episode}',flush=True)
    paths=load_full_paths(episode)
    preflight(paths)
    hand_path,report=align_feedback(load_feedback(episode),[round(p[0]*1e9) for p in paths['left']])
    body_q,joint_q=first_joint_frame(episode)
    print('双手实际反馈对齐: '+json.dumps(report,ensure_ascii=False),flush=True)
    print(f'首帧躯干: {body_q}; 头+臂关节数: {len(joint_q)}; 双手 qpos: 22+22',flush=True)
    print('头/躯干只恢复首帧；回放双臂与双手。默认预检不连接硬件。',flush=True)
    if args.execute:
        with control_locks():
            execute(paths,body_q,joint_q,hand_path,args.hands_config)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('已停止发送新目标',flush=True)
