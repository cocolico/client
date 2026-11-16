#!/usr/bin/env python3
"""只回放 GDK 双臂的末端位姿变化，不控制 Sharpa 手或旧夹爪。

默认只解析 TF 并预检。--execute 将录制片段相对首帧的位姿变化叠加到
机器人当前 base_link 下的末端位姿，避免跳往旧机器人录制时的绝对首帧。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import time

from tf2_msgs_pb.msg.TFMessage_pb2 import TFMessage

FRAMES = {'left': 'arm_l_end_link', 'right': 'arm_r_end_link'}
GROUPS = {'left': 4, 'right': 8}
RATE_HZ = 100.0


def normalized(q: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x*x for x in q))
    if not math.isfinite(norm) or abs(norm-1) > 0.02:
        raise ValueError(f'Invalid quaternion norm: {norm}')
    return tuple(x/norm for x in q)


def inverse(q: tuple[float, ...]) -> tuple[float, ...]:
    x,y,z,w = q
    return (-x,-y,-z,w)


def multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    x1,y1,z1,w1 = a
    x2,y2,z2,w2 = b
    return normalized((w1*x2+x1*w2+y1*z2-z1*y2,
                       w1*y2-x1*z2+y1*w2+z1*x2,
                       w1*z2+x1*y2-y1*x2+z1*w2,
                       w1*w2-x1*x2-y1*y2-z1*z2))


def degrees(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.degrees(2*math.acos(min(1.0,abs(sum(x*y for x,y in zip(a,b))))))


def load(episode: Path, side: str, seconds: float) -> list[tuple[float,tuple[float,...],tuple[float,...]]]:
    path = episode/'record/-tf.pbdat'
    target = FRAMES[side]
    out = []
    with path.open('rb') as f:
        header = f.read(8)
        if len(header)!=8:raise ValueError('TF file has no frame count')
        count, = struct.unpack('<q',header)
        for _ in range(count):
            frame_header=f.read(40)
            if len(frame_header)!=40:raise ValueError('Truncated TF frame header')
            size=struct.unpack('<5q',frame_header)[4]
            if not 0<size<=2_000_000:raise ValueError('Invalid TF payload length')
            raw=f.read(size)
            if len(raw)!=size:raise ValueError('Truncated TF payload')
            msg=TFMessage.FromString(raw)
            for transform in msg.transforms:
                if transform.child_frame_id!=target or transform.header.frame_id!='base_link':
                    continue
                stamp=transform.header.stamp
                t=stamp.sec+stamp.nanosec*1e-9
                v=transform.transform.translation
                r=transform.transform.rotation
                pos=(v.x,v.y,v.z)
                quat=normalized((r.x,r.y,r.z,r.w))
                if not all(math.isfinite(x) for x in pos):raise ValueError('Non-finite TF position')
                if out and t<=out[-1][0]:raise ValueError('TF timestamps not increasing')
                if out and t-out[0][0]>seconds+0.005:break
                out.append((t,pos,quat))
                break
            if out and out[-1][0]-out[0][0]>=seconds-0.005:break
    if len(out)<3 or out[-1][0]-out[0][0]<seconds-0.02:
        raise ValueError('Insufficient EEF TF frames for requested segment')
    return out


def check_path(path, max_translation: float, max_rotation_deg: float) -> dict:
    start=path[0]
    max_d=max(math.dist(start[1],x[1]) for x in path)
    max_a=max(degrees(start[2],x[2]) for x in path)
    max_step=max(math.dist(path[i-1][1],path[i][1]) for i in range(1,len(path)))
    max_step_a=max(degrees(path[i-1][2],path[i][2]) for i in range(1,len(path)))
    intervals=[path[i][0]-path[i-1][0] for i in range(1,len(path))]
    if max_d>max_translation or max_a>max_rotation_deg:
        raise ValueError(f'Segment exceeds limits: {max_d:.4f}m, {max_a:.1f}deg')
    if max_step>0.003 or max_step_a>1.5 or min(intervals)<0.005 or max(intervals)>0.03:
        raise ValueError(f'TF discontinuity: step {max_step:.4f}m/{max_step_a:.1f}deg, dt {min(intervals):.3f}..{max(intervals):.3f}s')
    return {'samples':len(path),'seconds':path[-1][0]-start[0],
            'max_translation_m':max_d,'max_rotation_deg':max_a,
            'max_step_m':max_step,'max_step_deg':max_step_a}


def live_pose(tf, frame: str):
    timestamp=tf.get_latest_timestamp(frame)
    age=abs(time.time_ns()-timestamp)/1e9
    if age>0.9:raise RuntimeError(f'Stale live TF: {age:.3f}s')
    transform=tf.get_tf_from_base_link(frame)
    v=transform.translation;r=transform.rotation
    return (v.x,v.y,v.z),normalized((r.x,r.y,r.z,r.w))


def execute(path, side: str) -> None:
    import agibot_gdk as gdk
    if gdk.gdk_init()!=gdk.GDKRes.kSuccess:raise RuntimeError('GDK init failed')
    robot,tf=gdk.Robot(),gdk.TF()
    time.sleep(1)
    status=robot.get_motion_control_status()
    if status.error_code or status.collision_pairs_1 or status.collision_pairs_2:
        raise RuntimeError(f'Motion controller not ready: {status}')
    frame=FRAMES[side]
    p_live,q_live=live_pose(tf,frame)
    p0,q0=path[0][1:]
    # 两个姿态都在 base_link；应用录制片段的 base-frame 位移和旋转。
    targets=[]
    for _,p,q in path:
        pos=tuple(a+b-c for a,b,c in zip(p_live,p,p0))
        quat=multiply(multiply(q,inverse(q0)),q_live)
        targets.append((pos,quat))
    print(f'实机 {side} 当前 EEF={p_live}；相对回放 {len(targets)} 点，Sharpa 手不动',flush=True)
    started=time.monotonic()
    for i,(pos,quat) in enumerate(targets):
        deadline=started+(path[i][0]-path[0][0])
        delay=deadline-time.monotonic()
        if delay < -0.1:raise RuntimeError('EEF replay is over 100ms late; stopped')
        time.sleep(max(0,delay))
        if i%10==0:
            actual_pos,actual_q=live_pose(tf,frame)
            if math.dist(actual_pos,pos)>0.05 or degrees(actual_q,quat)>30:
                raise RuntimeError(f'EEF tracking diverged at {i}: {math.dist(actual_pos,pos):.3f}m/{degrees(actual_q,quat):.1f}deg')
            status=robot.get_motion_control_status()
            if status.error_code or status.collision_pairs_1 or status.collision_pairs_2:
                raise RuntimeError(f'Motion controller fault at {i}: {status}')
        command=gdk.EndEffectorPose()
        command.group=GROUPS[side]
        command.life_time=0.05
        pose=gdk.Pose()
        pose.position.x,pose.position.y,pose.position.z=pos
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=quat
        if side=='left':command.left_end_effector_pose=pose
        else:command.right_end_effector_pose=pose
        if robot.end_effector_pose_control(command)!=0:
            raise RuntimeError(f'EEF command failed at frame {i}')
    time.sleep(0.15)
    actual_pos,actual_q=live_pose(tf,frame)
    final_pos,final_q=targets[-1]
    position_error=math.dist(actual_pos,final_pos)
    rotation_error=degrees(actual_q,final_q)
    print(f'发送完成；末端误差 {position_error:.4f}m / {rotation_error:.2f}deg',flush=True)
    if position_error>0.01 or rotation_error>5:
        raise RuntimeError('EEF trajectory sent, but final pose did not track within 1cm/5deg')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--side',choices=tuple(FRAMES),default='left')
    parser.add_argument('--seconds',type=float,default=1.25)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not 0.1<=args.seconds<=1.5:parser.error('seconds must be 0.1..1.5')
    path=load(args.episode,args.side,args.seconds)
    info=check_path(path,max_translation=0.03,max_rotation_deg=12)
    print(f'TF 预检通过：{info}；绝对首帧={path[0][1]}；默认不连接机器人',flush=True)
    if args.execute:execute(path,args.side)


if __name__=='__main__':
    main()
