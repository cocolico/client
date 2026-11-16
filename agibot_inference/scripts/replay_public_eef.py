#!/usr/bin/env python3
"""71D public-frame sample -> existing Agibot replay. Offline unless --execute.

Defaults to measured state, matching the original feedback replay. The immutable
offline_replay.npz is the validated numeric mirror of the Parquet; both hashes
are checked. This avoids adding PyArrow to the existing GDK Python environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from agibot_tasks import SHARPA_JOINT_NAMES
from replay import BODY_NAMES, JOINT_NAMES, head_target
from replay_full_eef import preflight, execute, HAND_CONFIG
from replay_feedback import control_locks

ORDER=['r1x','r1y','r1z','r2x','r2y','r2z']


def rotation6d(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():
        raise ValueError('Expected finite Nx6 rotation columns')
    a,b=x[:,:3],x[:,3:]
    if (not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-6,rtol=0)
            or not np.allclose(np.linalg.norm(b,axis=1),1,atol=1e-6,rtol=0)
            or not np.allclose(np.sum(a*b,axis=1),0,atol=1e-6,rtol=0)):
        raise ValueError('Rotation columns must be orthonormal')
    a=a/np.linalg.norm(a,axis=1,keepdims=True)
    b=b-np.sum(a*b,axis=1,keepdims=True)*a
    b=b/np.linalg.norm(b,axis=1,keepdims=True)
    return np.stack([a,b,np.cross(a,b)],axis=2)


class HeadKinematics:
    """URDF base -> body -> head_link3. Matrix-only Z/X/Y head inverse."""
    def __init__(self,path):
        self.parents={};self.nodes={}
        for j in ET.parse(path).getroot().findall('joint'):
            origin=j.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz
            axis=j.find('axis');a=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')
            limit=j.find('limit')
            node=(j.find('parent').get('link'),j.get('name'),j.get('type'),t,a,
                  None if limit is None else (float(limit.get('lower')),float(limit.get('upper'))))
            self.parents[j.find('child').get('link')]=node
            self.nodes[j.get('name')]=node
        # Analytical inverse is only valid for this verified three-axis head.
        for name,parent,axis in zip(JOINT_NAMES[:3],('body_link5','head_link1','head_link2'),
                                   ([0,0,1],[1,0,0],[0,1,0])):
            n=self.nodes[name]
            if (n[0]!=parent or n[2]!='revolute' or not np.allclose(n[3][:3,:3],np.eye(3),atol=1e-12)
                    or not np.array_equal(n[4],axis) or n[5] is None):
                raise ValueError('URDF head chain does not match validated Z/X/Y inverse')
        self.bounds=np.array([self.nodes[n][5] for n in JOINT_NAMES[:3]])

    def pose(self,values,tip):
        chain=[];link=tip
        while link!='base_link':
            if len(chain)>50:raise ValueError('URDF chain cycle')
            n=self.parents[link];chain.append(n);link=n[0]
        t=np.eye(4)
        for _,name,kind,origin,axis,_ in reversed(chain):
            t=t@origin
            if kind=='fixed':continue
            motion=np.eye(4)
            if kind in ('revolute','continuous'):
                motion[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
            elif kind=='prismatic':motion[:3,3]=axis*values[name]
            else:raise ValueError('Unsupported URDF joint')
            t=t@motion
        return t

    def head_pose(self,body,head):
        return self.pose(dict(zip(BODY_NAMES+JOINT_NAMES[:3],np.r_[body,head])),'head_link3')

    def solve(self,targets,body,*,position_diagnostic_only=False,clip_joint_limits=False):
        """Invert head EEF; optional model-target projection keeps URDF bounds."""
        targets=np.asarray(targets,dtype=float)
        if targets.ndim!=3 or targets.shape[1:]!=(4,4) or not np.isfinite(targets).all():
            raise ValueError('Invalid head EEF matrices')
        bodies=np.asarray(body,dtype=float)
        if bodies.shape==(5,):bodies=np.broadcast_to(bodies,(len(targets),5))
        if bodies.shape!=(len(targets),5) or not np.isfinite(bodies).all():
            raise ValueError('Body path must match head target frames')
        parents=np.stack([self.pose(dict(zip(BODY_NAMES,row)),'body_link5') for row in bodies])
        relative=np.einsum('nij,njk->nik',parents[:,:3,:3].transpose(0,2,1),targets[:,:3,:3])
        # Intrinsic ZXY = Rz(yaw) Rx(roll) Ry(pitch); no quaternion target.
        q=Rotation.from_matrix(relative).as_euler('ZXY')
        outside=(q<self.bounds[:,0]-1e-6) | (q>self.bounds[:,1]+1e-6)
        if np.any(outside) and not clip_joint_limits:
            frame,joint=map(int,np.argwhere(outside)[0])
            angle=float(q[frame,joint])
            low,high=map(float,self.bounds[joint])
            excess=max(low-angle,angle-high)
            detail=dict(frame_index_base=0,first_bad_frame=frame,
                joint=JOINT_NAMES[joint],predicted_rad=angle,
                predicted_deg=float(np.rad2deg(angle)),allowed_rad=[low,high],
                allowed_deg=np.rad2deg([low,high]).tolist(),
                excess_rad=excess,excess_deg=float(np.rad2deg(excess)),
                violating_frames=int(np.count_nonzero(np.any(outside,axis=1))),
                violating_joint_values=int(np.count_nonzero(outside)),
                frame_head_rad=q[frame].tolist(),frame_body_rad=bodies[frame].tolist())
            source=getattr(self,'limit_source','URDF')
            detail['limit_source']=source
            error=ValueError(f'Head EEF inverse exceeds {source} joint limits: '+json.dumps(detail))
            error.diagnostic=detail
            raise error
        predicted=q.copy()
        if clip_joint_limits:
            q=np.clip(q,self.bounds[:,0],self.bounds[:,1])
        # Preserve the existing conservative head pitch hardware boundary.
        q=np.array([head_target(tuple(row)) for row in q])
        actual=np.stack([self.head_pose(b,row) for b,row in zip(bodies,q)])
        def residual(poses):
            pos=np.linalg.norm(poses[:,:3,3]-targets[:,:3,3],axis=1)*1000
            rot=np.rad2deg(Rotation.from_matrix(
                np.einsum('nij,njk->nik',targets[:,:3,:3].transpose(0,2,1),poses[:,:3,:3])).magnitude())
            return pos,rot
        pos,rot=residual(actual)
        check_pos,check_rot=pos,rot
        changed=q!=predicted
        if clip_joint_limits and np.any(changed):
            # Verify the inverse before intentional projection; report the
            # final projected pose's deviation from the prediction below.
            unprojected=np.stack([self.head_pose(b,row) for b,row in zip(bodies,predicted)])
            check_pos,check_rot=residual(unprojected)
        if (not position_diagnostic_only and np.max(check_pos)>1.) or np.max(check_rot)>.1:
            i=int(np.argmax(check_pos))
            raise ValueError(f'Head EEF incompatible with corresponding body pose: frame={i}, '
                             f'position={check_pos[i]:.6f}mm, max_rotation={max(check_rot):.6f}deg')
        self.report={'head_max_position_error_mm':float(max(pos)),
                     'head_max_rotation_error_deg':float(max(rot))}
        if clip_joint_limits:
            clipped=[]
            for frame,joint in np.argwhere(changed):
                before,after=float(predicted[frame,joint]),float(q[frame,joint])
                clipped.append(dict(frame_index_base=0,frame=int(frame),joint=JOINT_NAMES[joint],
                    predicted_rad=before,predicted_deg=float(np.rad2deg(before)),
                    applied_rad=after,applied_deg=float(np.rad2deg(after)),
                    clipped_rad=abs(before-after),
                    urdf_allowed_rad=self.bounds[joint].tolist()))
            self.report.update(head_joint_limit_policy='clip',
                head_joint_clipped_frames=int(np.count_nonzero(np.any(changed,axis=1))),
                head_joint_clipped_values=len(clipped),
                head_joint_clip_max_rad=float(np.max(np.abs(q-predicted))),
                head_joint_clipping=clipped)
        if position_diagnostic_only:
            self.report.update(head_position_error_mm=pos.tolist(),
                               head_rotation_error_deg=rot.tolist(),
                               head_target_xyz_base=targets[:,:3,3].tolist(),
                               head_fk_xyz_base=actual[:,:3,3].tolist())
        return q


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def reexpress_arm_paths(paths, body_path, fk):
    """Express fixed-base EEF targets in the waist-mounted arm_base_link.

    This is the coordinate convention used by native teleoperation.  Keeping
    the targets in base_link lets whole-body control spend the other waist
    joints to satisfy an otherwise world-fixed arm target.
    """
    if any(len(paths[side]) != len(body_path) for side in ('left','right')):
        raise ValueError('Arm and body paths must have the same number of frames')
    result={side:[] for side in ('left','right')}
    for i,body in enumerate(body_path):
        base_from_arm=fk.pose(dict(zip(BODY_NAMES,body)),'arm_base_link')
        arm_from_base=np.linalg.inv(base_from_arm)
        for side in ('left','right'):
            timestamp,position,quaternion=paths[side][i]
            base_from_target=np.eye(4)
            base_from_target[:3,3]=position
            base_from_target[:3,:3]=Rotation.from_quat(quaternion).as_matrix()
            arm_from_target=arm_from_base@base_from_target
            result[side].append((timestamp,tuple(arm_from_target[:3,3]),
                tuple(Rotation.from_matrix(arm_from_target[:3,:3]).as_quat())))
    return result


def prepare(dataset,source='state',speed=1.):
    if source not in ('state','action'):raise ValueError('source must be state/action')
    if not np.isfinite(speed) or not .05<=speed<=1.:
        raise ValueError('speed must be between 0.05 and 1')
    root=Path(dataset).resolve(strict=True)
    manifest=json.loads((root/'manifest.json').read_text())
    by_path={entry['path']:entry for entry in manifest['files']}
    required=('offline_replay.npz','meta/info.json','meta/common_frame.json','meta/robot.urdf',
              'data/chunk-000/episode_000000.parquet')
    for name in required:
        path=(root/name).resolve(strict=True);entry=by_path[name]
        if not path.is_relative_to(root) or path.stat().st_size!=entry['bytes'] or file_hash(path)!=entry['sha256']:
            raise ValueError('Sample integrity mismatch: '+name)
    info=json.loads((root/'meta/info.json').read_text())
    common=json.loads((root/'meta/common_frame.json').read_text())
    expected=[]
    for role in ('left','right','head'):
        expected += [role+'_ee_'+axis for axis in ['x','y','z']+ORDER]
        if role!='head':expected += [role+'_hand_'+name for name in SHARPA_JOINT_NAMES]
    feature=info['features']['observation.state' if source=='state' else 'action']
    if (feature['shape']!=[71] or feature['names']!=expected or info['fps']!=20
            or info['total_episodes']!=1 or info.get('chunk_relative') is not False
            or info.get('eef_coordinate_frame')!='base_link' or common['public_frame']!='base_link'
            or info.get('eef_rotation_order')!=ORDER
            or info.get('eef_rotation_representation')!='rotation_matrix_first_two_columns_concatenated'
            or common.get('T_public_native_base')!={}):
        raise ValueError('Expected one 71D, 20Hz, fixed base_link episode with declared rotation columns')
    with np.load(root/'offline_replay.npz',allow_pickle=False) as saved:
        a=np.array(saved['public_'+source],dtype=float)
        times=np.array(saved['timestamp_s'],dtype=float)
        names=list(saved['joint_names'])
        joints=np.array(saved['measured_joints_rad' if source=='state' else 'target_joints_rad'],dtype=float)
    if (a.shape!=(info['total_frames'],71) or len(a)<2 or not np.isfinite(a).all()
            or times.shape!=(len(a),) or not np.allclose(times,np.arange(len(a))/20,atol=2e-5,rtol=0)
            or len(names)!=len(set(names)) or joints.shape!=(len(a),len(names)) or not np.isfinite(joints).all()):
        raise ValueError('Invalid numeric mirror shape/timestamps')
    # Exact 100Hz grid, ending on last source sample. No extra hold or time shift.
    times=np.arange(len(a))/20
    subdivisions=int(np.ceil(5/speed))
    grid=np.arange((len(a)-1)*subdivisions+1)/(20*subdivisions)
    replay_grid=np.arange(len(grid))/100
    body_samples=joints[:,[names.index(n) for n in BODY_NAMES]]
    body_path=np.column_stack([np.interp(grid,times,body_samples[:,i]) for i in range(5)])
    body=body_path[0]
    fk=HeadKinematics(root/'meta/robot.urdf')
    for col,n in enumerate(BODY_NAMES):
        lo,hi=fk.nodes[n][5]
        if np.any(body_path[:,col]<lo-1e-3) or np.any(body_path[:,col]>hi+1e-3):
            raise ValueError('Body trajectory outside URDF limits: '+n)
    if np.max(np.abs(np.diff(body_path,axis=0)))>.35:
        raise ValueError('Body target discontinuity exceeds 0.35rad step convention')
    paths={};head_targets=None
    for role,start in (('left',0),('right',31),('head',62)):
        rotations=rotation6d(a[:,start+3:start+9])
        interp=Slerp(times,Rotation.from_matrix(rotations))(grid)
        positions=np.column_stack([np.interp(grid,times,a[:,start+i]) for i in range(3)])
        if role=='head':
            head_targets=np.repeat(np.eye(4)[None],len(grid),axis=0)
            head_targets[:,:3,:3]=interp.as_matrix();head_targets[:,:3,3]=positions
        else:
            # GDK EndEffectorPose protocol requires xyzw; arm IK remains native GDK.
            paths[role]=list(zip(replay_grid.tolist(),map(tuple,positions),map(tuple,interp.as_quat())))
    head=fk.solve(head_targets,body_path)
    if np.max(np.abs(np.diff(head,axis=0)))>.35:
        raise ValueError('Head target discontinuity exceeds existing 0.35rad step convention')
    # The original helper uses 17D named joints solely to enter the first pose.
    first=np.r_[head[0],joints[0,[names.index(n) for n in JOINT_NAMES[3:]]]]
    hands=[]
    for start in (9,40):
        values=np.column_stack([np.interp(grid,times,a[:,start+i]) for i in range(22)])
        if np.max(np.abs(np.diff(values,axis=0)))>.35:
            raise ValueError('Aligned hand qpos step exceeds original 0.35rad bound')
        hands.append(values)
    preflight(paths)  # Existing arm continuity checks, unchanged.
    arm_base_paths=reexpress_arm_paths(paths,body_path,fk)
    report=dict(fk.report,source=source,source_frames=len(a),replay_frames=len(grid),
                source_hz=20,replay_hz=100,duration_s=float(replay_grid[-1]),
                source_duration_s=float(grid[-1]),effective_speed=5/subdivisions,
                head_range_deg=np.rad2deg(np.ptp(head,axis=0)).tolist(),
                head_solver='URDF matrix ZXY inverse; not recorded head qpos',
                body_source='measured_joints_rad' if source=='state' else 'target_joints_rad',
                body_range_deg=np.rad2deg(np.ptp(body_path,axis=0)).tolist(),
                body_replayed_per_frame=True,
                physical_replay_run=False)
    return dict(paths=paths,arm_base_paths=arm_base_paths,
                body_q=tuple(body),body_path=list(map(tuple,body_path)),joint_q=tuple(first),
                hand_path=list(zip(map(tuple,hands[0]),map(tuple,hands[1]))),
                head_path=list(map(tuple,head)),report=report)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--source',choices=('state','action'),default='state',
                   help='state matches original measured-feedback replay; action explicitly replays recorded targets')
    p.add_argument('--hands-config',type=Path,default=HAND_CONFIG)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--speed',type=float,default=1.,help='Synchronized playback speed, 0.05..1; output stays 100Hz')
    args=p.parse_args(argv)
    plan=prepare(args.dataset,args.source,args.speed)
    print(json.dumps(plan['report'],ensure_ascii=False,indent=2),flush=True)
    if args.execute:
        with control_locks():
            execute(plan['paths'],plan['body_q'],plan['joint_q'],plan['hand_path'],
                    args.hands_config,head_path=plan['head_path'],body_path=plan['body_path'],
                    arm_base_paths=plan['arm_base_paths'])
    else:
        print('离线检查通过；未初始化GDK、未连接手、未启动真机。',flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('已停止发送新目标；请确认实机状态。',flush=True)
