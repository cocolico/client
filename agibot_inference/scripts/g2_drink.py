"""Drink-only 78D waist path. Planning is offline; only execute_plan sends IO.

The chunk anchor is measured once. Every 100Hz subframe is solved with its
PREDICTED target body, never with the observed/initial/final waist as a seed.
Old tasks never import/use this executor. Native replay's message builder is
reused without replay's reset, first-pose motion, or hand connection lifecycle.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial.transform import Rotation
from egoav78_contract import g2_to78, g2_from78, validate78
from g2_chunk_camera import G2ChunkCamera, _frozen, transform71, pose_matrices
from replay import BODY_NAMES, JOINT_NAMES, read_positions
from replay_public_eef import HeadKinematics, reexpress_arm_paths
from robot_wrapper import _slerp_xyzw
from controller_diagnostics import ControllerFault


@dataclass(frozen=True)
class DrinkContext:
    sample_ns: int
    base_from_camera: np.ndarray
    body: np.ndarray
    state78: np.ndarray
    native61: np.ndarray


@dataclass(frozen=True)
class DrinkPlan:
    body: np.ndarray
    head: np.ndarray
    hands: np.ndarray
    arm_base: dict
    base_poses: dict
    start_body: np.ndarray
    startup_ticks: int
    frame_indices: tuple
    report: dict


class DrinkAdapter:
    def __init__(self, urdf):
        self.urdf = urdf
        self.fk = HeadKinematics(urdf)
        # Explicit online drink override; the source URDF/controller is unchanged.
        self.fk.bounds[1,0] = np.deg2rad(-25.)
        self.fk.limit_source = 'online drink'
        self.body_names = BODY_NAMES
        self.locked = None

    def begin(self, native61, measured_body, *, sample_ns):
        body = np.asarray(measured_body, dtype=float)
        # Reuse fixed adapter's validation/FK but feed ACTUAL body this time.
        observed = G2ChunkCamera(self.urdf, body).begin(native61, sample_ns=sample_ns)
        if self.locked is None:
            self.locked = _frozen(body[:4])
        if np.max(np.abs(body[:4]-self.locked)) > .1:
            raise ValueError('Non-yaw body joints moved; reacquire/restart drink')
        state = g2_to78(observed.state71); state[27] = body[4]
        return DrinkContext(sample_ns, observed.base_from_camera, _frozen(body),
                            _frozen(state,np.float32), _frozen(native61))

    def plan(self, context, actions78, *, startup_transition=False):
        if not isinstance(context, DrinkContext) or self.locked is None:
            raise ValueError('Drink observation context required')
        a = validate78(actions78, 'g2', task='drink').copy()
        if a.ndim != 2 or not 1 <= len(a) <= 32:
            raise ValueError('Drink requires 1..32 actions')
        waist = a[:,27].astype(float)
        # Preserve waist separately; do NOT silently discard it via legacy71.
        a[:,27] = 0
        base = transform71(g2_from78(a), context.base_from_camera)
        initial = context.state78.copy(); initial[27] = 0
        initial = transform71(g2_from78(initial)[None],context.base_from_camera)[0]
        path = np.vstack((initial,base))
        body = np.tile(np.r_[self.locked,context.body[4]],(len(path),1))
        body[1:,4] = waist
        for j,name in enumerate(BODY_NAMES):
            lo,hi = self.fk.nodes[name][5]
            if np.any(body[:,j]<lo-1e-6) or np.any(body[:,j]>hi+1e-6):
                raise ValueError('Body target outside URDF limits: '+name)
        if np.max(np.abs(np.diff(body,axis=0))) > .35:
            raise ValueError('Body target step exceeds existing replay 0.35rad bound')
        matrices = {s:pose_matrices(path[:,s:s+9]) for s in (0,31,62)}
        # Preserve existing 20Hz head-step check, using TARGET body conditions.
        key_head = self.fk.solve(matrices[62],body,position_diagnostic_only=True)
        if np.max(np.abs(np.diff(key_head,axis=0))) > .35:
            raise ValueError('Predicted head joint step exceeds 0.35 rad')
        quats = {s:Rotation.from_matrix(t[:,:3,:3]).as_quat() for s,t in matrices.items()}
        startup = 0
        if startup_transition:
            duration = max(3.,1.5*np.max(np.abs(key_head[1]-key_head[0]))/.3,
                           1.5*np.max(np.abs(body[1]-body[0]))/.3,
                           1.5*np.max(np.abs(path[1,9:31]-path[0,9:31]))/.3,
                           1.5*np.max(np.abs(path[1,40:62]-path[0,40:62]))/.3)
            for s in (0,31):
                distance = np.linalg.norm(path[1,s:s+3]-path[0,s:s+3])
                angle = 2*np.arccos(np.clip(abs(np.dot(quats[s][0],quats[s][1])),0.,1.))
                duration = max(duration,1.5*distance/.1,1.5*angle/.3)
            startup = int(np.ceil(duration*100))
        n = startup+len(a)*5
        bodies, fingers, heads = [],[],[]
        poses = {'left':[], 'right':[]}; head_matrices=[]; indices=[]
        for tick in range(1,n+1):
            if tick <= startup:
                u=tick/startup;alpha=u*u*(3-2*u);left,right=0,1
                index,fraction=-1,tick-1
            else:
                index,fraction=divmod(tick-startup-1,5);alpha=(fraction+1)/5
                left=1 if startup and index==0 else index;right=index+1
            indices.append((index,fraction))
            bodies.append(body[left]*(1-alpha)+body[right]*alpha)
            fingers.append(np.r_[path[left,9:31]*(1-alpha)+path[right,9:31]*alpha,
                                  path[left,40:62]*(1-alpha)+path[right,40:62]*alpha])
            for name,s in (('left',0),('right',31),('head',62)):
                p=path[left,s:s+3]*(1-alpha)+path[right,s:s+3]*alpha
                q=_slerp_xyzw(quats[s][left],quats[s][right],alpha)
                if name=='head':
                    t=np.eye(4);t[:3,3]=p;t[:3,:3]=Rotation.from_quat(q).as_matrix();head_matrices.append(t)
                else:poses[name].append((tick*.01,tuple(p),tuple(q)))
        bodies=np.asarray(bodies)
        # Solve AFTER interpolating both EEF and target waist at 100Hz.
        head=self.fk.solve(np.asarray(head_matrices),bodies,position_diagnostic_only=True)
        if np.max(np.abs(np.diff(np.vstack((context.native61[58:],head)),axis=0))) > .35:
            raise ValueError('Head subframe discontinuity exceeds existing bound')
        arm_base = reexpress_arm_paths(poses,bodies,self.fk)
        report=dict(self.fk.report,task='drink',sample_ns=context.sample_ns,
            action_frames=len(a),subframes=n,startup_ticks=startup,output_hz=100,
            head_joint_limit_source=self.fk.limit_source,head_joint_limits_rad=self.fk.bounds.tolist(),
            waist_joint='idx05_body_joint5',waist_slot=27,
            waist_min_rad=float(bodies[:,4].min()),waist_max_rad=float(bodies[:,4].max()),
            locked_body_targets=self.locked.tolist(),
            kinematic_condition='predicted target waist at every 100Hz subframe',
            arm_target_frame='arm_base_link',head_xyz='diagnostic_only',
            base_from_camera=context.base_from_camera.tolist(),robot_io=False)
        return DrinkPlan(_frozen(bodies),_frozen(head),_frozen(fingers),arm_base,poses,
                         context.body,startup,tuple(indices),report)


def execute_plan(robot, plan, hands, start_joints, check=None):
    """Explicit live-only boundary, using existing native dynamic-body replay IO."""
    from replay_full_eef import (send_teleop_replay, assert_ready,
        validate_arm_tracking, HEAD_TRACKING_LIMIT_RAD)
    from replay_eef import FRAMES, live_pose, degrees
    from genie_msgs_pb.msg.ReactiveControl_pb2 import ReactiveControl
    if not isinstance(plan,DrinkPlan):raise ValueError('Expected a validated drink plan')
    n=len(plan.body)
    for value,width in ((plan.body,5),(plan.head,3),(plan.hands,44)):
        if value.shape!=(n,width) or not np.isfinite(value).all() or not n:
            raise ValueError('Malformed drink plan')
    if len(plan.frame_indices)!=n:
        raise ValueError('Malformed drink frame indices')
    for item in plan.frame_indices:
        if (not isinstance(item,tuple) or len(item)!=2
                or any(not isinstance(x,(int,np.integer)) for x in item)):
            raise ValueError('Malformed drink frame indices')
    for paths in (plan.arm_base,plan.base_poses):
        for side in ('left','right'):
            if len(paths[side])!=n:
                raise ValueError('Malformed drink arm path')
            for i,(stamp,point,quat) in enumerate(paths[side]):
                if np.shape(point)!=(3,) or np.shape(quat)!=(4,):
                    raise ValueError('Malformed drink arm pose dimensions')
                values=np.asarray([stamp,*point,*quat],float)
                if (values.shape!=(8,) or not np.isfinite(values).all()
                        or abs(stamp-(i+1)*.01)>1e-7
                        or abs(np.linalg.norm(quat)-1)>1e-5):
                    raise ValueError('Malformed drink arm pose')
    check=check or (lambda:None)
    check()
    try:
        assert_ready(robot.robot)
    except ControllerFault as exc:
        exc.execution = dict(phase='drink_preflight', failed_subframe=None,
                             last_sent_subframe=None, monitoring_interval_s=.1)
        raise
    hands.require_connected()
    names=list(BODY_NAMES)+list(JOINT_NAMES)
    expected=np.asarray([start_joints[n] for n in names],float)
    current=np.asarray(read_positions(robot.robot,names),float)
    if (not np.isfinite(expected).all() or not np.isfinite(current).all()
            or np.max(np.abs(current-expected))>.4):
        changed = [i for i in range(len(names))
                   if not np.isfinite(expected[i]) or not np.isfinite(current[i])
                   or abs(current[i]-expected[i]) > .4]
        details = '; '.join(
            f'{names[i]}: observed={expected[i]:.6f}rad current={current[i]:.6f}rad '
            f'delta={abs(current[i]-expected[i]):.6f}rad' for i in changed)
        raise RuntimeError('Robot moved while inference was running; reacquire observation; '
                           'limit=0.400000rad; ' + details)
    if not np.allclose(expected[:5],plan.start_body,atol=1e-7,rtol=0):
        raise ValueError('Drink plan/body observation mismatch')
    if not hasattr(robot,'_drink_publisher'):
        robot._drink_publisher=robot.gdk.Publisher('/egoav/teleop/retarget',
            ReactiveControl.DESCRIPTOR.full_name,robot.gdk.QoS())
    started=time.monotonic()
    for i in range(len(plan.body)):
        check()
        delay=started+(i+1)*.01-time.monotonic()
        if delay < -.1:raise RuntimeError('Action execution is over 100 ms late')
        time.sleep(max(0.,delay));check()
        # Retain dynamic-body replay's checks and 10Hz monitoring cadence.
        if i % 10 == 0:
            try:
                assert_ready(robot.robot)
                for side in ('left','right'):
                    actual_p,actual_q=live_pose(robot.tf,FRAMES[side])
                    _,target_p,target_q=plan.base_poses[side][i]
                    validate_arm_tracking(side,float(np.linalg.norm(np.asarray(actual_p)-target_p)),
                                          degrees(actual_q,target_q))
                actual=np.asarray(read_positions(robot.robot,JOINT_NAMES[:3]),float)
                error=float(np.max(np.abs(actual-plan.head[i])))
                if not np.isfinite(actual).all() or error>HEAD_TRACKING_LIMIT_RAD:
                    raise RuntimeError(f'Head tracking error {error:.4f}rad')
            except RuntimeError as exc:
                if isinstance(exc, ControllerFault):
                    exc.execution = dict(
                        phase='drink_execution', failed_subframe=i,
                        last_sent_subframe=i-1 if i else None,
                        action_frame=plan.frame_indices[i][0],
                        action_subframe=plan.frame_indices[i][1],
                        monitoring_interval_s=.1,
                        send_note='Last sent means the client publisher returned success; '
                                  'it does not confirm controller acceptance. The failing '
                                  'subframe target was not sent.')
                raise RuntimeError(f'Drink subframe {i}: {exc}') from exc
        send_teleop_replay(robot._drink_publisher,robot.gdk,
            plan.arm_base['left'][i],plan.arm_base['right'][i],plan.head[i],plan.body[i],i)
        check()
        index,subframe=plan.frame_indices[i]
        hands.send_qpos(plan.hands[i,:22],plan.hands[i,22:],frame_index=index,subframe_index=subframe)
