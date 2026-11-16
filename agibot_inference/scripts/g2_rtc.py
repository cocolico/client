"""Opt-in G2 asynchronous chunk scheduling. Original executors stay untouched.

Pure geometry/scheduling imports are inert. Only run_live_rtc() sends robot IO.
This is asynchronous prefetch + timestamp compensation + cross-fade, not a
prefix-conditioned diffusion/flow RTC sampler. No EMA or extra filter is added.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import json
import math
from pathlib import Path
import socket
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation


def frozen(value, dtype=None):
    a=np.asarray(value,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class Settings:
    action_dt_s: float=.05
    servo_dt_s: float=.01
    blend_s: float=.1
    initial_lead_s: float=.75
    planning_reserve_s: float=.15
    margin_s: float=.1
    def __post_init__(self):
        if not all(np.isfinite(x) for x in self.__dict__.values()):raise ValueError('Nonfinite RTC setting')
        if self.action_dt_s!=.05 or self.servo_dt_s!=.01:raise ValueError('RTC requires original20Hz/100Hz timing')
        if not .01<=self.blend_s<=.25:raise ValueError('RTC blend must be .01..25 seconds')
        if min(self.initial_lead_s,self.planning_reserve_s,self.margin_s)<=0:raise ValueError('Positive RTC timing required')
        if abs(round(self.blend_s/.01)*.01-self.blend_s)>1e-8:raise ValueError('Blend must be whole100Hz ticks')
    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text()))
    @property
    def dt_ns(self):return round(self.servo_dt_s*1e9)
    @property
    def blend_ns(self):return round(self.blend_s*1e9)


def _rotations(v,start):
    c=np.asarray(v)[...,start+3:start+9].reshape(-1,2,3).transpose(0,2,1)
    if not np.allclose(c.transpose(0,2,1)@c,np.eye(2),atol=1e-3):raise ValueError('Invalid physical rotation columns')
    return np.concatenate((c,np.cross(c[:,:,0],c[:,:,1])[:,:,None]),axis=2)


def mix_targets(a,b,alpha):
    """Same-time base targets; translations/qpos linear, rotations shortest SLERP."""
    a,b=np.asarray(a,float),np.asarray(b,float);w=np.asarray(alpha,float).reshape(-1,1)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=78 or len(w)!=len(a):raise ValueError('Expected aligned physical78 targets')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(w).all() or np.any((w<0)|(w>1)):raise ValueError('Invalid blend')
    out=a*(1-w)+b*w
    for p in (0,9,18):
        q0=Rotation.from_matrix(_rotations(a,p)).as_quat();q1=Rotation.from_matrix(_rotations(b,p)).as_quat()
        dot=np.sum(q0*q1,axis=1,keepdims=True);q1=np.where(dot<0,-q1,q1);dot=np.clip(abs(dot),0,1)
        theta=np.arccos(dot);den=np.sin(theta);near=dot>.9995
        safe=np.where(near,1,den)
        q=np.where(near,q0*(1-w)+q1*w,(np.sin((1-w)*theta)*q0+np.sin(w*theta)*q1)/safe)
        q/=np.linalg.norm(q,axis=1,keepdims=True)
        R=Rotation.from_quat(q).as_matrix();out[:,p+3:p+9]=np.c_[R[:,:,0],R[:,:,1]]
    return out


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    values: np.ndarray
    def __post_init__(self):
        t=np.asarray(self.times,np.int64);v=np.asarray(self.values,float)
        if t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or v.shape!=(len(t),78) or not np.isfinite(v).all():raise ValueError('Invalid timed physical78 trajectory')
        for p in (0,9,18):_rotations(v,p)
        object.__setattr__(self,'times',frozen(t));object.__setattr__(self,'values',frozen(v))
    def sample(self,times):
        t=np.atleast_1d(np.asarray(times,np.int64))
        if np.any(t<self.times[0]) or np.any(t>self.times[-1]):raise ValueError('RTC trajectory exhausted; extrapolation/hold forbidden')
        i=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
        w=(t-self.times[i])/(self.times[i+1]-self.times[i])
        return mix_targets(self.values[i],self.values[i+1],w)


def replacement(old,new,start_ns,dt_ns,blend_ns):
    """Only future, same-clock targets are cross-faded; never blend camera frames."""
    if start_ns<old.times[0] or start_ns+blend_ns>old.times[-1]:raise ValueError('Old chunk cannot cover complete RTC blend')
    if start_ns<new.times[0] or start_ns+blend_ns>=new.times[-1]:raise ValueError('New chunk has expired or has no useful tail')
    times=np.arange(start_ns,int(new.times[-1])+1,dt_ns,dtype=np.int64)
    values=new.sample(times);slots=times<=start_ns+blend_ns
    u=(times[slots]-start_ns)/blend_ns
    w=u*u*u*(10+u*(-15+6*u))  # quintic smoothstep: zero endpoint slope/acceleration
    values[slots]=mix_targets(old.sample(times[slots]),values[slots],w)
    return Trajectory(times,values)


def to_base78(values,context,*,drink=False):
    from egoav78_contract import g2_from78,g2_to78,validate78
    from g2_chunk_camera import transform71
    a=validate78(values,'g2',task='drink' if drink else None).copy()
    body=a[...,27:32].copy();a[...,27:32]=0
    base=g2_to78(transform71(g2_from78(a),context.base_from_camera)).astype(float)
    base[...,27:32]=body
    return base


def prediction_trajectory(packet,actions,settings,*,drink=False,origin_ns=None):
    stamp=packet['sample_ns'] if origin_ns is None else origin_ns
    values=to_base78(np.vstack((packet['state'],actions)),packet['context'],drink=drink)
    return Trajectory(stamp+np.arange(len(values),dtype=np.int64)*round(settings.action_dt_s*1e9),values)


@dataclass(frozen=True)
class CompiledPlan:
    trajectory: Trajectory
    native61: np.ndarray
    head: np.ndarray
    body: np.ndarray
    arm_base: dict
    report: dict


def compile_plan(trajectory,adapter,*,drink=False):
    """Call original head inverse/native mapping and drink target-waist transform."""
    from g2_chunk_camera import pose_matrices
    from mixed76_contract import decode_actions
    from replay import BODY_NAMES
    from replay_public_eef import reexpress_arm_paths
    v=trajectory.values;n=len(v)
    if drink:
        if adapter.locked is None:raise ValueError('Drink locked-body observation required')
        body=np.tile(np.r_[adapter.locked,0.],(n,1));body[:,4]=v[:,27]
    else:body=np.tile(adapter.body_target,(n,1))
    for j,name in enumerate(BODY_NAMES):
        low,high=adapter.fk.nodes[name][5]
        tolerance=1e-6 if drink else 1e-3  # same tolerances as the original task adapters
        if np.any(body[:,j]<low-tolerance) or np.any(body[:,j]>high+tolerance):raise ValueError('RTC body target outside original URDF bounds: '+name)
    if n>5 and np.max(abs(body[5:]-body[:-5]))>.35:raise ValueError('RTC body20Hz step exceeds original0.35rad bound')
    head=adapter.fk.solve(pose_matrices(v[:,18:27]),body,position_diagnostic_only=True,clip_joint_limits=not drink)
    if n>5 and np.max(abs(head[5:]-head[:-5]))>.35:raise ValueError('RTC head20Hz step exceeds original0.35rad bound')
    packed=np.zeros((n,76),np.float32)
    packed[:,12:21]=v[:,:9];packed[:,21:30]=v[:,9:18]
    packed[:,30:52]=v[:,32:54];packed[:,53:75]=v[:,55:77];packed[:,9:12]=head
    native=decode_actions(packed,'g2');arms={}
    if drink:
        paths={}
        for side,start in (('left',0),('right',9)):
            q=Rotation.from_matrix(_rotations(v,start)).as_quat()
            paths[side]=[((i+1)*.01,tuple(v[i,start:start+3]),tuple(q[i])) for i in range(n)]
        transformed=reexpress_arm_paths(paths,body,adapter.fk)
        arms={side:frozen(np.array([[t,*p,*q] for t,p,q in rows])) for side,rows in transformed.items()}
    return CompiledPlan(trajectory,frozen(native),frozen(head),frozen(body),arms,
        dict(adapter.fk.report,output_hz=100,drink=drink,head_limit_policy='strict' if drink else 'original_clip',
             kinematic_condition='predicted target waist at every100Hz subframe' if drink else 'original fixed body target'))


class PredictionWorker:
    """One outstanding RPC. Worker plans only; it never publishes robot targets."""
    def __init__(self,client,capture,adapter,settings,*,task,steps,drink,max_response_age,emit):
        self.client,self.capture,self.adapter,self.settings=client,capture,adapter,settings
        self.task,self.steps,self.drink,self.max_age=task,steps,drink,max_response_age
        self.emit=emit;self.job=None;self.thread=None
    def predict(self):
        packet=self.capture.snapshot();stamp=packet['sample_ns'];begin=time.monotonic_ns()
        actions=self.client.infer(packet['state'],packet['head_rgb'],task=self.task,embodiment='g2',sample_ns=stamp)
        if (time.monotonic_ns()-stamp)/1e9>self.max_age:raise TimeoutError('Observation expired; RTC action chunk discarded')
        actions=np.asarray(actions)[:self.steps].copy()
        return packet,actions,dict(self.client.last_response),begin,time.monotonic_ns()
    def submit(self,active,origin_ns):
        if self.job is not None:raise RuntimeError('RTC permits only one outstanding request')
        job={'done':threading.Event(),'submitted_ns':time.monotonic_ns()};self.job=job
        def run():
            try:
                packet,actions,response,begin,end=self.predict()
                new=prediction_trajectory(packet,actions,self.settings,drink=self.drink)
                now=time.monotonic_ns();dt=self.settings.dt_ns
                switch=origin_ns+math.ceil((now+round(self.settings.planning_reserve_s*1e9)-origin_ns)/dt)*dt
                merged=replacement(active.trajectory,new,switch,dt,self.settings.blend_ns)
                planned=compile_plan(merged,self.adapter,drink=self.drink)
                ready=time.monotonic_ns()
                if ready>=switch:raise TimeoutError('RTC planning missed future handoff; original chunk not overwritten')
                job.update(plan=planned,ready_ns=ready,packet=packet,response=response,
                    infer_s=(end-begin)/1e9,planning_s=(ready-now)/1e9,
                    observation_age_at_switch_s=(switch-packet['sample_ns'])/1e9,
                    skipped_prediction_frames=max(0,int((switch-packet['sample_ns'])//50_000_000)),
                    total_s=(ready-job['submitted_ns'])/1e9)
            except BaseException as exc:job['error']=exc
            finally:job['done'].set()
        self.thread=threading.Thread(target=run,name='g2-rtc-predict-plan',daemon=True);self.thread.start()
    def poll(self):
        if self.job is None or not self.job['done'].is_set():return None
        job=self.job;self.job=None
        if 'error' in job:raise job['error']
        return job
    def close(self):
        if self.thread is not None and self.thread.is_alive():
            sock=getattr(self.client,'socket',None)
            if sock is not None:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            self.thread.join(timeout=5)
            if self.thread.is_alive():raise RuntimeError('RTC inference worker did not stop')


class LiveSender:
    """Reuse the original publishers and drink checks. Main thread is sole writer."""
    def __init__(self,robot,hands,*,drink):
        self.robot,self.hands,self.drink=robot,hands,drink
        self.last_head=None;self.sent=0
        hands.require_connected()
        if drink:
            from genie_msgs_pb.msg.ReactiveControl_pb2 import ReactiveControl
            if not hasattr(robot,'_drink_publisher'):
                robot._drink_publisher=robot.gdk.Publisher('/egoav/teleop/retarget',ReactiveControl.DESCRIPTOR.full_name,robot.gdk.QoS())
        else:
            from genie_msgs_pb.msg.TrajectoryTrackingControl_pb2 import TrajectoryTrackingControl
            if not hasattr(robot,'_eef_head_publisher'):
                robot._eef_head_publisher=robot.gdk.Publisher('/gdk/model_predict',TrajectoryTrackingControl.DESCRIPTOR.full_name,robot.gdk.QoS())
    def send(self,plan,index):
        row=plan.native61[index]
        if self.last_head is not None and np.max(abs(row[58:61]-self.last_head))>.35:raise RuntimeError('RTC handoff exceeds original head0.35rad bound')
        if self.drink:
            from replay_full_eef import send_teleop_replay,assert_ready,validate_arm_tracking,HEAD_TRACKING_LIMIT_RAD
            from replay_eef import FRAMES,live_pose,degrees
            from replay import JOINT_NAMES,read_positions
            if self.sent%10==0:
                assert_ready(self.robot.robot)
                for side,start in (('left',0),('right',9)):
                    p,q=live_pose(self.robot.tf,FRAMES[side]);v=plan.trajectory.values[index]
                    target_q=Rotation.from_matrix(_rotations(v[None],start)[0]).as_quat()
                    validate_arm_tracking(side,float(np.linalg.norm(np.asarray(p)-v[start:start+3])),degrees(q,target_q))
                actual=np.asarray(read_positions(self.robot.robot,JOINT_NAMES[:3]),float)
                if not np.isfinite(actual).all() or np.max(abs(actual-plan.head[index]))>HEAD_TRACKING_LIMIT_RAD:raise RuntimeError('RTC drink head tracking error')
            def target(side):
                a=plan.arm_base[side][index];return (float(a[0]),tuple(a[1:4]),tuple(a[4:8]))
            send_teleop_replay(self.robot._drink_publisher,self.robot.gdk,target('left'),target('right'),plan.head[index],plan.body[index],self.sent)
        else:
            from agibot_tasks import SHARPA61_EEF_HEAD_INDICES
            self.robot._send_combined_eef_head(row[list(SHARPA61_EEF_HEAD_INDICES)],self.sent)
        self.hands.send_qpos(row[7:29],row[36:58],frame_index=self.sent//5,subframe_index=self.sent%5)
        self.last_head=row[58:61].copy();self.sent+=1


def run_live_rtc(robot,hands,client,capture,adapter,*,task,steps,iterations,settings,max_response_age,log_path):
    """Explicit live boundary. Bootstrap unchanged once, then stream future plans."""
    from run_mixed76_client import restore_g2_actions
    from g2_drink import execute_plan
    from replay import JOINT_NAMES,read_positions
    drink=task=='drink';log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    stream=log_path.open('x',buffering=1)
    def emit(event,**fields):
        row=dict(event=event,monotonic_ns=time.monotonic_ns(),**fields);stream.write(json.dumps(row,allow_nan=False)+'\n')
        if event!='SERVO_STATS':print('G2_RTC '+json.dumps(row),flush=True)
    worker=PredictionWorker(client,capture,adapter,settings,task=task,steps=steps,drink=drink,max_response_age=max_response_age,emit=emit)
    try:
        emit('CONFIG',task=task,steps=steps,settings=settings.__dict__,ema=False,extra_filter=False,
             baseline_entry='run_inference.sh',scheduler='async_prefetch_timestamp_crossfade; no model prefix conditioning')
        packet,actions,response,_,_=worker.predict()
        # Preserve the exact existing first-target smooth approach, hand reset having already completed.
        if drink:
            first=adapter.plan(packet['context'],actions[:1],startup_transition=True)
            execute_plan(robot,first,hands,packet['observation']['joints'],check=capture.check)
        else:
            native,_=restore_g2_actions(actions[:1],context=packet['context'],adapter=adapter,wire78=True)
            robot.execute_sharpa61(native,packet['observation']['state'],packet['observation']['joints'],hands,check=capture.check,startup_transition=True)
        # Reobserve after the slow initial approach. Never run an expired3s-old prediction.
        packet,actions,response,_,_=worker.predict()
        current=np.asarray(read_positions(robot.robot,JOINT_NAMES),float)
        observed=np.asarray([packet['observation']['joints'][n] for n in JOINT_NAMES],float)
        if np.max(abs(current-observed))>.3:raise RuntimeError('Robot moved during RTC stationary bootstrap; reacquire observation')
        sender=LiveSender(robot,hands,drink=drink)
        origin=time.monotonic_ns()+round(settings.planning_reserve_s*1e9)
        # Initial stationary wait is explicitly retimed. Later moving chunks are NEVER retimed.
        raw=prediction_trajectory(packet,actions,settings,drink=drink,origin_ns=origin)
        times=np.arange(origin+settings.dt_ns,int(raw.times[-1])+1,settings.dt_ns,dtype=np.int64)
        active=compile_plan(Trajectory(times,raw.sample(times)),adapter,drink=drink)
        if np.max(abs(active.head[0]-observed[:3]))>.35:raise RuntimeError('RTC first target exceeds original head-step bound')
        if time.monotonic_ns()>=times[0]:raise TimeoutError('RTC bootstrap planning reserve exceeded; no stream targets sent')
        emit('STREAM_START',first_tick_ns=int(times[0]),initial_wait_retimed=True)
        pending=None;latencies=deque(maxlen=20);tick=1;switches=0;last_sent=None;intervals=[];max_late=0.
        while True:
            capture.check()
            job=worker.poll()
            if job is not None:
                pending=job['plan'];latencies.append(job['total_s'])
                emit('NEXT_READY',switch_ns=int(pending.trajectory.times[0]),infer_s=job['infer_s'],planning_s=job['planning_s'],
                     observation_age_at_switch_s=job['observation_age_at_switch_s'],skipped_prediction_frames=job['skipped_prediction_frames'],
                     request_id=job['response'].get('request_id'),sample_ns=job['packet']['sample_ns'])
            target_ns=origin+tick*settings.dt_ns
            delay=(target_ns-time.monotonic_ns())/1e9
            if delay<-.1:raise RuntimeError('Action execution is over100ms late (original bound)')
            if delay>0:time.sleep(delay)
            capture.check()
            # A result can become ready during the final10ms sleep. Consume it
            # before the scheduled tick, rather than missing its handoff by one tick.
            job=worker.poll()
            if job is not None:
                if pending is not None:raise RuntimeError('Multiple pending RTC plans')
                pending=job['plan'];latencies.append(job['total_s'])
                emit('NEXT_READY',switch_ns=int(pending.trajectory.times[0]),infer_s=job['infer_s'],planning_s=job['planning_s'],
                     observation_age_at_switch_s=job['observation_age_at_switch_s'],skipped_prediction_frames=job['skipped_prediction_frames'],
                     request_id=job['response'].get('request_id'),sample_ns=job['packet']['sample_ns'])
            if pending is not None and target_ns>=pending.trajectory.times[0]:
                if target_ns!=pending.trajectory.times[0]:raise RuntimeError('RTC missed scheduled handoff; do not replay past targets')
                active=pending;pending=None;switches+=1
                emit('HANDOFF',switch= switches,time_ns=int(target_ns),blend_s=settings.blend_s)
            if target_ns>active.trajectory.times[-1]:raise RuntimeError('RTC queue underrun; no stale target held or extrapolated')
            index=int(np.searchsorted(active.trajectory.times,target_ns))
            if index>=len(active.trajectory.times) or active.trajectory.times[index]!=target_ns:raise RuntimeError('RTC100Hz clock mismatch')
            sender.send(active,index)
            now=time.monotonic_ns();late=max(0.,(now-target_ns)/1e9);max_late=max(max_late,late)
            if last_sent is not None:intervals.append((now-last_sent)/1e9)
            last_sent=now
            remaining=(active.trajectory.times[-1]-target_ns)/1e9
            lead=max(settings.initial_lead_s,(max(latencies) if latencies else 0)+settings.planning_reserve_s+settings.blend_s+settings.margin_s)
            if worker.job is None and pending is None and remaining<=lead:
                if switches>=iterations-1:
                    if remaining<=0:break
                else:
                    worker.submit(active,origin);emit('PREFETCH',queue_remaining_s=float(remaining),lead_s=float(lead))
            if tick%100==0:
                emit('SERVO_STATS',sent=sender.sent,interval_min_s=min(intervals) if intervals else None,
                     interval_max_s=max(intervals) if intervals else None,max_late_s=max_late,queue_remaining_s=float(remaining))
                intervals=[];max_late=0.
            tick+=1
        emit('COMPLETE',sent=sender.sent,switches=switches)
    except BaseException as exc:
        emit('STOP',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        try:worker.close()
        finally:stream.close()
