"""Recorded hand reset: pure file fixtures and fake hardware, no robot IO."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from genie_msgs_pb.msg.JointState_pb2 import JointState
from agibot_tasks import SHARPA_JOINT_NAMES


class Clock:
    now=0.
    def monotonic(self):return self.now
    def sleep(self,t):self.now+=t


class Hands:
    def __init__(self, follow=True):
        self.current=np.zeros((2,22));self.sent=[];self.follow=follow
    def read_qpos(self):return self.current.copy()
    def send_qpos(self,left,right,**context):
        q=np.asarray([left,right]);self.sent.append(q.copy())
        if self.follow:self.current=q.copy()


class HandStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.config=self.root/'reset.json'
        mapping={}
        for index,task in enumerate(('drink','yifu','diguan','dengpao')):
            episode=self.root/task;record=episode/'record';record.mkdir(parents=True)
            mapping[task]=str(episode)
            for side,value in [('left',.2+index*.1),('right',.3+index*.1)]:
                self.write(record/f'-sharpa-{side}-state.pbdat',value)
        self.config.write_text(json.dumps(mapping))

    def write(self,path,value,*,bad_name=False,nan=False):
        names=list(reversed(SHARPA_JOINT_NAMES))
        if bad_name:names[0]='wrong'
        message=JointState();message.name.extend(names)
        message.position.extend([float('nan') if nan else value]*22)
        message.motor_position.extend([99.]*22)
        message.header.stamp.sec=100;message.header.stamp.nanosec=123
        payload=message.SerializeToString()
        with path.open('wb') as stream:
            stream.write(struct.pack('<q',2))
            stream.write(struct.pack('<5q',100000000123,0,0,0,len(payload)));stream.write(payload)
            message.position[:]=[9.]*22;payload=message.SerializeToString()
            stream.write(struct.pack('<5q',100010000123,0,0,0,len(payload)));stream.write(payload)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('hand_startup'), 'Recorded hand reset is missing')
        import hand_startup
        return hand_startup

    def test_task_selects_first_feedback_not_motor_or_second_frame(self):
        m=self.api()
        for task,left,right in [('drink',.2,.3),('yifu',.3,.4),('diguan',.4,.5),('dengpao',.5,.6)]:
            reset=m.load_hand_reset(task,self.config)
            np.testing.assert_allclose(reset.target[0],left)
            np.testing.assert_allclose(reset.target[1],right)
            self.assertEqual(reset.episode,str(self.root/task))
            self.assertEqual(reset.task,task)

    def test_unknown_task_missing_file_bad_names_and_nan_fail_closed(self):
        m=self.api()
        with self.assertRaises(ValueError):m.load_hand_reset('unknown',self.config)
        path=self.root/'drink/record/-sharpa-left-state.pbdat'
        for kwargs in ({'bad_name':True},{'nan':True}):
            self.write(path,.2,**kwargs)
            with self.assertRaises(ValueError):m.load_hand_reset('drink',self.config)
        path.unlink()
        with self.assertRaises(FileNotFoundError):m.load_hand_reset('drink',self.config)

    def test_reorders_recorded_joint_names_into_sdk_order(self):
        m=self.api();path=self.root/'drink/record/-sharpa-left-state.pbdat'
        message=JointState();message.name.extend(reversed(SHARPA_JOINT_NAMES))
        message.position.extend(reversed([i*.01 for i in range(22)]));message.header.stamp.sec=100
        payload=message.SerializeToString();path.write_bytes(struct.pack('<q',1)+struct.pack('<5q',100000000000,0,0,0,len(payload))+payload)
        np.testing.assert_allclose(m.load_hand_reset('drink',self.config).target[0],np.arange(22)*.01)

    def test_smooth_reset_reaches_both_hands_before_ready(self):
        m=self.api();reset=m.load_hand_reset('drink',self.config)
        hands=Hands();clock=Clock();out=io.StringIO()
        import replay_feedback
        with patch.object(replay_feedback.time,'monotonic',clock.monotonic),patch.object(replay_feedback.time,'sleep',clock.sleep),contextlib.redirect_stdout(out):
            m.reset_hands_to_recording(hands,reset,check=lambda:None)
        self.assertGreaterEqual(clock.now,4.)
        np.testing.assert_allclose(hands.current,reset.target)
        trajectory=np.concatenate([np.zeros((1,2,22)),np.array(hands.sent)])
        self.assertLessEqual(np.abs(np.diff(trajectory,axis=0)).max()/.01,.30001)
        self.assertIn('HAND_RESET_READY',out.getvalue())

    def test_no_arrival_never_reports_ready(self):
        m=self.api();hands=Hands(follow=False);clock=Clock();out=io.StringIO()
        import replay_feedback
        with patch.object(replay_feedback.time,'monotonic',clock.monotonic),patch.object(replay_feedback.time,'sleep',clock.sleep),contextlib.redirect_stdout(out):
            with self.assertRaises(RuntimeError):
                m.reset_hands_to_recording(hands,m.load_hand_reset('drink',self.config),check=lambda:None)
        self.assertNotIn('HAND_RESET_READY',out.getvalue())

    def test_fault_before_reset_sends_nothing(self):
        m=self.api();hands=Hands()
        def fault():raise RuntimeError('controller fault')
        with self.assertRaisesRegex(RuntimeError,'controller fault'):
            m.reset_hands_to_recording(hands,m.load_hand_reset('drink',self.config),check=fault)
        self.assertEqual(hands.sent,[])

    def invoke_client(self,task,*,legacy=False,follow=True,missing=False,execute=True):
        """Real startup flow/ramp, only device and network boundaries replaced."""
        m=self.api()
        import run_mixed76_client as current
        import run_ours_client as old
        import robot_wrapper
        import client_deployed78
        import sharpa_hands
        import replay_feedback
        import g2_drink
        from test_g2_chunk_camera import native
        from replay import BODY_NAMES,JOINT_NAMES
        reset=m.load_hand_reset(task,self.config);events=[];hands=Hands(follow);clock=Clock()
        hands.connect=lambda:events.append('connect')
        hands.close=lambda:events.append('close_hands')
        def observe(h):
            # Missing reset or acquiring observation too early reproduces bug.
            np.testing.assert_allclose(h.current,reset.target,atol=.1,rtol=0)
            events.append('observe')
            state=native();state[7:29]=h.current[0];state[36:58]=h.current[1]
            joints=dict(zip(BODY_NAMES,[-.3,.6,-.3,.05,.1]))
            joints.update(dict(zip(JOINT_NAMES,np.r_[state[58:],np.zeros(14)])))
            return dict(state=state,joints=joints,images={'head_color':np.zeros((8,8,3),np.uint8)})
        robot=NS(robot=NS(get_motion_control_status=lambda:NS(error_code=0,collision_pairs_1=[],collision_pairs_2=[])),
                 get_observation61=observe,execute_sharpa61=lambda *a,**k:events.append('execute'))
        def infer(state,*a,**k):events.append('infer');return np.tile(state,(32,1))
        client=NS(info={'horizon':32},server_info={},infer=infer,close=lambda:events.append('close_client'),
                  policy_infer=lambda obs:dict(actions=np.tile(obs['state'],(50,1)),inference_seconds=.1),
                  _disconnect=lambda:events.append('close_client'))
        runner=old if legacy else current
        args=['client','--task',task,'--model','pi05' if legacy else 'joint','--iterations','1','--inference','local']
        if not legacy:args+=['--representation','g2-camera78','--embodiment','g2','--steps','1']
        if execute:args+=['--execute']
        else:args+=['--test'] if legacy else []
        def load(task_arg):
            events.append('load')
            if missing:raise FileNotFoundError('missing source')
            self.assertEqual(task_arg,task);return reset
        try:
            with contextlib.ExitStack() as stack:
                for p in [patch.object(sys,'argv',args),
                    patch.object(runner,'inference_connection',return_value=contextlib.nullcontext(('test',0))),
                    patch.object(client_deployed78,'open_client78',return_value=client),
                    patch.object(old,'OursClient',return_value=client),
                    patch.object(old,'startup_reset',side_effect=lambda *a:events.append('body_reset')),
                    patch.object(current,'read_reset_body_target',return_value=[-.3,.6,-.3,.05,.1]),
                    patch.object(robot_wrapper,'AgiBotRobot',return_value=robot),
                    patch.object(sharpa_hands.SharpaHands,'from_config',return_value=hands),
                    patch.object(m,'load_hand_reset',side_effect=load),
                    patch.object(g2_drink,'execute_plan',side_effect=lambda *a:events.append('execute')),
                    patch.object(replay_feedback.time,'monotonic',clock.monotonic),
                    patch.object(replay_feedback.time,'sleep',clock.sleep),
                    contextlib.redirect_stdout(io.StringIO())]:stack.enter_context(p)
                runner.main()
        except BaseException as exc:return events,exc
        return events,None

    def test_current_tasks_and_legacy61_observe_only_after_hand_arrival(self):
        for task,legacy in [('drink',False),('yifu',False),('diguan',False),('dengpao',False),('yifu',True)]:
            with self.subTest(task=task,legacy=legacy):
                events,error=self.invoke_client(task,legacy=legacy)
                self.assertIsNone(error,str(error))
                self.assertLess(events.index('load'),events.index('body_reset'))
                self.assertLess(events.index('body_reset'),events.index('connect'))
                self.assertLess(events.index('connect'),events.index('observe'))
                self.assertEqual(events.count('observe'),1)
                self.assertIn('execute',events)

    def test_missing_source_prevents_body_reset_and_enable(self):
        for legacy in (False,True):
            events,error=self.invoke_client('yifu',legacy=legacy,missing=True)
            self.assertIsInstance(error,FileNotFoundError)
            self.assertNotIn('body_reset',events);self.assertNotIn('connect',events)

    def test_failed_hand_arrival_blocks_observation_inference_and_closes(self):
        for legacy in (False,True):
            events,error=self.invoke_client('yifu',legacy=legacy,follow=False)
            self.assertIsInstance(error,RuntimeError)
            self.assertNotIn('observe',events);self.assertNotIn('infer',events)
            self.assertIn('close_hands',events);self.assertIn('close_client',events)

    def test_offline_handshake_does_not_load_reset_or_enable(self):
        events,error=self.invoke_client('drink',execute=False,missing=True)
        self.assertIsNone(error,str(error))
        self.assertNotIn('load',events);self.assertNotIn('body_reset',events)
        self.assertNotIn('connect',events)


if __name__=='__main__':unittest.main()
