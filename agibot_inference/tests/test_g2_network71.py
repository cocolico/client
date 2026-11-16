"""Real loopback TCP + existing CLI; all physical devices are replaced/blocked."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _ours_protocol import send_message,recv_message
from client_mixed76 import Mixed76Client
import run_mixed76_client as runner
import hand_startup
from test_g2_chunk_camera import native

BODY=[-.3,.6,-.3,.05,.1]
REP='g2_eef71_chunk_first_camera_rot6d'


def metadata():
    return dict(protocol_version=5,model='pi05',representation=REP,state_dim=71,action_dim=71,
        state_units='physical',action_units='physical',action_mode='chunk_relative',
        coordinate_frame='chunk_first_camera_fixed',rotation_order='first_two_columns',
        head_frame='head_link3',position_units='meter',joint_units='radian',
        horizon=32,dt_s=.05,camera_keys=['head_rgb'],weights_strict=True,
        checkpoint='SIMULATED_TEST_ONLY',normalization_sha256='test',deployment_id='test71',
        prompts={'yifu':'test'},embodiments=['g2'])


class Peer:
    def __init__(self,mutate=None):
        self.listener=socket.socket();self.listener.bind(('127.0.0.1',0));self.listener.listen()
        self.listener.settimeout(3);self.port=self.listener.getsockname()[1]
        self.requests=[];self.errors=[];self.connections=0;self.mutate=mutate
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        try:
            with self.listener:
                conn,_=self.listener.accept();self.connections+=1
                with conn:
                    conn.settimeout(3)
                    while True:
                        try:req=recv_message(conn)
                        except ConnectionError:return
                        self.requests.append(req)
                        if req['cmd']=='info':out=metadata()
                        elif req['cmd']=='reset':
                            out=dict(protocol_version=5,request_id=req['request_id'],session_id='s71',
                                     deployment_id='test71',ok=True)
                        else:
                            out={k:req[k] for k in ('request_id','session_id','sample_ns','embodiment')}
                            actions=np.tile(req['observation']['state'],(32,1))
                            actions[:,62]+=.2  # head position mismatch must be diagnostic
                            out.update(protocol_version=5,deployment_id='test71',representation=REP,
                                action_units='physical',dt_s=.05,action_mask=[True]*71,
                                actions=actions,inference_seconds=.001)
                        if self.mutate:self.mutate(req,out)
                        send_message(conn,out)
        except BaseException as exc:self.errors.append(exc)

    def finish(self,test):
        self.thread.join(4)
        test.assertFalse(self.thread.is_alive())
        test.assertFalse(self.errors,self.errors)


class Network71Tests(unittest.TestCase):
    def ready(self):
        self.assertIn('representation',inspect.signature(Mixed76Client).parameters,
                      'Existing client needs the explicitly selected 71D schema')

    def client(self,peer):
        self.ready()
        return Mixed76Client('127.0.0.1',peer.port,model='pi05',representation='g2-camera71')

    def test_persistent_connection_and_exact_physical71_requests(self):
        self.ready()
        from g2_chunk_camera import G2ChunkCamera
        adapter=G2ChunkCamera(ROOT/'assets/robot/agibot.urdf',BODY)
        peer=Peer();client=self.client(peer)
        try:
            for i in range(2):
                x=native();x[58]=.2-.4*i
                state,ctx=runner.prepare_g2_observation({'state':x},sample_ns=100+i,adapter=adapter)
                actions=client.infer(state,np.zeros((8,9,3),np.uint8),task='yifu',embodiment='g2',sample_ns=100+i)
                self.assertEqual(actions.shape,(32,71))
                actual,report=runner.restore_g2_actions(actions,context=ctx,adapter=adapter)
                np.testing.assert_allclose(actual,np.tile(x,(32,1)),atol=2e-7)
                self.assertAlmostEqual(report['head_position_error_mm'][0],200.,places=3)
        finally:client.close()
        peer.finish(self)
        self.assertEqual(peer.connections,1)
        self.assertEqual([r['cmd'] for r in peer.requests],['info','reset','infer','infer'])
        self.assertNotEqual(peer.requests[2]['request_id'],peer.requests[3]['request_id'])
        for req in peer.requests[2:]:
            self.assertEqual(req['observation']['state'].shape,(71,))
            np.testing.assert_allclose(req['observation']['state'][62:],[0,0,0,1,0,0,0,1,0],atol=1e-7)

    def test_wrong_contract_and_old_mode_cannot_silently_mix(self):
        self.ready()
        for key,value in [('protocol_version',4),('state_dim',76),('action_mode','absolute'),
                          ('rotation_order','rows'),('coordinate_frame','base_link')]:
            with self.subTest(key=key):
                peer=Peer(lambda req,out:out.update({key:value}) if req['cmd']=='info' else None)
                with self.assertRaises(ValueError):self.client(peer)
                peer.finish(self)
        peer=Peer()
        with self.assertRaises(ValueError):Mixed76Client('127.0.0.1',peer.port,model='pi05')
        peer.finish(self)

    def test_stale_or_malformed_response_is_rejected(self):
        self.ready()
        for key,value in [('request_id','stale'),('sample_ns',-1),('action_mask',[True]*76),
                          ('actions',np.zeros((32,76),np.float32))]:
            with self.subTest(key=key):
                peer=Peer(lambda req,out:out.update({key:value}) if req['cmd']=='infer' else None)
                client=self.client(peer)
                try:
                    with self.assertRaises(ValueError):
                        client.infer(np.zeros(71),np.zeros((2,2,3),np.uint8),task='yifu',embodiment='g2')
                finally:client.close()
                peer.finish(self)

    def test_cli_recorded_roundtrip_no_robot_or_reset(self):
        self.ready()
        peer=Peer()
        with tempfile.TemporaryDirectory() as tmp:
            obs=Path(tmp)/'obs.npz';out=Path(tmp)/'result.npz'
            np.savez(obs,state=native(),head_rgb=np.zeros((8,9,3),np.uint8),body_target=BODY)
            args=['client','--representation','g2-camera71','--model','pi05','--embodiment','g2',
                  '--task','yifu','--inference','local','--port',str(peer.port),
                  '--observation',str(obs),'--output',str(out)]
            with patch.object(sys,'argv',args),patch.dict(sys.modules,{'agibot_gdk':None,'sharpa':None}),\
                    patch.object(runner,'read_reset_body_target',side_effect=AssertionError('offline reset read')),\
                    contextlib.redirect_stdout(io.StringIO()):runner.main()
            with np.load(out,allow_pickle=False) as data:
                self.assertEqual(data['actions71'].shape,(32,71))
                np.testing.assert_allclose(data['actions61'],np.tile(native(),(32,1)),atol=2e-7)
        peer.finish(self)

    def test_execute_routing_simulated_only_reobserves_and_reuses_executor(self):
        self.ready()
        peer=Peer();calls=[];observations=[]
        class Robot:
            def __init__(self,**kwargs):pass
            def get_observation61(self,hands):
                x=native();x[58]-=.4*len(observations)
                obs={'state':x,'joints':{'source':len(observations)},'images':{'head_color':np.zeros((2,2,3),np.uint8)}}
                observations.append(obs);return obs
            def execute_sharpa61(self,actions,state,joints,hands,**kwargs):
                calls.append((actions.copy(),state,joints,kwargs))
        class Hands:
            closed=False
            @classmethod
            def from_config(cls,*a):return cls()
            def connect(self):pass
            def close(self):Hands.closed=True
        args=['client','--representation','g2-camera71','--model','pi05','--embodiment','g2',
              '--task','yifu','--inference','local','--port',str(peer.port),'--execute','--iterations','2']
        import run_ours_client
        with patch.object(sys,'argv',args),patch.object(runner,'read_reset_body_target',return_value=BODY),\
                patch.object(run_ours_client,'startup_reset') as reset,\
                patch.object(hand_startup,'load_hand_reset'),\
                patch.object(hand_startup,'reset_hands_to_recording'),\
                patch.dict(sys.modules,{'robot_wrapper':NS(AgiBotRobot=Robot),'sharpa_hands':NS(SharpaHands=Hands)}),\
                contextlib.redirect_stdout(io.StringIO()):runner.main()
        peer.finish(self);self.assertEqual(reset.call_count,1);self.assertTrue(Hands.closed)
        self.assertEqual(len(calls),2)
        for i,(actions,state,joints,kwargs) in enumerate(calls):
            np.testing.assert_allclose(actions,np.tile(observations[i]['state'],(32,1)),atol=2e-7)
            self.assertIs(state,observations[i]['state']);self.assertIs(joints,observations[i]['joints'])
            self.assertEqual(kwargs,{'startup_transition':i==0})

    def test_read_body_target_uses_readonly_original_reset_helper(self):
        self.ready()
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'input.json';config.write_text('{}')
            names=[f'idx0{i}_body_joint{i}' for i in range(1,6)]
            pose=dict(name='G2零点',unit='rad',joint_names=names,joint_positions=BODY)
            reply=NS(stdout='SHARPA_RESET_V1 '+json.dumps(dict(validated=True,motion_sent=False,pose=pose)))
            modules={'sharpa_pose_studio.common.paths':NS(INPUT_CONFIG_PATH=config),
                     'sharpa_pose_studio.teleop.session':NS(RealOps=lambda *a:NS(
                         reset_command=lambda:['ssh','agi','python robot_reset.py --execute']))}
            with patch.dict(sys.modules,modules),patch('subprocess.run',return_value=reply) as run:
                values=runner.read_reset_body_target(Path(tmp),'agi')
            self.assertEqual(values,BODY)
            self.assertEqual(run.call_args.args[0],['ssh','agi','python robot_reset.py'])

    def test_failed_handshake_never_reaches_reset_or_body_read(self):
        self.ready()
        peer=Peer(lambda req,out:out.update(protocol_version=4) if req['cmd']=='info' else None)
        args=['client','--representation','g2-camera71','--model','pi05','--embodiment','g2',
              '--task','yifu','--inference','local','--port',str(peer.port),'--execute']
        with patch.object(sys,'argv',args),patch.object(runner,'read_reset_body_target',
                side_effect=AssertionError('Invalid server reached robot')):
            with self.assertRaisesRegex(ValueError,'protocol_version'):runner.main()
        peer.finish(self)

    def test_tcp71_to_original_100hz_executor_with_fake_device_sinks(self):
        self.ready()
        from test_live61 import Live61Tests
        from test_gdk_eef_execution import Clock,Command
        from test_sharpa_execution import CombinedTransport
        from sharpa_hands import SharpaHands
        import robot_wrapper as wrapper
        import run_ours_client
        fixture=Live61Tests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        clock=Clock();combined=CombinedTransport();robot=fixture.robot
        robot.gdk.EndEffectorPose=Command;robot.gdk.EndEffectorControlGroup=NS(kBothArms=12)
        robot.gdk.Publisher=combined.publisher;robot.gdk.QoS=NS;robot.gdk.PubMsgInfo=NS
        peer=Peer()
        args=['client','--representation','g2-camera71','--model','pi05','--embodiment','g2',
              '--task','yifu','--inference','local','--port',str(peer.port),'--execute']
        with patch.object(sys,'argv',args),patch.object(runner,'read_reset_body_target',return_value=BODY),\
                patch.object(run_ours_client,'startup_reset'),\
                patch.object(hand_startup,'load_hand_reset'),\
                patch.object(hand_startup,'reset_hands_to_recording'),\
                patch.object(wrapper,'AgiBotRobot',return_value=robot),\
                patch.object(SharpaHands,'from_config',return_value=fixture.hands),\
                patch.object(wrapper,'read_positions',return_value=tuple(fixture.joints.values())),\
                patch.object(wrapper.time,'monotonic',clock.monotonic),\
                patch.object(wrapper.time,'sleep',clock.sleep),\
                contextlib.redirect_stdout(io.StringIO()):runner.main()
        peer.finish(self)
        # Original startup300ticks + 32frames*5ticks; official protobuf is captured only.
        self.assertEqual(len(combined.messages),460)
        self.assertAlmostEqual(clock.now,4.6)
        self.assertEqual([g.group_id for g in combined.messages[-1][0].retarget_groups_trajectory[0].retarget_infos],
                         [2,3,0,9,1])
        for serial in ('C8519039C852','CE569038CE55'):
            self.assertEqual(len(fixture.sdk.waves[serial].commands),460)
        self.assertEqual(fixture.sdk.connected,set())


if __name__=='__main__':unittest.main()
