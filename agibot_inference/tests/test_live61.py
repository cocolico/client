"""No hardware: real observation assembly/CLI routing, fake external transports."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import robot_wrapper as wrapper
import run_ours_client as cli
import hand_startup
from sharpa_hands import SharpaHands
from fake_sharpa_sdk import SDK

ROOT = Path(__file__).resolve().parents[1]


class Live61Tests(unittest.TestCase):
    def setUp(self):
        self.robot = object.__new__(wrapper.AgiBotRobot)
        self.robot.gdk = NS(CameraType=NS(kHeadColor='h', kHandLeftColor='l', kHandRightColor='r'),
                            Encoding=NS(JPEG=1, PNG=2, UNCOMPRESSED=3), ColorFormat=NS(BGR=1, RGB=2))
        self.stamp = 1_000_000_000
        self.joints = dict(zip(wrapper.JOINT_NAMES, np.arange(17) / 100))
        self.robot.robot = NS(get_joint_states=lambda: dict(timestamp=self.stamp, states=[
            dict(name=n, motor_position=v) for n,v in self.joints.items()]))
        self.camera_calls = []
        def image(camera, timeout):
            self.camera_calls.append(camera)
            color = {'h': [3,2,1], 'l': [6,5,4], 'r': [9,8,7]}[camera]
            return NS(data=np.array(color, np.uint8), width=1, height=1, encoding=3,
                      color_format=1, timestamp_ns=self.stamp)
        self.robot.camera = NS(get_latest_image=image)
        self.robot.tf = NS(get_tf_from_base_link=lambda frame: NS(
            translation=NS(x=.5, y=.2 if frame=='arm_l_end_link' else -.2, z=.8),
            rotation=NS(x=0.,y=0.,z=0.,w=1.)))
        self.robot.last_image_timestamps = {k:0 for k in ('head_color','left_wrist_camera','right_wrist_camera')}
        self.sdk = SDK()
        self.hands = SharpaHands.from_config(ROOT/'configs/sharpa_hands.json', sdk=self.sdk)
        self.hands.connect()
        self.addCleanup(self.hands.close)
        self.sdk.waves['C8519039C852'].values = list(np.arange(22)/100)
        self.sdk.waves['CE569038CE55'].values = list(np.arange(22)/50)

    def test_head_rgb_measured_eef_and_all61_fields(self):
        self.assertTrue(hasattr(self.robot, 'get_observation61'), 'live 61D observation missing')
        obs = self.robot.get_observation61(self.hands)
        expected = np.r_[.5,.2,.8,0,0,0,1,np.arange(22)/100,
                         .5,-.2,.8,0,0,0,1,np.arange(22)/50,0,.01,.02]
        np.testing.assert_allclose(obs['state'], expected)
        self.assertEqual(obs['joints'], self.joints)
        self.assertEqual(self.camera_calls, ['h'])
        for name, pixel in [('head_color',[1,2,3])]:
            self.assertEqual(obs['images'][name].shape, (480,640,3))
            np.testing.assert_equal(obs['images'][name][0,0], pixel)
        self.assertFalse(any(w.commands for w in self.sdk.waves.values()))

    def test_existing_camera_freshness_and_1s_joint_skew_preserved(self):
        self.assertTrue(hasattr(self.robot, 'get_observation61'), 'live 61D observation missing')
        self.robot.last_image_timestamps['right_wrist_camera'] = self.stamp
        obs = self.robot.get_observation61(self.hands)
        self.assertNotIn('right_wrist_camera', obs['images'])
        self.robot.last_image_timestamps = {k:0 for k in self.robot.last_image_timestamps}
        self.robot.robot.get_joint_states = lambda: dict(timestamp=self.stamp+1_000_000_001, states=[])
        with self.assertRaisesRegex(RuntimeError, '1 second'):
            self.robot.get_observation61(self.hands)

    def test_first_sample_waits_for_new_frame_on_each_camera(self):
        self.robot.last_image_timestamps = {}
        original = self.robot.camera.get_latest_image
        def advancing(*args):
            self.stamp += 50_000_000
            return original(*args)
        self.robot.camera.get_latest_image = advancing
        with patch.object(wrapper.time, 'sleep'):
            obs = self.robot.get_observation61(self.hands)
        self.assertEqual(self.camera_calls, ['h','h'])
        self.assertEqual(obs['state'].shape, (61,))

    def test_removed_wrist_hardware_is_never_accessed(self):
        original = self.robot.camera.get_latest_image
        calls = []
        def head_only(kind, timeout):
            calls.append(kind)
            if kind != 'h': raise AssertionError('Removed wrist camera accessed')
            self.assertEqual(timeout, 1000.)
            return original(kind, timeout)
        self.robot.camera.get_latest_image = head_only
        del self.robot.gdk.CameraType.kHandLeftColor
        del self.robot.gdk.CameraType.kHandRightColor
        obs = self.robot.get_observation61(self.hands)
        self.assertEqual(calls, ['h'])
        self.assertEqual(set(obs['images']), {'head_color'})
        self.assertEqual(set(self.robot.last_image_timestamps), {'head_color'})
        self.assertEqual(obs['state'].shape, (61,))

    def test_head_stall_still_exits(self):
        from test_gdk_eef_execution import Clock
        clock = Clock()
        self.robot.last_image_timestamps['head_color'] = self.stamp
        with patch.object(wrapper.time, 'monotonic', clock.monotonic), patch.object(wrapper.time, 'sleep', clock.sleep):
            with self.assertRaisesRegex(RuntimeError, 'No fresh RGB image: head_color'):
                self.robot.get_observation61(self.hands)

    def test_client_to_real_execution_loop_with_native_sdk_double(self):
        # A slicing/routing regression loses commands at these external sinks.
        from test_gdk_eef_execution import Clock, Command
        clock = Clock()
        sent_head, sent_arms = [], []
        self.robot.gdk.EndEffectorPose = Command
        self.robot.gdk.EndEffectorControlGroup = NS(kBothArms=12)
        self.robot.robot.move_head_joint_servo = lambda *a: sent_head.append(a) or 0
        self.robot.robot.end_effector_pose_control = lambda c: sent_arms.append(c) or 0
        from test_sharpa_execution import CombinedTransport
        combined = CombinedTransport()
        self.robot.gdk.Publisher = combined.publisher
        self.robot.gdk.QoS = NS
        self.robot.gdk.PubMsgInfo = NS
        client = NS(policy_infer=lambda obs: {'actions':np.tile(obs['state'],(50,1)), 'inference_seconds':.1},
                    _disconnect=lambda:None)
        with patch.object(sys,'argv',['client','--task','yifu','--model','pi05','--execute','--iterations','1']), \
             patch.object(cli,'OursClient',return_value=client), \
             patch.object(cli,'startup_reset'), \
             patch.object(hand_startup,'load_hand_reset'), \
             patch.object(hand_startup,'reset_hands_to_recording'), \
             patch.object(wrapper,'AgiBotRobot',return_value=self.robot), \
             patch.object(SharpaHands,'from_config',return_value=self.hands), \
             patch.object(wrapper,'read_positions',return_value=tuple(self.joints.values())), \
             patch.object(wrapper.time,'monotonic',clock.monotonic), \
             patch.object(wrapper.time,'sleep',clock.sleep):
            cli.main()
        self.assertEqual((len(sent_head),len(sent_arms)),(0,0))
        self.assertEqual(len(combined.messages),550)
        self.assertEqual([g.group_id for g in combined.messages[-1][0].retarget_groups_trajectory[0].retarget_infos],
                         [2,3,0,9,1])
        for serial in ('C8519039C852','CE569038CE55'):
            self.assertEqual(len(self.sdk.waves[serial].commands),550)
        self.assertAlmostEqual(clock.now,5.5)
        self.assertEqual(self.sdk.connected,set())


class ClientRoutingTests(unittest.TestCase):
    def invoke(self, args, client, robot, hands, events):
        with patch.object(sys, 'argv', ['run_ours_client.py', *args]), \
             patch.object(cli, 'OursClient', return_value=client), \
             patch.object(wrapper, 'AgiBotRobot', return_value=robot), \
             patch.object(SharpaHands, 'from_config', return_value=hands), \
             patch.object(hand_startup,'load_hand_reset'), \
             patch.object(hand_startup,'reset_hands_to_recording',side_effect=lambda *a,**k:events.append('hand_reset')), \
             patch.object(cli, 'startup_reset', side_effect=lambda *a: events.append('reset'), create=True):
            cli.main()

    def test_full50_once_after_reset_then_observe_next_and_cleanup(self):
        events=[]
        obs={'state':np.arange(61), 'joints':{'measured':1}, 'images':{}}
        actions=np.ones((50,61))
        client=NS(policy_infer=lambda o: events.append('infer') or {'actions':actions,'inference_seconds':.1},
                  _disconnect=lambda:events.append('disconnect'))
        transitions=[]
        def execute(a,s,j,h,*,startup_transition=False):
            transitions.append(startup_transition)
            self.assertEqual(a.shape,(50,61)); self.assertIs(s,obs['state'])
            self.assertIs(j,obs['joints']); self.assertIs(h,hands);events.append('execute50')
        robot=NS(get_observation61=lambda h:events.append('observe') or obs, execute_sharpa61=execute)
        hands=NS(connect=lambda:events.append('connect_hands'), close=lambda:events.append('close_hands'))
        self.invoke(['--task','yifu','--model','pi05','--execute','--iterations','2'],client,robot,hands,events)
        self.assertEqual(events,['reset','connect_hands','hand_reset','observe','infer','execute50','observe','infer','execute50','close_hands','disconnect'])
        self.assertEqual(transitions,[True,False])

    def test_execution_failure_stops_next_inference_and_closes_hands(self):
        events=[]
        client=NS(policy_infer=lambda o:{'actions':np.zeros((50,61)),'inference_seconds':.1},
                  _disconnect=lambda:events.append('disconnect'))
        def fail(*a,**kw):raise RuntimeError('GDK command failure')
        robot=NS(get_observation61=lambda h:dict(state=np.zeros(61),joints={}),execute_sharpa61=fail)
        hands=NS(connect=lambda:events.append('connect_hands'),close=lambda:events.append('close_hands'))
        with self.assertRaisesRegex(RuntimeError,'GDK command failure'):
            self.invoke(['--task','yifu','--model','pi05','--execute'],client,robot,hands,events)
        self.assertEqual(events,['reset','connect_hands','hand_reset','close_hands','disconnect'])

    def test_no_execute_and_legacy_live_rejected_before_any_hardware(self):
        for args in (['--task','yifu','--model','pi05'],['--execute'],
                     ['--task','yifu','--model','pi05','--test','--execute']):
            with patch.object(sys,'argv',['client',*args]),patch.object(cli,'OursClient') as client:
                with self.assertRaises(SystemExit):cli.main()
                client.assert_not_called()

    def test_reset_failure_never_connects_hardware(self):
        events=[]
        client=NS(_disconnect=lambda:events.append('disconnect'))
        with patch.object(sys,'argv',['client','--task','yifu','--model','pi05','--execute']), \
             patch.object(cli,'OursClient',return_value=client), \
             patch.object(cli,'startup_reset',side_effect=RuntimeError('reset failed')), \
             patch.object(hand_startup,'load_hand_reset'), \
             patch.object(wrapper,'AgiBotRobot') as robot, \
             patch.object(SharpaHands,'from_config') as hands:
            with self.assertRaisesRegex(RuntimeError,'reset failed'):cli.main()
            robot.assert_not_called();hands.assert_not_called()
        self.assertEqual(events,['disconnect'])


if __name__=='__main__': unittest.main()
