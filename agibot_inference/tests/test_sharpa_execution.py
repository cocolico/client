import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
import robot_wrapper as wrapper
from test_gdk_eef_execution import Clock, Command, Transport


class HandSink:
    def __init__(self): self.calls = []
    def require_connected(self): pass
    def send_qpos(self, left, right, **context):
        self.calls.append((tuple(left), tuple(right), context))


class CombinedTransport(Transport):
    """Capture the real executor's outbound payload, never connect to GDK."""
    def __init__(self):
        super().__init__()
        self.combined = []
        self.combined_result = 0
        self.separate_calls = []
        self.publisher_args = []
        self.messages = []

    def publisher(self, topic, message_type, qos):
        self.publisher_args.append((topic, message_type, qos))
        return self

    def move_head_joint_servo(self, *args):
        self.separate_calls.append('head')
        return super().move_head_joint_servo(*args)

    def end_effector_pose_control(self, *args):
        self.separate_calls.append('arms')
        return super().end_effector_pose_control(*args)

    def publish(self, message, info):
        # Round-trip the real official protobuf, not a hand-made schema.
        message = type(message).FromString(message.SerializeToString())
        self.messages.append((message, info))
        groups = {g.group_id:g for g in message.retarget_groups_trajectory[0].retarget_infos}
        action = {}
        for name, group in (('left_arm',2), ('right_arm',3)):
            p = groups[group].target_frame_poses[0]
            action[name] = dict(control_type='ABS_POSE', action_data=[p.position.x,p.position.y,p.position.z,
                p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
        action['head'] = dict(control_type='ABS_JOINT', action_data=[
            groups[g].target_joint_positions[0] for g in (0,9,1)])
        actions = [action]
        self.combined.append((info.source_ts_ns, {}, actions,
            dict(robot_link=groups[2].frame_id, trajectory_reference_time=message.trajectory_reference_time)))
        action = actions[0]
        self.head.append((action['head']['action_data'],))
        command = Command()
        for name, pose in (('left_arm', command.left_end_effector_pose),
                           ('right_arm', command.right_end_effector_pose)):
            data = action[name]['action_data']
            pose.position.x, pose.position.y, pose.position.z = data[:3]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = data[3:]
        self.arms.append(command)
        return self.combined_result == 0


class SharpaExecutionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(wrapper.AgiBotRobot, 'execute_sharpa61'), '61D execution missing')
        self.robot = object.__new__(wrapper.AgiBotRobot)
        self.robot.robot = CombinedTransport()
        self.robot.gdk = NS(EndEffectorPose=Command, EndEffectorControlGroup=NS(kBothArms=12))
        self.robot.gdk.Publisher = self.robot.robot.publisher
        self.robot.gdk.QoS = NS
        self.robot.gdk.PubMsgInfo = NS
        self.joints = dict.fromkeys(wrapper.JOINT_NAMES, 0.)
        self.start = np.zeros(61); self.start[6] = self.start[35] = 1.
        self.start[7:29] = .1; self.start[36:58] = .2
        self.hands = HandSink(); self.clock = Clock()
        for p in [patch.object(wrapper, 'read_positions', return_value=(0.,)*17),
                  patch.object(wrapper.time, 'monotonic', self.clock.monotonic),
                  patch.object(wrapper.time, 'sleep', self.clock.sleep)]:
            p.start(); self.addCleanup(p.stop)

    def run_actions(self, actions, **kwargs):
        self.robot.execute_sharpa61(actions, self.start, self.joints, self.hands, **kwargs)

    def test_startup_smooth_entry_then_all50_frames_without_return_to_start(self):
        target = self.start.copy()
        target[0] = .1; target[29] = -.1
        target[58:] = [.1, -.1, .2]
        target[7:29] = .3; target[36:58] = .4
        target[3:7] = [0., 0., np.sin(.1), np.cos(.1)]
        actions = np.tile(target, (50, 1))
        actions[1:, 0] = .11
        self.run_actions(actions, startup_transition=True)
        self.assertAlmostEqual(self.clock.now, 5.5)
        self.assertEqual(len(self.hands.calls), 550)
        self.assertAlmostEqual(self.robot.robot.arms[149].left_end_effector_pose.position.x, .05)
        np.testing.assert_allclose(self.hands.calls[149][0], .2)
        np.testing.assert_allclose(self.robot.robot.head[149][0], [.05, -.05, .1])
        q = self.robot.robot.arms[149].left_end_effector_pose.orientation
        np.testing.assert_allclose([q.x,q.y,q.z,q.w], [0.,0.,np.sin(.05),np.cos(.05)])
        positions = [c.left_end_effector_pose.position.x for c in self.robot.robot.arms]
        self.assertLess(positions[0], .00001)  # Not the old 20% first tick.
        self.assertAlmostEqual(positions[299], .1)
        np.testing.assert_allclose(positions[300:305], .1)  # Never go back to measured start.
        self.assertAlmostEqual(positions[305], .102)
        self.assertEqual([c[2]['frame_index'] for c in self.hands.calls[300:]],
                         [i for i in range(50) for _ in range(5)])
        self.assertEqual(self.hands.calls[0][2]['frame_index'], -1)

    def test_large_startup_hand_delta_extends_transition_beyond_three_seconds(self):
        target = self.start.copy(); target[7:29] += 1.2
        self.run_actions(np.tile(target, (50, 1)), startup_transition=True)
        self.assertGreaterEqual(self.clock.now, 8.5 - 1e-9)
        q = np.array([self.start[7]] + [c[0][0] for c in self.hands.calls])
        self.assertLessEqual(np.max(np.abs(np.diff(q))) / .01, .300001)
        self.assertEqual(self.hands.calls[-1][2]['frame_index'], 49)

    def test_startup_does_not_hide_invalid_later_frame(self):
        actions = np.tile(self.start, (50, 1)); actions[49, 35] = 0.
        with self.assertRaises(ValueError): self.run_actions(actions, startup_transition=True)
        self.assertEqual(self.robot.robot.arms, [])
        self.assertEqual(self.hands.calls, [])

    def test_startup_keeps_existing_head_step_check(self):
        target = self.start.copy(); target[58] = .36
        with self.assertRaisesRegex(RuntimeError, '0.35 rad'):
            self.run_actions(target[None], startup_transition=True)
        self.assertEqual(self.hands.calls, [])

    def test_startup_cancel_stops_before_normal_chunk(self):
        def check():
            if self.clock.now >= .5: raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.run_actions(self.start[None], startup_transition=True, check=check)
        self.assertTrue(self.hands.calls)
        self.assertTrue(all(c[2]['frame_index'] == -1 for c in self.hands.calls))

    def test_all_61_fields_split_correctly_and_interpolate_in_one_loop(self):
        target = self.start.copy()
        target[0] = .1; target[29] = -.2; target[58:] = [.1, -.1, .2]
        target[7:29] = .2; target[36:58] = .3
        self.run_actions(target[None])
        self.assertEqual(len(self.hands.calls), 5)
        np.testing.assert_allclose(self.hands.calls[0][0], .12)
        np.testing.assert_allclose(self.hands.calls[0][1], .22)
        np.testing.assert_allclose(self.hands.calls[-1][0], .2)
        np.testing.assert_allclose(self.hands.calls[-1][1], .3)
        last = self.robot.robot.arms[-1]
        self.assertAlmostEqual(last.left_end_effector_pose.position.x, .1)
        self.assertAlmostEqual(last.right_end_effector_pose.position.x, -.2)
        np.testing.assert_allclose(self.robot.robot.head[-1][0], [.1, -.1, .2])
        self.assertEqual(self.hands.calls[-1][2], {'frame_index': 0, 'subframe_index': 4})
        self.assertAlmostEqual(self.clock.now, .05)

    def test_50_frames_execute_250_subframes_over_2point5_seconds(self):
        self.run_actions(np.tile(self.start, (50, 1)))
        self.assertEqual(len(self.hands.calls), 250)
        self.assertEqual(len(self.robot.robot.arms), 250)
        self.assertAlmostEqual(self.clock.now, 2.5)
        self.assertEqual(self.hands.calls[-1][2]['frame_index'], 49)

    def test_hand_failure_prevents_remaining_ticks(self):
        def fail(*args, **kwargs): raise RuntimeError('right hand send failed')
        self.hands.send_qpos = fail
        with self.assertRaisesRegex(RuntimeError, 'right hand'): self.run_actions(self.start[None])
        self.assertEqual(len(self.robot.robot.arms), 1)

    def test_nonfinite_hand_target_rejected_before_head_or_arm_send(self):
        target = self.start.copy(); target[55] = np.nan
        with self.assertRaises(ValueError): self.run_actions(target[None])
        self.assertEqual(self.robot.robot.arms, [])
        self.assertEqual(self.robot.robot.head, [])

    def test_head_check_still_applies(self):
        target = self.start.copy(); target[58] = .36
        with self.assertRaisesRegex(RuntimeError, '0.35 rad'): self.run_actions(target[None])
        self.assertEqual(self.hands.calls, [])

    def test_transport_order_within_every_tick(self):
        events = []
        self.robot.robot.move_head_joint_servo = lambda *a: events.append('head') or 0
        self.robot.robot.end_effector_pose_control = lambda *a: events.append('arms') or 0
        self.robot.robot.publish = lambda *a: events.append('combined') or True
        self.hands.send_qpos = lambda *a, **k: events.append('hands')
        self.run_actions(self.start[None])
        self.assertEqual(events, ['combined', 'hands']*5)

    def test_head3_and_both_absolute_eef_share_each_native_request(self):
        target = self.start.copy()
        target[:3] = [.1, .2, .3]
        target[29:32] = [-.1, -.2, .4]
        target[58:] = [.1, -.12, .23]
        self.run_actions(target[None])
        self.assertEqual(self.robot.robot.separate_calls, [])
        self.assertEqual(len(self.robot.robot.combined), 5)
        _, _, actions, kwargs = self.robot.robot.combined[-1]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0], {
            'left_arm': {'control_type': 'ABS_POSE', 'action_data': [.1,.2,.3,0.,0.,0.,1.]},
            'right_arm': {'control_type': 'ABS_POSE', 'action_data': [-.1,-.2,.4,0.,0.,0.,1.]},
            'head': {'control_type': 'ABS_JOINT', 'action_data': [.1,-.12,.23]},
        })
        self.assertEqual(kwargs, {'robot_link': 'base_link', 'trajectory_reference_time': .01})
        self.assertEqual(len(self.robot.robot.publisher_args), 1)
        self.assertEqual(self.robot.robot.publisher_args[0][:2],
                         ('/gdk/model_predict', 'genie_msgs.msg.pb.TrajectoryTrackingControl'))
        msg, info = self.robot.robot.messages[-1]
        self.assertEqual(msg.header.stamp.sec*10**9+msg.header.stamp.nanosec, info.source_ts_ns)
        groups = msg.retarget_groups_trajectory[0].retarget_infos
        self.assertEqual([g.group_id for g in groups], [2,3,0,9,1])
        self.assertEqual([g.control_type for g in groups], [0,0,3,3,3])
        self.assertEqual([g.input_type for g in groups], [54]*5)
        self.assertEqual([list(g.target_frame_names) for g in groups[:2]],
                         [['arm_l_end_link'], ['arm_r_end_link']])
        self.assertTrue(all(not g.reference_frame_names and not g.reference_frame_poses for g in groups))
        self.run_actions(target[None])
        self.assertEqual(len(self.robot.robot.publisher_args), 1)  # Reuse DDS writer across chunks.

    def test_combined_failure_stops_before_hands_and_remaining_frames(self):
        self.robot.robot.combined_result = 1
        with self.assertRaisesRegex(RuntimeError, 'combined'):
            self.run_actions(self.start[None])
        self.assertEqual(len(self.robot.robot.combined), 1)
        self.assertEqual(self.hands.calls, [])

    def test_native_publish_exception_stops_before_hands_without_fallback(self):
        def fail(message, info):
            raise RuntimeError('native publish failure')
        self.robot.robot.publish = fail
        with self.assertRaisesRegex(RuntimeError, 'native publish failure'):
            self.run_actions(self.start[None])
        self.assertEqual(self.hands.calls, [])
        self.assertEqual(self.robot.robot.separate_calls, [])

    def test_large_startup_eef_translation_and_rotation_extend_entry(self):
        for translation, rotation in ((.4, 0.), (0., 1.2)):
            with self.subTest(translation=translation, rotation=rotation):
                self.clock.now = 0.
                target = self.start.copy()
                target[0] = translation
                target[3:7] = [0., 0., np.sin(rotation/2), np.cos(rotation/2)]
                self.run_actions(target[None], startup_transition=True)
                self.assertGreaterEqual(self.clock.now, 6.05-1e-9)

    def test_missing_hands_does_not_silently_execute_only_head_and_arms(self):
        with self.assertRaises(ValueError):
            self.robot.execute_sharpa61(self.start[None], self.start, self.joints, None)
        self.assertEqual(self.robot.robot.arms, [])

    def test_unconnected_hands_rejected_before_head_or_arm_send(self):
        from sharpa_hands import SharpaHands
        from fake_sharpa_sdk import SDK
        from pathlib import Path
        hands = SharpaHands.from_config(Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json', sdk=SDK())
        with self.assertRaisesRegex(RuntimeError, 'connected'):
            self.robot.execute_sharpa61(self.start[None], self.start, self.joints, hands)
        self.assertEqual(self.robot.robot.arms, [])
        self.assertEqual(self.robot.robot.head, [])

    def test_integrated_native_adapter_50_frames_preserve_all44_joint_indices(self):
        from sharpa_hands import SharpaHands
        from fake_sharpa_sdk import SDK
        from pathlib import Path
        sdk = SDK()
        hands = SharpaHands.from_config(Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json', sdk=sdk)
        hands.connect(); self.addCleanup(hands.close)
        target = self.start.copy()
        target[7:29] = np.arange(22)*.001
        target[36:58] = np.arange(22)*.002
        self.robot.execute_sharpa61(np.tile(target, (50, 1)), self.start, self.joints, hands)
        left, right = hands.read_qpos()
        np.testing.assert_allclose(left, np.arange(22)*.001)
        np.testing.assert_allclose(right, np.arange(22)*.002)
        self.assertEqual(len(sdk.waves['C8519039C852'].commands), 250)
        self.assertEqual(len(sdk.waves['CE569038CE55'].commands), 250)


if __name__ == '__main__': unittest.main()
