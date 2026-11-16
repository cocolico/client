"""Offline execution tests: no GDK initialization or hardware connection."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import robot_wrapper as wrapper


class Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, delay): self.now += delay


def pose():
    return NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=1.))


class Command:
    def __init__(self):
        self.left_end_effector_pose = pose()
        self.right_end_effector_pose = pose()


class Transport:
    def __init__(self):
        self.head = []
        self.arms = []
        self.head_result = 0
        self.arm_result = 0
    def move_head_joint_servo(self, positions, period, low_latency):
        self.head.append((positions, period, low_latency))
        return self.head_result
    def end_effector_pose_control(self, command):
        self.arms.append(command)
        return self.arm_result


class ExecutionTests(unittest.TestCase):
    def test_quaternion_failure_identifies_side_frame_norm_before_send(self):
        actions = np.tile(self.start, (5, 1))
        actions[3, 10:14] = 0.
        with self.assertRaises(ValueError) as caught:
            self.run_eef(actions)
        detail = getattr(caught.exception, 'diagnostic', {})
        self.assertEqual(detail.get('side'), 'right')
        self.assertEqual(detail.get('source'), 'prediction')
        self.assertEqual(detail.get('frame_index'), 3)
        self.assertEqual(detail.get('norm'), 0.)
        self.assertEqual(detail.get('quaternion_xyzw'), [0.,0.,0.,0.])
        self.assertEqual(self.robot.robot.arms, [])
        self.assertEqual(self.robot.robot.head, [])

    def test_model_quaternion_is_normalized_without_near_unit_range(self):
        target = self.start.copy()
        q = np.array([0.4751534163951874, 0.40102383494377136,
                      0.5295037031173706, 0.5410286784172058])
        target[10:14] = q
        target[3:7] = [0., 0., 0., 2.]
        self.run_eef(target[None])
        sent = self.robot.robot.arms[-1]
        right = sent.right_end_effector_pose.orientation
        np.testing.assert_allclose([right.x,right.y,right.z,right.w], q/np.linalg.norm(q))
        left = sent.left_end_effector_pose.orientation
        np.testing.assert_allclose([left.x,left.y,left.z,left.w], [0.,0.,0.,1.])
        np.testing.assert_allclose(target[10:14], q)  # Never mutate raw model output.

    def setUp(self):
        self.robot = object.__new__(wrapper.AgiBotRobot)
        self.robot.robot = Transport()
        self.robot.gdk = NS(EndEffectorPose=Command,
                           EndEffectorControlGroup=NS(kBothArms=12))
        self.joints = dict.fromkeys(wrapper.JOINT_NAMES, 0.)
        self.start = np.array([.4, .2, .5, 0., 0., 0., 1.,
                               .4, -.2, .5, 0., 0., 0., 1., 0., 0., 0.])
        self.clock = Clock()
        for p in [patch.object(wrapper, 'read_positions', return_value=(0.,)*17),
                  patch.object(wrapper.time, 'monotonic', self.clock.monotonic),
                  patch.object(wrapper.time, 'sleep', self.clock.sleep)]:
            p.start(); self.addCleanup(p.stop)

    def run_eef(self, actions=None, **kwargs):
        self.robot.execute_eef(self.start[None] if actions is None else actions,
                               self.start, self.joints, **kwargs)

    def test_absolute_eef_and_head_sent_five_ticks_without_joint_arm_commands(self):
        target = self.start.copy()
        target[0] = .5; target[8] = -.3; target[14:] = [.1, -.1, .2]
        with patch.object(wrapper, 'send_servo', side_effect=AssertionError('old arm servo used')):
            self.run_eef(target[None])
        self.assertEqual(len(self.robot.robot.arms), 5)
        first, last = self.robot.robot.arms[0], self.robot.robot.arms[-1]
        self.assertEqual(last.group, 12)
        self.assertEqual(last.life_time, .05)
        self.assertAlmostEqual(first.left_end_effector_pose.position.x, .42)
        self.assertAlmostEqual(last.left_end_effector_pose.position.x, .5)
        self.assertAlmostEqual(last.right_end_effector_pose.position.y, -.3)
        np.testing.assert_allclose(self.robot.robot.head[-1][0], [.1, -.1, .2])
        self.assertEqual(self.robot.robot.head[-1][1:], (.02, False))
        self.assertAlmostEqual(self.clock.now, .05)

    def test_quaternion_slerp_and_antipodal_representations(self):
        target = self.start.copy(); target[3:7] = [0., 0., 1., 0.]; target[10:14] *= -1
        self.run_eef(target[None])
        for i, command in enumerate(self.robot.robot.arms, 1):
            q = command.left_end_effector_pose.orientation
            np.testing.assert_allclose([q.x, q.y, q.z, q.w],
                [0., 0., np.sin(np.pi*i/10), np.cos(np.pi*i/10)], atol=1e-12)
            r = command.right_end_effector_pose.orientation
            np.testing.assert_allclose([r.x, r.y, r.z, r.w], [0., 0., 0., 1.])

    def test_head_step_check_rejects_entire_chunk_before_any_send(self):
        actions = np.tile(self.start, (2, 1)); actions[1, 14] = .36
        with self.assertRaisesRegex(RuntimeError, '0.35 rad'): self.run_eef(actions)
        self.assertEqual(self.robot.robot.head, [])
        self.assertEqual(self.robot.robot.arms, [])

    def test_invalid_later_quaternion_rejects_entire_chunk_before_any_send(self):
        actions = np.tile(self.start, (2, 1)); actions[1, 3:7] = 0.
        with self.assertRaises(ValueError): self.run_eef(actions)
        self.assertEqual(self.robot.robot.head, [])
        self.assertEqual(self.robot.robot.arms, [])

    def test_head_boundary_behavior_preserved(self):
        self.start[16] = .5235; self.joints[wrapper.JOINT_NAMES[2]] = .5235
        target = self.start.copy(); target[16] = .52358
        with patch.object(wrapper, 'read_positions', return_value=(0., 0., .5235)+(0.,)*14):
            self.run_eef(target[None])
        self.assertAlmostEqual(self.robot.robot.head[-1][0][2], .5235)

    def test_observation_drift_below_point_three_is_allowed(self):
        with patch.object(wrapper, 'read_positions', return_value=(0.,)*3+(.299,)+(0.,)*13):
            self.run_eef()
        self.assertTrue(self.robot.robot.arms)

    def test_observation_drift_still_blocks_before_send(self):
        with patch.object(wrapper, 'read_positions', return_value=(0.,)*3+(.301,)+(0.,)*13):
            with self.assertRaisesRegex(RuntimeError, 'Robot moved'): self.run_eef()
        self.assertEqual(self.robot.robot.arms, [])

    def test_lateness_still_blocks_before_send(self):
        with patch.object(wrapper.time, 'monotonic', side_effect=[0., .111]):
            with self.assertRaisesRegex(RuntimeError, '100 ms late'): self.run_eef()
        self.assertEqual(self.robot.robot.head, [])

    def test_nonfinite_and_bad_shapes_rejected_without_send(self):
        for actions in [np.zeros((0, 17)), np.zeros((33, 17)), np.zeros((1, 19)), np.full((1, 17), np.nan)]:
            with self.subTest(shape=actions.shape):
                with self.assertRaises(ValueError): self.run_eef(actions)
        self.assertEqual(self.robot.robot.arms, [])

    def test_transport_errors_stop_remaining_commands(self):
        self.robot.robot.head_result = 1
        with self.assertRaisesRegex(RuntimeError, 'head'): self.run_eef()
        self.assertEqual(len(self.robot.robot.head), 1)
        self.assertEqual(self.robot.robot.arms, [])
        self.robot.robot.head_result = 0; self.robot.robot.arm_result = 1
        with self.assertRaisesRegex(RuntimeError, 'EEF'): self.run_eef()
        self.assertEqual(len(self.robot.robot.arms), 1)

    def test_cancellation_stops_before_next_tick(self):
        def cancel():
            if self.robot.robot.arms: raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.run_eef(check=cancel)
        self.assertEqual(len(self.robot.robot.arms), 1)
        self.assertEqual(len(self.robot.robot.head), 1)

    def test_legacy_qpos_keeps_timing_joint_order_and_gripper_clipping(self):
        action = np.arange(19, dtype=float)*.001; action[7] = .1; action[15] = -1.
        with patch.object(wrapper, 'send_servo') as sender:
            self.robot.execute(action[None], np.zeros(19))
        self.assertEqual(sender.call_count, 5)
        np.testing.assert_allclose(sender.call_args.args[1], np.r_[action[16:19], action[:7], action[8:15]])
        np.testing.assert_allclose(sender.call_args.args[2], [0., -.785])
        self.assertAlmostEqual(self.clock.now, .05)

    def test_legacy_qpos_arm_step_check_remains(self):
        action = np.zeros((1, 19)); action[0, 0] = .36
        with patch.object(wrapper, 'send_servo') as sender:
            with self.assertRaisesRegex(RuntimeError, '0.35 rad'):
                self.robot.execute(action, np.zeros(19))
        sender.assert_not_called()
