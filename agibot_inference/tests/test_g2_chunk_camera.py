"""Coordinate adapter tests: no GDK instance, hand connection or motion."""
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
URDF = ROOT / 'assets/robot/agibot.urdf'


def native():
    x = np.zeros(61)
    x[:7] = [.4, .2, .9, 0, 0, 0, 1]
    x[29:36] = [.5, -.3, .8, 0, 0, 0, 1]
    x[7:29] = np.arange(22) / 100
    x[36:58] = np.arange(22) / 80
    x[58:61] = [.2, -.1, .3]
    return x


class ChunkCameraTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('g2_chunk_camera'),
                             'Missing G2 coordinate adapter')
        return importlib.import_module('g2_chunk_camera')

    def adapter(self):
        return self.api().G2ChunkCamera(URDF, [-.3, .6, -.3, .05, .1])

    def test_rotated_translated_anchor_has_hand_calculated_coordinates(self):
        m = self.api()
        # Camera at (1,2,3), rotated +90deg about base Z.
        anchor = np.array([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1.]])
        x = np.zeros((1,71))
        for start in (0,31,62):
            x[0,start:start+9] = [2,4,6,1,0,0,0,1,0]
        x[0,9:31] = .23; x[0,40:62] = .41
        y = m.transform71(x, np.linalg.inv(anchor))
        for start in (0,31,62):
            np.testing.assert_allclose(y[0,start:start+9], [2,-1,3,0,-1,0,1,0,0], atol=1e-12)
        np.testing.assert_array_equal(y[:,9:31], x[:,9:31])
        np.testing.assert_array_equal(y[:,40:62], x[:,40:62])

    def test_begin_head_identity_and_roundtrip_to_original_61(self):
        adapter = self.adapter(); x = native()
        chunk = adapter.begin(x, sample_ns=123)
        self.assertEqual(chunk.sample_ns, 123)
        np.testing.assert_allclose(chunk.state71[62:], [0,0,0,1,0,0,0,1,0], atol=1e-7)
        actions, report = adapter.decode(chunk, np.tile(chunk.state71, (32,1)))
        np.testing.assert_allclose(actions, np.tile(x,(32,1)), atol=2e-7)
        self.assertLess(report['head_max_position_error_mm'], .001)

    def test_anchor_and_fixed_body_do_not_change_after_next_observation(self):
        adapter = self.adapter(); x = native()
        first = adapter.begin(x, sample_ns=1)
        x[58:61] = [-.3, .05, -.2]
        second = adapter.begin(x, sample_ns=2)
        self.assertFalse(np.allclose(first.base_from_camera, second.base_from_camera))
        actual, _ = adapter.decode(first, first.state71[None])
        np.testing.assert_allclose(actual[0], native(), atol=2e-7)
        with self.assertRaises(ValueError):first.base_from_camera[0,3] = 0
        with self.assertRaises(ValueError):first.state71.setflags(write=True)

    def test_head_xyz_error_is_diagnostic_only(self):
        adapter = self.adapter(); chunk = adapter.begin(native(), sample_ns=1)
        actions = chunk.state71[None].copy(); actions[0,62] += .2
        actual, report = adapter.decode(chunk, actions)
        np.testing.assert_allclose(actual[0,58:61], native()[58:61], atol=1e-7)
        self.assertAlmostEqual(report['head_position_error_mm'][0], 200., places=3)
        self.assertEqual(len(report['head_target_xyz_base']), 1)
        self.assertEqual(len(report['head_fk_xyz_base']), 1)

    def test_head_rotation_changes_qpos_without_using_predicted_xyz(self):
        adapter = self.adapter(); chunk = adapter.begin(native(), sample_ns=1)
        expected = np.array([-.2,.1,.2])
        # Independent head orientation: fixed body FK followed by Rz Rx Ry.
        parent = adapter.fk.pose(dict(zip(adapter.body_names, adapter.body_target)), 'body_link5')
        base = parent[:3,:3] @ Rotation.from_euler('ZXY', expected).as_matrix()
        relative = chunk.base_from_camera[:3,:3].T @ base
        actions = chunk.state71[None].copy()
        actions[0,62:65] = [8,9,10]  # impossible position must not veto orientation
        actions[0,65:] = np.r_[relative[:,0],relative[:,1]]
        actual, report = adapter.decode(chunk, actions)
        np.testing.assert_allclose(actual[0,58:61], expected, atol=1e-7)
        self.assertGreater(report['head_max_position_error_mm'],1000)

    def test_nonorthonormal_predictions_are_projected_but_degenerate_are_rejected(self):
        adapter = self.adapter(); chunk = adapter.begin(native(), sample_ns=1)
        x = chunk.state71[None].copy()
        for start in (0,31,62):
            x[:,start+6:start+9] += .2*x[:,start+3:start+6]
            x[:,start+3:start+6] *= 2
        actual, _ = adapter.decode(chunk, x)
        np.testing.assert_allclose(actual[0], native(), atol=2e-7)
        for start in (0,31,62):
            bad = chunk.state71[None].copy();bad[:,start+3:start+9] = 0
            with self.assertRaises(ValueError):adapter.decode(chunk,bad)

    def test_invalid_shapes_nonfinite_and_missing_body_rejected(self):
        m = self.api()
        for body in (None, [0]*4, [float('nan')]*5):
            with self.assertRaises(ValueError):m.G2ChunkCamera(URDF,body)
        adapter = self.adapter(); chunk = adapter.begin(native(),sample_ns=1)
        for values in (np.zeros((1,76)),np.zeros((0,71)),np.full((2,71),np.nan)):
            with self.assertRaises(ValueError):adapter.decode(chunk,values)

    def test_predicted_head_angles_clip_to_joint_limits_with_diagnostics(self):
        from replay import HEAD3_MAX, JOINT_NAMES
        adapter = self.adapter(); chunk = adapter.begin(native(),sample_ns=1)
        cases = [(0,2.),(0,-2.),(1,.5),(1,-.5),(2,.6),(2,-.6)]
        for column,angle in cases:
            with self.subTest(joint=JOINT_NAMES[column],angle=angle):
                q = native()[58:61].copy(); q[column] = angle
                head = adapter.fk.head_pose(adapter.body_target,q)
                relative = np.linalg.inv(chunk.base_from_camera) @ head
                actions = np.tile(chunk.state71,(2,1)).astype(float)
                actions[1,62:] = self.api().pose_columns(relative[None])[0]
                original = actions.copy()
                actual,report = adapter.decode(chunk,actions)
                expected = np.clip(q,adapter.fk.bounds[:,0],adapter.fk.bounds[:,1])
                expected[2] = min(expected[2],HEAD3_MAX)
                np.testing.assert_allclose(actual[0],native(),atol=2e-7)
                np.testing.assert_allclose(actual[1,:58],native()[:58],atol=2e-7)
                np.testing.assert_allclose(actual[1,58:61],expected,atol=1e-7)
                np.testing.assert_array_equal(actions,original)
                self.assertEqual(report['head_joint_limit_policy'],'clip')
                self.assertEqual(report['head_joint_clipped_frames'],1)
                self.assertEqual(report['head_joint_clipped_values'],1)
                self.assertEqual(len(report['head_joint_clipping']),1)
                detail = report['head_joint_clipping'][0]
                self.assertEqual(detail['frame_index_base'],0)
                self.assertEqual(detail['frame'],1)
                self.assertEqual(detail['joint'],JOINT_NAMES[column])
                self.assertAlmostEqual(detail['predicted_rad'],angle)
                self.assertAlmostEqual(detail['applied_rad'],expected[column])
                self.assertAlmostEqual(detail['clipped_rad'],abs(angle-expected[column]))
                self.assertAlmostEqual(report['head_joint_clip_max_rad'],abs(angle-expected[column]))
                self.assertGreater(report['head_max_rotation_error_deg'],.1)

    def test_legal_head_predictions_report_no_clipping(self):
        adapter = self.adapter(); chunk = adapter.begin(native(),sample_ns=1)
        actual,report = adapter.decode(chunk,chunk.state71[None])
        np.testing.assert_allclose(actual[0],native(),atol=2e-7)
        self.assertEqual(report['head_joint_clipped_frames'],0)
        self.assertEqual(report['head_joint_clipped_values'],0)
        self.assertEqual(report['head_joint_clipping'],[])
        self.assertEqual(report['head_joint_clip_max_rad'],0.)

    def test_replay_still_rejects_joint_limits_and_xyz_mismatch(self):
        adapter = self.adapter()
        invalid = adapter.fk.head_pose(adapter.body_target,[2.,0,0])
        with self.assertRaisesRegex(ValueError,'URDF joint limits'):
            adapter.fk.solve(invalid[None],adapter.body_target)
        valid = adapter.fk.head_pose(adapter.body_target,native()[58:61]); valid[0,3]+=.2
        with self.assertRaisesRegex(ValueError,'incompatible'):
            adapter.fk.solve(valid[None],adapter.body_target)
        # Projection must not hide a mismatch in the underlying kinematics.
        invalid[0,3] += .2
        with self.assertRaisesRegex(ValueError,'incompatible'):
            adapter.fk.solve(invalid[None],adapter.body_target,clip_joint_limits=True)

    def test_existing_client_boundary_defaults_unchanged_and_adapter_is_optional(self):
        m = self.api()
        import run_mixed76_client as runner
        from mixed76_contract import encode_state,decode_actions
        obs = {'state':native()}
        state, context = runner.prepare_g2_observation(obs, sample_ns=1)
        np.testing.assert_array_equal(state,encode_state(native(),'g2'));self.assertIsNone(context)
        actions, report = runner.restore_g2_actions(np.tile(state,(32,1)))
        np.testing.assert_array_equal(actions,decode_actions(np.tile(state,(32,1)),'g2'))
        self.assertIsNone(report)
        adapter = m.G2ChunkCamera(URDF,[-.3,.6,-.3,.05,.1])
        state, context = runner.prepare_g2_observation(obs,sample_ns=2,adapter=adapter)
        actions, report = runner.restore_g2_actions(np.tile(state,(32,1)),context=context,adapter=adapter)
        np.testing.assert_allclose(actions,np.tile(native(),(32,1)),atol=2e-7)

    def test_recorded_540_frames_validate_with_fixed_command_body_without_hardware(self):
        m = self.api()
        sample = Path('/home/descfly/code/egoav/pico/data/lerobot_versions/'
                      'egoav_new_pilot_20261020/agibot_yifu/offline_replay.npz')
        self.assertTrue(hasattr(m,'validate_recorded'), 'Missing recorded-data validation')
        with patch.dict(sys.modules,{'agibot_gdk':None,'sharpa':None}):
            report = m.validate_recorded(sample,URDF,chunk_size=32)
        self.assertEqual(report['frames'],540)
        self.assertEqual(report['chunks'],17)
        self.assertFalse(report['robot_io'])
        self.assertFalse(report['server_verified'])
        self.assertLess(report['max_arm_position_error_m'],1e-6)
        self.assertLess(report['max_arm_rotation_error_rad'],1e-6)
        self.assertLess(report['max_head_qpos_error_rad'],1e-6)
        self.assertLess(report['max_hand_qpos_error_rad'],1e-6)

    def test_existing_cli_coordinate_check_never_connects_server_or_robot(self):
        self.api()
        import run_mixed76_client as runner
        sample = '/home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu/offline_replay.npz'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'report.json'
            args = ['client','--model','pi05','--task','yifu','--embodiment','g2',
                    '--coordinate-check',sample,'--output',str(output)]
            with patch.object(sys,'argv',args), patch.object(runner,'inference_connection',
                    side_effect=AssertionError('Offline check opened transport')):
                runner.main()
            import json
            self.assertEqual(json.loads(output.read_text())['frames'],540)

    def test_recorded_validation_detects_wrong_observation_arm_rotation(self):
        from dataclasses import replace
        m = self.api()
        original = m.G2ChunkCamera.begin
        def wrong_rotation(adapter,*args,**kwargs):
            context = original(adapter,*args,**kwargs)
            state = context.state71.copy()
            r = Rotation.from_euler('z',.2).as_matrix()
            state[3:6] = r @ state[3:6];state[6:9] = r @ state[6:9]
            return replace(context,state71=state)
        sample = '/home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu/offline_replay.npz'
        with patch.object(m.G2ChunkCamera,'begin',wrong_rotation):
            with self.assertRaisesRegex(ValueError,'Recorded conversion mismatch'):
                m.validate_recorded(sample,URDF)


if __name__ == '__main__':unittest.main()
