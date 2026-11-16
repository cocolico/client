"""Read-only real-data regression and dispatch checks; no robot execution."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import replay_tasks as m


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads(m.REGISTRY.read_text())

    def test_registered_paths_and_required_streams_all_exist(self):
        self.assertEqual(set(self.registry['tasks']), {'yifu', 'diguan', 'dengpao', 'drink'})
        for task, cfg in self.registry['tasks'].items():
            indices = [e['index'] for e in cfg['episodes']]
            self.assertEqual(len(indices), len(set(indices)))
            for entry in cfg['episodes']:
                resolved_task, resolved = m.resolve(task, str(entry['index']), self.registry)
                self.assertEqual(resolved_task, task)
                self.assertEqual(resolved, entry)
                path = Path(entry['path'])
                files = ('manifest.json', 'offline_replay.npz', 'meta/robot.urdf') if entry['format'] == 'public71' else (
                    'record/-tf.pbdat', 'record/-hal-joint_state.pbdat',
                    'record/-sharpa-left-state.pbdat', 'record/-sharpa-right-state.pbdat')
                for name in files:
                    self.assertTrue((path / name).is_file(), str(path / name))

    def test_mismatched_task_does_not_accept_another_tasks_path(self):
        path = self.registry['tasks']['diguan']['episodes'][0]['path']
        with self.assertRaises(ValueError):
            m.resolve('drink', path, self.registry)

    def test_alias_name_and_numeric_select_same_recording(self):
        _, e = m.resolve('滴管', '0', self.registry)
        self.assertEqual(m.resolve('diguan', e['name'], self.registry)[1], e)
        with self.assertRaises(ValueError):
            m.resolve('drink', '-1', self.registry)

    def test_alignment_uses_source_ns_and_refuses_missing_coverage(self):
        offset = 1_790_000_000_000_000_000
        stamps = offset + np.array([0, 10_000_000, 20_000_000], dtype=np.int64)
        values = np.array([[0., 1.], [2., 3.], [4., 5.]])
        actual, edge = m.align_joints(stamps, values, offset + np.array([5_000_000, 15_000_000]))
        np.testing.assert_array_equal(actual, [[1., 2.], [3., 4.]])
        self.assertEqual(edge, 0)
        with self.assertRaisesRegex(ValueError, 'coverage missing'):
            m.align_joints(stamps, values, [offset - 30_000_000, offset])

    def test_dry_run_never_calls_execution(self):
        fake = dict(report={}, paths={}, body_q=(), joint_q=(), hand_path=[])
        with patch.object(m, 'prepare_raw', return_value=fake), patch('replay_full_eef.execute') as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                m.main(['--task', 'diguan', '--episode', '0', '--dry-run'])
            execute.assert_not_called()

    def test_execute_delegates_full_drink_plan_to_existing_executor(self):
        fake = dict(report={}, paths={'left': [], 'right': []}, body_q=(0.,) * 5,
                    joint_q=(0.,) * 17, hand_path=[], head_path=[], body_path=[], arm_base_paths={})
        with patch.object(m, 'prepare_raw', return_value=fake), patch('replay_full_eef.execute') as execute:
            with patch('replay_feedback.control_locks', return_value=contextlib.nullcontext()):
                with contextlib.redirect_stdout(io.StringIO()):
                    m.main(['--task', 'drink', '--episode', '1', '--execute'])
            execute.assert_called_once()
            self.assertIs(execute.call_args.kwargs['head_path'], fake['head_path'])
            self.assertIs(execute.call_args.kwargs['body_path'], fake['body_path'])
            self.assertIs(execute.call_args.kwargs['arm_base_paths'], fake['arm_base_paths'])


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = json.loads(m.REGISTRY.read_text())
        cls.plans = {}
        # Poison hardware imports: these offline reads must remain usable without SDK initialization.
        with patch.dict(sys.modules, {'agibot_gdk': None, 'sharpa': None}):
            for task, ep in (('diguan', 0), ('dengpao', 1), ('drink', 1), ('yifu', 1)):
                _, entry = m.resolve(task, str(ep), registry)
                cls.plans[task] = m.prepare_raw(entry['path'], task)

    def test_all_tracks_have_identical_frame_counts_and_finite_values(self):
        for task, plan in self.plans.items():
            n = len(plan['paths']['left'])
            self.assertEqual(len(plan['paths']['right']), n, task)
            self.assertEqual(len(plan['head_path']), n, task)
            self.assertEqual(np.shape(plan['hand_path']), (n, 2, 22), task)
            self.assertTrue(np.isfinite(plan['hand_path']).all())
            np.testing.assert_array_equal(plan['head_path'][0], plan['joint_q'][:3])
            self.assertAlmostEqual(plan['report']['replay_hz'], 100., delta=.1)

    def test_only_drink_has_body_trajectory_and_waist_varies(self):
        for task, plan in self.plans.items():
            if task != 'drink':
                self.assertIsNone(plan['body_path'])
                self.assertIsNone(plan['arm_base_paths'])
        body = np.asarray(self.plans['drink']['body_path'])
        np.testing.assert_array_equal(np.ptp(body[:, :4], axis=0), [0.] * 4)
        self.assertGreater(np.ptp(body[:, 4]), np.deg2rad(25.))

    def test_drink_target_waist_and_arm_reexpression_roundtrip_each_frame(self):
        from replay_public_eef import HeadKinematics
        from replay import BODY_NAMES
        plan = self.plans['drink']
        fk = HeadKinematics(m.ROOT / 'assets/robot/agibot.urdf')
        # Every frame verifies the arm transform uses that frame's recorded target yaw.
        for i, body in enumerate(plan['body_path']):
            base_arm = fk.pose(dict(zip(BODY_NAMES, body)), 'arm_base_link')
            for side in ('left', 'right'):
                _, p, q = plan['arm_base_paths'][side][i]
                local = np.eye(4)
                local[:3, :3] = Rotation.from_quat(q).as_matrix()
                local[:3, 3] = p
                result = base_arm @ local
                _, p0, q0 = plan['paths'][side][i]
                np.testing.assert_allclose(result[:3, 3], p0, atol=1e-12)
                np.testing.assert_allclose(result[:3, :3], Rotation.from_quat(q0).as_matrix(), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
