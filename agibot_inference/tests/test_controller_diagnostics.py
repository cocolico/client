"""Offline controller-fault recording tests; no robot or service is initialized."""
from contextlib import redirect_stderr
from datetime import datetime
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from replay_full_eef import assert_ready


class FakeRobot:
    """Supply only passive read methods used by assert_ready."""

    def __init__(self, *, error_code=2, collision_pairs_1=None,
                 collision_pairs_2=None, joint_error=None):
        self.status = SimpleNamespace(
            error_code=error_code, error_msg='fixture controller error',
            mode=3, control_mode=4,
            collision_pairs_1=list(collision_pairs_1 or []),
            collision_pairs_2=list(collision_pairs_2 or []),
            frame_names=['arm_l_end_link', 'arm_r_end_link'],
        )
        self.feedback = {
            'timestamp': 1770000000123456789,
            'states': [
                {'name': 'arm_l_joint1', 'position': 0.125,
                 'velocity': -0.25, 'effort': 0.5},
                {'name': 'arm_r_joint1', 'position': -0.375,
                 'velocity': 0.0, 'effort': 0.75},
            ],
        }
        self.joint_error = joint_error
        self.status_reads = 0
        self.joint_reads = 0

    def get_motion_control_status(self):
        self.status_reads += 1
        return self.status

    def get_joint_states(self):
        self.joint_reads += 1
        if self.joint_error is not None:
            raise self.joint_error
        return self.feedback


class ControllerDiagnosticsTests(unittest.TestCase):
    def api(self):
        # Assert missing functionality as RED without breaking test collection.
        self.assertIsNotNone(
            importlib.util.find_spec('controller_diagnostics'),
            'controller_diagnostics module has not been implemented',
        )
        return importlib.import_module('controller_diagnostics')

    def fault(self, robot):
        api = self.api()
        with self.assertRaises(api.ControllerFault) as caught:
            assert_ready(robot)
        self.assertIsInstance(caught.exception, RuntimeError)
        return caught.exception

    def load_standard_json(self, path):
        def reject_constant(value):
            raise AssertionError('Nonstandard JSON constant: ' + value)
        return json.loads(Path(path).read_text(), parse_constant=reject_constant)

    def record(self, fault, directory, context=None):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            path = self.api().record_controller_fault(
                fault, context={} if context is None else context,
                directory=directory,
            )
        self.assertIsInstance(path, Path)
        self.assertIn('CONTROLLER_FAULT_RECORDED', stderr.getvalue())
        self.assertIn(str(path), stderr.getvalue())
        return path, self.load_standard_json(path)

    def test_healthy_status_returns_none_without_reading_joint_feedback(self):
        robot = FakeRobot(error_code=0, joint_error=AssertionError('unexpected joint read'))
        self.assertIsNone(assert_ready(robot))
        self.assertEqual(robot.status_reads, 1)
        self.assertEqual(robot.joint_reads, 0)

    def test_error_code_two_with_empty_collision_pairs_still_records(self):
        robot = FakeRobot()
        earliest = time.monotonic_ns()
        fault = self.fault(robot)
        latest = time.monotonic_ns()
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = self.record(fault, Path(tmp) / 'records', {'phase': 'execute'})
        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['event'], 'controller_fault')
        diagnostic = payload['diagnostic']
        status = diagnostic['status']
        self.assertEqual(status['error_code'], 2)
        self.assertEqual(status['error_msg'], 'fixture controller error')
        self.assertEqual(status['mode'], 3)
        self.assertEqual(status['control_mode'], 4)
        self.assertEqual(status['collision_pairs_1'], [])
        self.assertEqual(status['collision_pairs_2'], [])
        self.assertEqual(status['frame_names'], ['arm_l_end_link', 'arm_r_end_link'])
        self.assertIn('fixture controller error', status['raw_repr'])
        self.assertEqual(diagnostic['joint_feedback']['timestamp'], 1770000000123456789)
        self.assertEqual(diagnostic['joint_feedback']['states'][0], {
            'name': 'arm_l_joint1', 'position': 0.125,
            'velocity': -0.25, 'effort': 0.5,
        })
        detected = datetime.fromisoformat(diagnostic['detected_at_utc'].replace('Z', '+00:00'))
        self.assertIsNotNone(detected.utcoffset())
        self.assertEqual(detected.utcoffset().total_seconds(), 0)
        self.assertLessEqual(earliest, diagnostic['detected_monotonic_ns'])
        self.assertLessEqual(diagnostic['detected_monotonic_ns'], latest)
        self.assertEqual(payload['context'], {'phase': 'execute'})
        self.assertIn('execution', payload)
        self.assertEqual(robot.joint_reads, 1)

    def test_collision_pairs_trigger_fault_when_error_code_is_zero(self):
        for field in ('collision_pairs_1', 'collision_pairs_2'):
            with self.subTest(field=field):
                robot = FakeRobot(error_code=0, **{field: ['arm_link_3:body_link']})
                fault = self.fault(robot)
                with tempfile.TemporaryDirectory() as tmp:
                    _, payload = self.record(fault, Path(tmp))
                self.assertEqual(payload['diagnostic']['status']['error_code'], 0)
                self.assertEqual(payload['diagnostic']['status'][field], ['arm_link_3:body_link'])

    def test_joint_read_failure_preserves_controller_fault_and_records_read_error(self):
        fault = self.fault(FakeRobot(joint_error=OSError('fixture feedback unavailable')))
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = self.record(fault, Path(tmp))
        self.assertEqual(payload['diagnostic']['status']['error_code'], 2)
        self.assertIn('fixture feedback unavailable',
                      json.dumps(payload['diagnostic']['joint_feedback_error']))

    def test_detection_snapshot_is_detached_from_later_status_and_joint_mutations(self):
        robot = FakeRobot(collision_pairs_1=['left:body'])
        robot.feedback['extra'] = {'sample': np.array([1.25, -2.5])}
        fault = self.fault(robot)
        robot.status.error_code = 0
        robot.status.error_msg = 'later healthy state'
        robot.status.collision_pairs_1.clear()
        robot.status.frame_names[0] = 'mutated_frame'
        robot.feedback['timestamp'] = 0
        robot.feedback['states'][0]['position'] = 99.0
        robot.feedback['extra']['sample'][0] = 77.0
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = self.record(fault, Path(tmp))
        diagnostic = payload['diagnostic']
        self.assertEqual(diagnostic['status']['error_code'], 2)
        self.assertEqual(diagnostic['status']['error_msg'], 'fixture controller error')
        self.assertEqual(diagnostic['status']['collision_pairs_1'], ['left:body'])
        self.assertEqual(diagnostic['status']['frame_names'][0], 'arm_l_end_link')
        self.assertEqual(diagnostic['joint_feedback']['timestamp'], 1770000000123456789)
        self.assertEqual(diagnostic['joint_feedback']['states'][0]['position'], 0.125)
        self.assertEqual(diagnostic['joint_feedback']['extra']['sample'], [1.25, -2.5])

    def test_cause_chain_records_execution_and_numpy_context_as_standard_json(self):
        fault = self.fault(FakeRobot())
        fault.execution = {'frame_index': np.int64(37), 'target': np.array([0.1, 0.2])}
        wrapped = RuntimeError('fixture execution interrupted')
        wrapped.__cause__ = fault
        context = {
            'chunk': np.int64(8), 'enabled': np.bool_(True),
            'targets': np.array([[1.25, 2.5], [3.75, 5.0]]),
            'nested': [np.float32(0.5), {'invalid': [np.nan, np.inf, -np.inf]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = self.record(wrapped, Path(tmp), context)
        self.assertEqual(payload['execution'], {'frame_index': 37, 'target': [0.1, 0.2]})
        self.assertEqual(payload['context']['chunk'], 8)
        self.assertIs(payload['context']['enabled'], True)
        self.assertEqual(payload['context']['targets'], [[1.25, 2.5], [3.75, 5.0]])
        self.assertEqual(payload['context']['nested'][0], 0.5)
        self.assertTrue(all(isinstance(value, str)
                            for value in payload['context']['nested'][1]['invalid']))
        chain = json.dumps(payload['exception_chain'])
        self.assertIn('fixture execution interrupted', chain)
        self.assertIn('ControllerFault', chain)

    def test_implicit_context_chain_also_finds_controller_fault(self):
        fault = self.fault(FakeRobot())
        wrapped = RuntimeError('fixture implicit wrapper')
        wrapped.__context__ = fault
        with tempfile.TemporaryDirectory() as tmp:
            _, payload = self.record(wrapped, Path(tmp))
        self.assertEqual(payload['diagnostic']['status']['error_code'], 2)
        self.assertIn('fixture implicit wrapper', json.dumps(payload['exception_chain']))

    def test_repeated_records_get_unique_complete_json_files(self):
        fault = self.fault(FakeRobot())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'records'
            paths = [self.record(fault, directory)[0] for _ in range(3)]
            self.assertEqual(len(set(paths)), 3)
            self.assertEqual(set(directory.iterdir()), set(paths))
            for path in paths:
                self.assertEqual(self.load_standard_json(path)['event'], 'controller_fault')

    def test_record_failure_keeps_original_fault_and_prints_full_json_fallback(self):
        fault = self.fault(FakeRobot())
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / 'not-a-directory'
            blocked.write_text('preserve this file')
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = self.api().record_controller_fault(
                    fault, context={'phase': 'failure-fixture'}, directory=blocked,
                )
            self.assertIsNone(result)
            self.assertEqual(blocked.read_text(), 'preserve this file')
            self.assertEqual(list(Path(tmp).iterdir()), [blocked])
        output = stderr.getvalue()
        self.assertIn('CONTROLLER_FAULT_RECORD_FAILED', output)
        payload, _ = json.JSONDecoder().raw_decode(output[output.index('{'):])
        self.assertEqual(payload['event'], 'controller_fault')
        self.assertEqual(payload['diagnostic']['status']['error_code'], 2)
        self.assertEqual(payload['context'], {'phase': 'failure-fixture'})
        self.assertIsInstance(fault, RuntimeError)

    def test_serialization_failure_cannot_replace_original_controller_fault(self):
        from unittest.mock import patch
        fault = self.fault(FakeRobot())
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stderr(stderr), patch.object(
                    self.api().json, 'dumps', side_effect=RuntimeError('fixture encoder failure')):
                try:
                    raise fault
                except self.api().ControllerFault:
                    result = self.api().record_controller_fault(
                        fault, context={}, directory=Path(tmp) / 'records')
                    self.assertIsNone(result)
                    with self.assertRaises(self.api().ControllerFault) as caught:
                        raise
                    self.assertIs(caught.exception, fault)
        self.assertIn('CONTROLLER_FAULT_RECORD_FAILED', stderr.getvalue())
        self.assertIn('fixture encoder failure', stderr.getvalue())

    def test_unrelated_runtime_error_creates_no_directory_or_output(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'must-not-exist'
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = api.record_controller_fault(
                    RuntimeError('ordinary failure'), context={}, directory=directory,
                )
            self.assertIsNone(result)
            self.assertFalse(directory.exists())
        self.assertEqual(stderr.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
