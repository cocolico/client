"""Hardware-free expiry diagnostics and persistence regression."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paired_history import PairedHistory, PairedObservationStale
from controller_diagnostics import record_controller_fault
from test_paired_history import packet

class StaleDiagnosticsTests(unittest.TestCase):
    def test_expiry_records_blocked_reader_without_hardware_access(self):
        def reader():
            raise AssertionError('diagnostics must not read hardware')
        reader.diagnostics = lambda: dict(stage='camera_wait', pending_cameras=['left_wrist_camera'])
        history = PairedHistory(reader)
        stamp = time.monotonic_ns()
        history.push(packet(stamp, 7))
        with patch('paired_history.time.monotonic_ns', return_value=stamp + 300_000_000):
            history.check()
        with patch('paired_history.time.monotonic_ns', return_value=stamp + 300_000_001):
            with self.assertRaises(PairedObservationStale) as caught:
                history.check()
        error = caught.exception
        self.assertEqual(error.diagnostic['source_id'], 7)
        self.assertEqual(error.diagnostic['capture']['pending_cameras'], ['left_wrist_camera'])
        with tempfile.TemporaryDirectory() as directory:
            path = record_controller_fault(error, context=dict(task='yifu'), directory=directory)
            data = json.loads(path.read_text())
        self.assertEqual(data['event'], 'paired_observation_stale')
        self.assertEqual(data['context']['task'], 'yifu')
        self.assertGreater(data['diagnostic']['age_s'], .3)

    def test_diagnostic_callback_failure_preserves_stale_error(self):
        def reader(): pass
        def broken(): raise ValueError('diagnostic unavailable')
        reader.diagnostics = broken
        history = PairedHistory(reader)
        history.push(packet(time.monotonic_ns() - 400_000_000))
        with self.assertRaises(PairedObservationStale) as caught:
            history.check()
        self.assertIn('diagnostic unavailable', caught.exception.diagnostic['capture_diagnostic_error'])
