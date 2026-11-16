from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from fake_sharpa_sdk import SDK, error

CONFIG = Path(__file__).resolve().parents[1] / 'configs/sharpa_hands.json'


class NativeHandTests(unittest.TestCase):
    def setUp(self):
        from sharpa_hands import SharpaHands
        self.sdk = SDK()
        self.hands = SharpaHands.from_config(CONFIG, sdk=self.sdk)
        self.addCleanup(self.hands.close)

    def test_construction_never_loads_sdk_connects_or_imports_hardware_package(self):
        self.assertEqual(self.sdk.instances, 0)
        self.assertNotIn('litchi_hardware', sys.modules)
        self.assertNotIn('sharpa', sys.modules)
        with self.assertRaises(RuntimeError): self.hands.read_qpos()

    def test_connect_once_and_send22_directly_without_mode_checks_per_frame(self):
        self.hands.connect(); self.hands.connect()
        left = self.sdk.waves['C8519039C852']; right = self.sdk.waves['CE569038CE55']
        gets = (list(left.gets), list(right.gets))
        ql = np.arange(22)*.01; qr = np.arange(22)*.02
        self.hands.send_qpos(ql, qr)
        self.hands.send_qpos(ql, qr)
        np.testing.assert_allclose(left.commands[-1][0], ql)
        np.testing.assert_allclose(right.commands[-1][0], qr)
        self.assertEqual(self.sdk.instances, 1)
        self.assertEqual(self.sdk.connections, [('C8519039C852', True), ('CE569038CE55', True)])
        self.assertEqual((left.gets, right.gets), gets)
        self.assertEqual((left.starts, right.starts), (1, 1))
        self.assertEqual(left.commands[-1][1], True)
        self.assertEqual((left.sets, right.sets), ([], []))

    def test_startup_only_writes_changed_settings(self):
        left = self.sdk.waves['C8519039C852']
        left.settings['control_source'] = 7
        left.settings['control_mode'] = 10
        left.settings['enable_state'] = False
        self.hands.connect()
        self.assertIn(('set_control_source', 4), left.sets)
        self.assertIn(('set_control_mode', 13), left.sets)
        self.assertIn(('set_enable_state', True), left.sets)

    def test_actual_feedback_is_not_cached_target(self):
        self.hands.connect()
        self.hands.send_qpos(np.zeros(22), np.zeros(22))
        self.sdk.waves['C8519039C852'].values = [.12]*22
        left, right = self.hands.read_qpos()
        np.testing.assert_allclose(left, .12)
        np.testing.assert_allclose(right, 0.)

    def test_bad_shape_nonfinite_fail_before_any_command(self):
        self.hands.connect()
        for target in (np.zeros(21), np.full(22, np.nan)):
            with self.assertRaises(ValueError): self.hands.send_qpos(np.zeros(22), target)
        self.assertFalse(any(w.commands for w in self.sdk.waves.values()))

    def test_invalid_feedback_reports_raw_state_without_retry_or_commands(self):
        self.hands.connect()
        for values, bad_indices in (([], []), ([float('nan')]+[0.]*21, [0])):
            self.sdk.waves['CE569038CE55'].values = values
            with self.assertLogs('sharpa_hands', level='ERROR'):
                with self.assertRaises(ValueError) as ctx:
                    self.hands.read_qpos()
            info = ctx.exception.diagnostic
            self.assertEqual(info['side'], 'right')
            self.assertEqual(info['count'], len(values))
            self.assertEqual(info['nonfinite_indices'], bad_indices)
            self.assertEqual(info['sequence'], 2)
            self.assertEqual(info['timestamp'], 1234)
            self.assertEqual(info['host_recv_time_us'], 5678)
            self.assertFalse(any(w.commands for w in self.sdk.waves.values()))

    def test_sdk_error_keeps_code_side_frame_duration_and_stops_other_hand(self):
        self.hands.connect()
        self.sdk.waves['C8519039C852'].send_error = error(0x501, 'Joint angle limit exceeded')
        with self.assertLogs('sharpa_hands', level='ERROR'):
            with self.assertRaises(RuntimeError) as ctx:
                self.hands.send_qpos(np.zeros(22), np.zeros(22), frame_index=4, subframe_index=2)
        info = ctx.exception.diagnostic
        self.assertEqual((info['side'], info['frame_index'], info['subframe_index']), ('left', 4, 2))
        self.assertEqual(info['operation'], 'set_joint_position')
        self.assertEqual(info['sdk_error_code'], 0x501)
        self.assertGreaterEqual(info['elapsed_ms'], 0.)
        self.assertEqual(self.sdk.waves['CE569038CE55'].commands, [])

    def test_partial_connect_failure_releases_left(self):
        self.sdk.fail_serial = 'CE569038CE55'
        with self.assertRaisesRegex(RuntimeError, 'right unavailable'): self.hands.connect()
        self.assertIn('C8519039C852', self.sdk.disconnections)
        with self.assertRaises(RuntimeError): self.hands.read_qpos()

    def test_close_uses_native_stop_disconnect_not_disable_or_disconnect_all(self):
        self.hands.connect(); self.hands.close(); self.hands.close()
        self.assertCountEqual(self.sdk.disconnections, ['C8519039C852', 'CE569038CE55'])
        for wave in self.sdk.waves.values():
            self.assertEqual(wave.stops, 1)
            self.assertNotIn(('set_enable_state', False), wave.sets)

    def test_disconnect_still_attempted_when_stop_throws(self):
        self.hands.connect()
        with patch.object(self.sdk.waves['C8519039C852'], 'stop', side_effect=RuntimeError('stop failed')):
            with self.assertRaisesRegex(RuntimeError, 'stop failed'): self.hands.close()
        self.assertCountEqual(self.sdk.disconnections, ['C8519039C852', 'CE569038CE55'])

    def test_second_adapter_cannot_take_over_or_close_existing_sdk_connections(self):
        from sharpa_hands import SharpaHands
        self.hands.connect()
        second = SharpaHands.from_config(CONFIG, sdk=self.sdk)
        with self.assertRaisesRegex(RuntimeError, 'already connected'):
            second.connect()
        second.close()
        self.assertEqual(self.sdk.disconnections, [])
        self.hands.send_qpos([.1]*22, [.2]*22)
        self.assertEqual(self.hands.read_qpos(), ((.1,)*22, (.2,)*22))


if __name__ == '__main__': unittest.main()
