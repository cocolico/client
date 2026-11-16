"""Pure fake-feedback tests: no GDK, hand SDK, network, or robot initialization."""
import importlib.util
from pathlib import Path
import threading
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


class G2CaptureTests(unittest.TestCase):
    def helper(self):
        path = ROOT/'scripts/abcd_g2_capture.py'
        self.assertTrue(path.exists(), 'Paired G2 observation reader is missing')
        spec = importlib.util.spec_from_file_location('abcd_g2_capture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_packet_uses_one_feedback_assembly_and_its_camera_anchor(self):
        helper = self.helper()
        obs = dict(state='measured-native61', images={'head_color': 'measured-rgb'},
                   joints={'body1': .1, 'body2': .2})
        calls = []
        context = types.SimpleNamespace(state71='camera-state71', base_from_camera='measured-anchor')
        class Robot:
            last_image_timestamps = {'head_color': 456}
            def get_observation61(self, hands):
                calls.append(hands)
                return obs
        class Adapter:
            def begin(self, state, *, sample_ns):
                self.args = state, sample_ns
                return context
        adapter = Adapter()
        reader = helper.g2_history_reader(Robot(), 'connected-hands', adapter,
            to78=lambda state: 'converted-'+state)
        packet = reader()
        self.assertEqual(calls, ['connected-hands'])
        self.assertEqual(packet['state'], 'converted-camera-state71')
        self.assertEqual(packet['head_rgb'], 'measured-rgb')
        self.assertEqual(packet['T_source_camera'], 'measured-anchor')
        self.assertEqual(packet['source_id'], 456)
        self.assertEqual(packet['image_timestamps_ns'], {'head_color': 456})
        self.assertIs(packet['observation'], obs)
        self.assertIs(packet['context'], context)
        self.assertEqual(adapter.args, ('measured-native61', packet['sample_ns']))

    def test_drink_history_uses_actual_body_feedback(self):
        helper = self.helper()
        obs = dict(state='measured-native61', images={'head_color': 'rgb'},
                   joints={'body1': .1, 'body2': .2})
        context = types.SimpleNamespace(state78='waist-enabled-state', base_from_camera='T')
        class Adapter:
            def begin(self, state, body, *, sample_ns):
                self.args = state, body
                return context
        robot = types.SimpleNamespace(last_image_timestamps={'head_color': 123},
            get_observation61=lambda hands: obs)
        adapter = Adapter()
        packet = helper.g2_history_reader(robot, object(), adapter, drink=True,
            body_names=('body1', 'body2'))()
        self.assertEqual(adapter.args, ('measured-native61', [.1, .2]))
        self.assertEqual(packet['state'], 'waist-enabled-state')

    def test_sdk_read_and_send_calls_are_serialized_without_changing_arguments(self):
        helper = self.helper()
        active = 0
        overlaps = []
        calls = []
        def operation(name, *args, **kwargs):
            nonlocal active
            active += 1
            overlaps.append(active)
            time.sleep(.005)
            calls.append((name, args, kwargs))
            active -= 1
            return name
        hand = types.SimpleNamespace(read_qpos=lambda: operation('read'),
            send_qpos=lambda *args, **kwargs: operation('send', *args, **kwargs))
        wrapped = helper.SerializedSDK(hand, threading.RLock())
        threads = [threading.Thread(target=wrapped.read_qpos),
                   threading.Thread(target=lambda: wrapped.send_qpos([1], [2], frame_index=3))]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(max(overlaps), 1)
        self.assertIn(('send', ([1], [2]), {'frame_index': 3}), calls)

    def test_close_waits_for_native_read_and_blocks_all_later_native_calls(self):
        helper = self.helper()
        started, release, closed = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def read():
            started.set()
            self.assertTrue(release.wait(1))
            calls.append('read-finished')
        def close():
            calls.append('closed')
            closed.set()
        target = types.SimpleNamespace(read_qpos=read, close=close,
            send_qpos=lambda *args: calls.append('send'),
            get_joint_states=lambda: calls.append('native-joints'))
        sdk = helper.SerializedSDK(target, threading.RLock())
        reader = threading.Thread(target=sdk.read_qpos)
        reader.start(); self.assertTrue(started.wait(1))
        closer = threading.Thread(target=sdk.close)
        closer.start()
        try:
            raced = closed.wait(.03)
        finally:
            release.set(); reader.join(1); closer.join(1)
        self.assertFalse(raced, 'native close raced an active feedback read')
        self.assertEqual(calls, ['read-finished', 'closed'])
        for function in (sdk.read_qpos, lambda: sdk.send_qpos([1], [2]), sdk.get_joint_states):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                function()
        sdk.close()
        self.assertEqual(calls, ['read-finished', 'closed'])


if __name__ == '__main__':
    unittest.main()
