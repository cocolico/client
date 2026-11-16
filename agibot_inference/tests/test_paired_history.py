"""CPU-only safety tests; no robot, camera, inference or reset imports."""
import threading
import time
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for common in (ROOT/'scripts', ROOT/'src/infer/mixed78_common'):
    if (common/'paired_history.py').exists():
        sys.path.insert(0, str(common))
        break

try:
    from paired_history import PairedHistory
except ImportError:
    PairedHistory = None


def packet(stamp, identity=0, anchor=None):
    state = np.arange(78, dtype=np.float32) / 100
    for start in (0, 9, 18):
        state[start:start + 3] = [1, 2, 3]
        state[start + 3:start + 9] = [1, 0, 0, 0, 1, 0]
    state[18:21] = 0
    return dict(sample_ns=stamp, source_id=identity, state=state,
                head_rgb=np.full((32, 48, 3), identity % 255, np.uint8),
                T_source_camera=np.eye(4) if anchor is None else anchor,
                context=object())


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(PairedHistory, "paired history implementation is missing")
        self.now = time.monotonic_ns()
        self.history = PairedHistory(lambda: None)

    def dense(self, anchors=None):
        for i in range(13):
            self.history.push(packet(self.now - (12-i)*100_000_000, i,
                                     None if anchors is None else anchors[i]))

    # Catches recapturing current, off-by-one temporal targets, and mutable buffers.
    def test_dense_pairs_preserve_exact_current_and_detach_input(self):
        self.dense()
        current = self.history.snapshot()
        prepared = self.history.prepare(current['sample_ns'])
        self.assertIs(current, self.history.snapshot())
        self.assertIs(prepared['head_rgb'], current['head_rgb'])
        self.assertEqual(prepared['selected_sample_ns'],
                         [self.now-x for x in (800_000_000,600_000_000,400_000_000,200_000_000,0)])
        self.assertEqual(prepared['valid_history_frames'], 4)
        self.assertEqual(prepared['history_offsets'], [-16, -12, -8, -4, 0])
        self.assertEqual(prepared['state_history'].shape, (5, 78))
        self.assertEqual(prepared['head_rgb_history'].shape, (5, 32, 48, 3))
        np.testing.assert_array_equal(prepared['head_rgb_history'][:,0,0,0], [4,6,8,10,12])
        self.assertFalse(prepared['history_is_pad'].any())
        self.assertTrue(all(abs(x) < 1e-8 for x in prepared['timestamp_errors_s']))
        with self.assertRaises(ValueError):
            current['state'][0] = 99
        with self.assertRaises(ValueError):
            current['state'].setflags(write=True)
        with self.assertRaises(TypeError):
            current['sample_ns'] = 99

    # Catches inverse/order errors and incorrect row-wise 6D rotations.
    def test_noncommuting_camera_anchors_transform_all_three_poses(self):
        rz = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        rx = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
        past = np.eye(4); past[:3,:3] = rz; past[:3,3] = [4,5,6]
        current = np.eye(4); current[:3,:3] = rx; current[:3,3] = [1,2,3]
        anchors = [past] * 12 + [current]
        self.dense(anchors)
        result = self.history.prepare(self.now)
        # inv(current) @ past maps (1,2,3) -> (1,6,-4), columns to
        # (0,0,-1), (-1,0,0). Literal expectations independently derived.
        for start in (0, 9, 18):
            expected = ([3,3,-3,0,0,-1,-1,0,0] if start == 18
                        else [1,6,-4,0,0,-1,-1,0,0])
            np.testing.assert_allclose(result['state_history'][0,start:start+9],
                                       expected, atol=1e-6)
        np.testing.assert_array_equal(result['state_history'][-1], result['state'])
        np.testing.assert_array_equal(result['state_history'][0,27:], result['state'][27:])

    # Catches hallucinated history where acquisition has not started yet.
    def test_cold_start_only_leading_padding_and_copies_are_owned(self):
        original = packet(self.now, 1)
        self.history.push(original)
        result = self.history.prepare(self.now)
        self.assertEqual(result['history_is_pad'].tolist(), [True,True,True,True,False])
        self.assertEqual(result['valid_history_frames'], 0)
        self.assertEqual(result['selected_sample_ns'], [self.now] * 5)
        self.assertIs(self.history.snapshot()['context'], original['context'])
        original['state'][:] = 99
        original['head_rgb'][:] = 99
        original['T_source_camera'][:] = 99
        np.testing.assert_array_equal(result['state'][:3], [1,2,3])
        self.assertEqual(int(result['head_rgb'][0,0,0]), 1)
        np.testing.assert_array_equal(self.history.snapshot()['T_source_camera'], np.eye(4))

    # Catches internal gaps being concealed by nearest available/duplicated images.
    def test_missing_middle_sparse_and_duplicate_images_fail_closed(self):
        for missing in ({5,6,7}, set(range(1,12))):
            history = PairedHistory(lambda: None)
            for i in range(13):
                if i not in missing:
                    history.push(packet(self.now-(12-i)*100_000_000, i))
            with self.assertRaisesRegex(RuntimeError, 'history|historical'):
                history.prepare(self.now)
        for i in range(13):
            accepted = self.history.push(packet(self.now-(12-i)*100_000_000, 7))
            self.assertEqual(accepted, i == 0)
        with self.assertRaises(RuntimeError):
            self.history.snapshot()

    # Catches accepting a frame selected from the future or a caller-made packet.
    def test_preparation_requires_stored_stamp_and_never_selects_after_current(self):
        self.dense()
        result = self.history.prepare(self.now-100_000_000)
        self.assertTrue(all(t <= self.now-100_000_000 for t in result['selected_sample_ns']))
        with self.assertRaises(KeyError):
            self.history.prepare(self.now+1)

    # Catches old/future frames, nonrigid anchors and invalid state entering control.
    def test_rejects_invalid_packets_and_stale_snapshot(self):
        invalid = []
        for field, value in [('state', np.zeros(77)), ('head_rgb', np.zeros((4,6,3))),
                             ('source_id', None), ('sample_ns', self.now+10_000_000_000)]:
            p = packet(self.now,1); p[field] = value; invalid.append(p)
        p = packet(self.now,1); p['T_source_camera'][0,0] = 2; invalid.append(p)
        p = packet(self.now,1); p['state'][35] = np.nan; invalid.append(p)
        for p in invalid:
            with self.assertRaises((ValueError, TypeError)):
                self.history.push(p)
        self.history.push(packet(self.now-500_000_000,1))
        with self.assertRaisesRegex(RuntimeError, 'stale|fresh'):
            self.history.snapshot()
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            self.history.push(packet(self.now-600_000_000,2))

    # Catches ancient image IDs becoming valid after buffer eviction, and bad
    # rotations entering the camera-frame transform despite a valid anchor.
    def test_image_sequence_and_pose_rotations_are_validated(self):
        self.dense()
        p = packet(time.monotonic_ns(), 0)
        with self.assertRaisesRegex(ValueError, 'source_id|image'):
            self.history.push(p)
        p = packet(time.monotonic_ns(), 13); p['source_id'] = 'not-a-sequence'
        with self.assertRaises((TypeError, ValueError)):
            self.history.push(p)
        for start in (0,9,18):
            p = packet(time.monotonic_ns(), 13)
            p['state'][start+3:start+9] = [1,0,0,1,0,0]
            with self.assertRaisesRegex(ValueError, 'rotation|SO'):
                self.history.push(p)
        p = packet(time.monotonic_ns(), 13)
        p['head_rgb'] = np.zeros((31,48,3), np.uint8)
        with self.assertRaisesRegex(ValueError, 'head_rgb'):
            self.history.push(p)

    # Catches input states expressed in another camera frame masquerading as
    # own-camera states; historical output heads are correctly nonidentity.
    def test_input_head_must_be_own_camera_identity(self):
        p = packet(self.now, 0)
        p['state'][18] = .1
        with self.assertRaisesRegex(ValueError, 'head|camera'):
            self.history.push(p)

    # Catches a synchronous capture that stops when inference/execution waits.
    def test_background_capture_continues_while_consumer_waits_and_closes(self):
        counter = iter(range(10000))
        def read():
            return packet(time.monotonic_ns(), next(counter))
        with PairedHistory(read, period=.02) as history:
            first = history.snapshot()
            time.sleep(.16)
            later = history.snapshot()
            self.assertGreater(later['sample_ns'], first['sample_ns']+80_000_000)
            history.check()
        final = history.snapshot()['sample_ns']
        time.sleep(.04)
        self.assertEqual(history.snapshot()['sample_ns'], final)

    # Catches swallowed acquisition exceptions and leaked producer threads.
    def test_background_failure_propagates_and_context_cleans_up(self):
        calls = iter(range(100))
        def read():
            if next(calls) > 0:
                raise OSError('camera unplugged')
            return packet(time.monotonic_ns(),1)
        with PairedHistory(read, period=.02) as history:
            time.sleep(.06)
            with self.assertRaisesRegex(RuntimeError, 'capture') as caught:
                history.check()
            self.assertIsInstance(caught.exception.__cause__, OSError)
            with self.assertRaises(RuntimeError):
                history.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'capture'):
            with PairedHistory(lambda: (_ for _ in ()).throw(OSError('initial failure'))):
                self.fail('must reject context entry')

    # Real SDK feedback reads can block for four seconds; cleanup must allow
    # that bounded read to finish instead of declaring a live-thread failure.
    def test_close_allows_sdk_read_longer_than_two_seconds(self):
        blocked = threading.Event()
        release = threading.Event()
        calls = iter(range(100))
        def read():
            identity = next(calls)
            if identity:
                blocked.set()
                release.wait(4.)
            return packet(time.monotonic_ns(), identity)
        history = PairedHistory(read, period=.01)
        history.__enter__()
        self.assertTrue(blocked.wait(.5))
        timer = threading.Timer(2.2, release.set)
        timer.start()
        try:
            history.close()
        finally:
            release.set(); timer.join(); history.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
