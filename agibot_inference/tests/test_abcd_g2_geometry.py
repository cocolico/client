"""End-to-end paired-reader/current-camera geometry with pure fake feedback."""
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class G2HistoryGeometryTests(unittest.TestCase):
    def test_actual_native61_mapping_and_past_poses_share_current_camera(self):
        from abcd_g2_capture import g2_history_reader
        from g2_chunk_camera import G2ChunkCamera
        from paired_history import PairedHistory
        past = np.eye(4)
        current = np.array([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1.]])
        adapter = G2ChunkCamera.__new__(G2ChunkCamera)
        adapter.body_target = np.zeros(5)
        adapter.fk = types.SimpleNamespace(head_pose=lambda body, head: current if head[0]>1.15 else past)
        class Robot:
            last_image_timestamps = {}
            index = 0
            def get_observation61(self, hands):
                state = np.zeros(61)
                state[:7] = [.4,.2,.9,0,0,0,1]
                state[29:36] = [.5,-.3,.8,0,0,0,1]
                state[7:29] = self.index/100
                state[36:58] = .2+self.index/100
                state[58] = self.index/10
                self.last_image_timestamps['head_color'] = self.index
                return dict(state=state, joints={}, images={
                    'head_color': np.full((32,32,3),self.index,np.uint8)})
        robot = Robot()
        reader = g2_history_reader(robot, object(), adapter)
        history = PairedHistory(reader)
        now = time.monotonic_ns()
        for i in range(13):
            robot.index = i
            with patch('abcd_g2_capture.time.monotonic_ns', return_value=now-(12-i)*100_000_000):
                packet = reader()
            history.push(packet)
        prepared = history.prepare(now)
        for frame in prepared['state_history']:
            np.testing.assert_allclose(frame[:18], prepared['state'][:18], atol=1e-6)
        np.testing.assert_allclose(prepared['state'][:3], [-1.8,.6,-2.1], atol=1e-6)
        np.testing.assert_allclose(prepared['state_history'][0,18:27],
            [-2,1,-3,0,-1,0,1,0,0], atol=1e-6)
        np.testing.assert_allclose(prepared['state_history'][:,32], [.04,.06,.08,.10,.12])
        np.testing.assert_array_equal(prepared['head_rgb_history'][:,0,0,0], [4,6,8,10,12])
        self.assertEqual(prepared['valid_history_frames'], 4)


if __name__ == '__main__': unittest.main()
