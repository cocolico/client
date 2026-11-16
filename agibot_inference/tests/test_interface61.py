import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from agibot_tasks import interface_spec, canonical_task
from recorded_sharpa import observation_from_row


class Interface61Tests(unittest.TestCase):
    def test_contract_and_legacy(self):
        spec = interface_spec('晾衣服', 'pi05')
        self.assertEqual((spec['action_dim'], spec['horizon'], spec['fps']), (61, 50, 20))
        self.assertEqual(len(spec['state_order']), 61)
        self.assertEqual(spec['state_order'][29], 'right_ee_x')
        self.assertEqual(spec['state_order'][58], 'idx11_head_joint1')
        self.assertEqual(interface_spec('clothes', 'ours')['action_dim'], 19)
        self.assertEqual(canonical_task('晾衣服'), 'yifu')
        with self.assertRaises(ValueError):
            interface_spec('yifu', 'ours')

    def test_recorded_rgb_and_feedback_not_action(self):
        import cv2
        _, encoded = cv2.imencode('.png', np.full((4, 6, 3), [0, 0, 255], dtype=np.uint8))
        row = {'observation.state': list(range(61)), 'action': [-1] * 61}
        for key in interface_spec('yifu', 'pi05')['camera_keys']:
            row['observation.images.' + key] = {'bytes': encoded.tobytes(), 'path': None}
        obs = observation_from_row(row)
        np.testing.assert_equal(obs['state'], np.arange(61))
        self.assertEqual(obs['images']['left_wrist_camera'].shape, (4, 6, 3))
        np.testing.assert_equal(obs['images']['head_color'][0, 0], [255, 0, 0])
        del row['observation.images.right_wrist_camera']
        with self.assertRaises(KeyError):
            observation_from_row(row)

    def test_client_full_50_frames_and_no_truncation(self):
        from run_ours_client import OursClient
        client = OursClient.__new__(OursClient)
        client.task, client.model, client.socket = 'yifu', 'pi05', None
        client.spec = interface_spec('yifu', 'pi05')
        expected = np.arange(50 * 61, dtype=np.float32).reshape(50, 61)
        with patch('run_ours_client.send_message'), patch('run_ours_client.recv_message', return_value={
            'actions': expected, 'inference_seconds': 1.0
        }):
            np.testing.assert_equal(client.policy_infer({})['actions'], expected)
        with patch('run_ours_client.send_message'), patch('run_ours_client.recv_message', return_value={
            'actions': expected[:32], 'inference_seconds': 1.0
        }):
            with self.assertRaises(ValueError):
                client.policy_infer({})


if __name__ == '__main__':
    unittest.main()
