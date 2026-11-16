"""Pure interface tests; the heavyweight model is replaced before import."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from agibot_tasks import SHARPA_CAMERA_KEYS, SHARPA_TRAIN_CONFIG


class FakePolicy:
    def __init__(self, *args, **kwargs):
        self.actions = np.arange(50 * 61, dtype=np.float32).reshape(50, 61)

    def infer(self, payload):
        self.payload = payload
        return self.actions


class Server61Tests(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType('pi_model')
        fake.PI05ActiveOnly = fake.PI05Sharpa61 = FakePolicy
        spec = importlib.util.spec_from_file_location('isolated_pi05_server', ROOT / 'server/experiments/agibot/run_pi05_server.py')
        self.server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pi_model': fake}):
            spec.loader.exec_module(self.server)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name) / '50000'
        assets = self.checkpoint / 'assets/yifu'
        assets.mkdir(parents=True)
        (assets / 'norm_stats.json').write_text(json.dumps({'norm_stats': {
            'state': {'mean': [0] * 61}, 'actions': {'mean': [0] * 61}}}))
        self.predictor = self.server.Predictor('yifu', self.checkpoint, SHARPA_TRAIN_CONFIG, 50)
        self.obs = {'state': np.zeros(61, np.float32),
                    'images': {key: np.zeros((4, 6, 3), np.uint8) for key in SHARPA_CAMERA_KEYS}}

    def test_exact_camera_prompt_state_and_full_output(self):
        result = self.predictor.predict(self.obs)
        np.testing.assert_equal(result, self.predictor.model.actions)
        self.assertEqual(self.predictor.model.payload['prompt'], '晾衣服')
        self.assertIs(self.predictor.model.payload['images'], self.obs['images'])
        np.testing.assert_equal(self.predictor.model.payload['state'], self.obs['state'])

    def test_missing_camera_wrong_state_wrong_output_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(self.obs, images={'head_color': self.obs['images']['head_color']}))
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(self.obs, state=np.zeros(19)))
        for actions in (np.zeros((32, 61)), np.zeros((50, 19)), np.full((50, 61), np.nan)):
            self.predictor.model.actions = actions
            with self.assertRaises(ValueError):
                self.predictor.predict(self.obs)

    def test_wrong_config_and_horizon_rejected_before_model_load(self):
        for config, horizon in ((SHARPA_TRAIN_CONFIG, 32), ('legacy', 50)):
            with self.assertRaises(ValueError):
                self.server.Predictor('yifu', self.checkpoint, config, horizon)


if __name__ == '__main__':
    unittest.main()
