import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

ROOT = Path('/home/descfly/code/agibot_inference')
spec = importlib.util.spec_from_file_location('local_server_target', ROOT/'model_servers/mid_a1_final/serve_local.py')
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'tests/local_server_warmup_20261111')
        self.addCleanup(self.tmp.cleanup)
        self.bundle = Path(self.tmp.name)
        (self.bundle/'verification').mkdir()
        self.output = self.bundle/'output'
        self.output.mkdir()
        state = np.zeros(78, np.float32)
        for p in (0, 9, 18):
            state[p+3:p+9] = [1, 0, 0, 0, 1, 0]
        self.actions = np.repeat(state[None], 32, axis=0)
        rgb = np.zeros((5, 480, 640, 3), np.uint8)
        np.savez(self.bundle/'fixture.npz', state=state, head_rgb_history=rgb)
        np.savez(self.bundle/'reference.npz', actions=self.actions)
        (self.bundle/'verification/manifest.json').write_text(json.dumps([
            dict(task='g2/diguan', embodiment='g2', fixture='fixture.npz', reference='reference.npz')]))
        self.runtime = SimpleNamespace(info=dict(tasks={'g2/diguan': dict(mask=[True]*78)},
            client_image_preprocess=dict(tasks={'g2/diguan': dict(source_wh=[640,480],
                crop_xyxy=[218,130,538,450], output_wh=[320,320], resize='none')})))
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return dict(actions=self.actions.copy(), inference_seconds=.3)

    def test_cold_request_can_exceed_deadline_but_warm_request_must_pass(self):
        self.runtime.predict = self.predict
        with patch.object(target.time, 'monotonic', side_effect=[0., 12., 12., 12.4]):
            report = target.warm_up(self.runtime, self.bundle, self.output)
        self.assertTrue(report['passed'])
        self.assertEqual(report['requests'], 2)
        self.assertAlmostEqual(report['max_warm_request_s'], .4)
        self.assertEqual(self.requests[0]['observation']['head_rgb_history'].shape, (5,320,320,3))
        self.assertNotIn('state_history', self.requests[0]['observation'])

    def test_slow_warm_request_does_not_claim_readiness(self):
        self.runtime.predict = self.predict
        with patch.object(target.time, 'monotonic', side_effect=[0., 12., 12., 18.1]):
            with self.assertRaisesRegex(RuntimeError, 'latency'):
                target.warm_up(self.runtime, self.bundle, self.output)
        self.assertFalse(json.loads((self.output/'warmup.json').read_text())['passed'])

    def test_reference_drift_aborts_warmup(self):
        def changed(request):
            result = self.predict(request)
            result['actions'][0,0] += .01
            return result
        self.runtime.predict = changed
        with self.assertRaisesRegex(RuntimeError, 'reference'):
            target.warm_up(self.runtime, self.bundle, self.output)

    def test_nonfinite_actions_abort_warmup(self):
        self.actions[0,0] = np.nan
        self.runtime.predict = self.predict
        with self.assertRaisesRegex(RuntimeError, 'finite'):
            target.warm_up(self.runtime, self.bundle, self.output)

    def test_live_request_log_preserves_prediction_and_records_failure(self):
        self.runtime.predict = self.predict
        logged = target.LoggedRuntime(self.runtime, self.output)
        reply = logged.predict(dict(request_id='r1',task='g2/diguan',sample_ns=0))
        np.testing.assert_array_equal(reply['actions'], self.actions)
        def fail(_): raise ValueError('bad input')
        self.runtime.predict = fail
        with self.assertRaisesRegex(ValueError, 'bad input'):
            logged.predict(dict(request_id='r2',task='g2/diguan',sample_ns=0))
        rows = [json.loads(x) for x in (self.output/'requests.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows], ['ok','error'])
        self.assertEqual(rows[0]['request_id'], 'r1')
        self.assertIn('total_s', rows[0])

    def test_missing_task_fixture_cannot_claim_readiness(self):
        self.runtime.info['tasks']['g2/yifu'] = dict(mask=[True]*78)
        self.runtime.predict = self.predict
        with self.assertRaisesRegex(RuntimeError, 'every served task'):
            target.warm_up(self.runtime, self.bundle, self.output)
        self.assertEqual(self.requests, [])

    def test_progress_is_visible_when_imports_already_configured_logging(self):
        import logging
        stream = io.StringIO()
        with redirect_stdout(stream):
            target.configure_logging(self.output)
            logging.info('[READY] test')
        self.assertIn('[READY] test', stream.getvalue())
        self.assertIn('[READY] test', (self.output/'server.log').read_text())
        # Do not leave a handler pointing at a temporary directory/closed stream.
        logging.basicConfig(level=logging.CRITICAL, force=True)


if __name__ == '__main__':
    unittest.main()
