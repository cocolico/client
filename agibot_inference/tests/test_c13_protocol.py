"""Offline TCP contract tests; the endpoint below never loads a model or robot."""
import copy
import importlib.util
import json
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
sys.path.insert(0, str(COMMON))
from _ours_protocol import send_message, recv_message
from client_deployed78 import open_client78, Deployed78Client

IDENTITY = '3da2934a1640be01d15cf33187f47777cebeaf65228e55772387ebcb8a0cbf0b'


class Endpoint:
    def __init__(self, edit=None, bad_response=False):
        self.info = json.loads((ROOT/'tests/fixtures/c13_info_18874.json').read_text())
        if edit:
            edit(self.info)
        self.requests = []
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                while True:
                    try:
                        req = recv_message(self.request)
                    except (ConnectionError, OSError):
                        return
                    outer.requests.append(req)
                    if req['cmd'] == 'info':
                        result = outer.info
                    elif req['cmd'] == 'reset':
                        result = dict(ok=True, session_id='offline-session', request_id=req['request_id'],
                                      protocol_version=4, deployment_id=IDENTITY)
                    else:
                        assert req['cmd'] == 'infer'
                        assert set(req['observation']) == {'state', 'head_rgb'}
                        assert req['state_frame'] == 'chunk_camera'
                        assert req['session_id'] == 'offline-session'
                        state = req['observation']['state']
                        actions = np.tile(state, (32, 1))
                        actions[:, 0] = .25
                        result = dict(protocol_version=4, representation='egoav78_rot6d_columns',
                                      request_id=req['request_id'], session_id='offline-session',
                                      sample_ns=req['sample_ns'], deployment_id=IDENTITY,
                                      task=req['task'], embodiment=req['embodiment'], action_units='physical',
                                      action_frame='source' if bad_response else 'chunk_camera',
                                      action_mode='absolute', horizon=32, dt_s=.05,
                                      action_mask=outer.info['tasks'][req['task']]['mask'],
                                      anchor_state=state.copy(), actions=actions, inference_seconds=.1)
                    send_message(self.request, result)

        self.server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def registry(info):
    names = ['basket', 'book', 'diguan', 'yifu', 'dengpao']
    if COMMON.name == 'scripts':
        names.append('drink')
    return {name: dict(embodiment=info['tasks'][info['task_aliases'][name]]['embodiment'].lower(),
                       instruction=info['tasks'][info['task_aliases'][name]]['prompt'],
                       validity=list(map(int, info['tasks'][info['task_aliases'][name]]['mask'])))
            for name in names}


class C13Tests(unittest.TestCase):
    def make_client(self, endpoint, task='book', **overrides):
        self.assertIsNotNone(importlib.util.find_spec('client_joint_relative_c13'), 'C13 adapter is missing')
        from client_joint_relative_c13 import C13Client
        kwargs = dict(model='joint_relative_c13', task=task, tasks=registry(endpoint.info),
                      embodiment='3b' if task in ('basket', 'book') else 'g2',
                      expected_deployment=IDENTITY, expected_experiment='relative')
        kwargs.update(overrides)
        return C13Client(*endpoint.server.server_address, **kwargs)

    def state(self):
        state = np.zeros(78, np.float32)
        for start in (0, 9, 18):
            state[start+3:start+9] = [1, 0, 0, 0, 1, 0]
        state[0] = .7
        return state

    def test_supported_tasks_preserve_absolute_actions_and_session(self):
        e = Endpoint()
        try:
            for task, spec in registry(e.info).items():
                c = self.make_client(e, task)
                try:
                    actions = c.infer(self.state(), np.zeros((32, 32, 3), np.uint8),
                                      task=task, embodiment=spec['embodiment'], sample_ns=123)
                    np.testing.assert_array_equal(actions[:, 0], np.full(32, .25, np.float32))
                    self.assertEqual(actions.shape, (32, 78))
                    self.assertEqual(c.info['history_proprio'], False)
                    self.assertEqual(c.info['denoise_steps'], 10)
                    self.assertEqual(c.execution_info['coordinate_frame'], 'chunk_first_camera_fixed')
                finally:
                    c.close()
        finally:
            e.close()

    def test_wrong_identity_experiment_variant_history_and_units_fail_before_reset(self):
        for field, value in [('deployment_id', 'other'), ('experiment', 'absolute'),
                             ('variant', 'joint_relative_continuous13_history'), ('history_proprio', True),
                             ('checkpoint_step', 20000), ('xyz_unit', 'mm'),
                             ('model_action_mode', 'absolute_in_chunk_camera')]:
            with self.subTest(field=field):
                e = Endpoint(lambda info: info.update({field: value}))
                try:
                    with self.assertRaises(ValueError):
                        self.make_client(e)
                    self.assertEqual([r['cmd'] for r in e.requests], ['info'])
                finally:
                    e.close()

    def test_wrong_response_frame_rejected(self):
        e = Endpoint(bad_response=True)
        try:
            c = self.make_client(e)
            try:
                with self.assertRaises(ValueError):
                    c.infer(self.state(), np.zeros((32, 32, 3), np.uint8),
                            task='book', embodiment='3b', sample_ns=123)
            finally:
                c.close()
        finally:
            e.close()

    def test_factory_uses_only_explicit_c13_profile(self):
        e = Endpoint()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tasks = registry(e.info)
                (root/'tasks.json').write_text(json.dumps(dict(schema='egoav_78d_20261021_v1', tasks=tasks)))
                config = root/'endpoint.json'
                config.write_text(json.dumps(dict(wire_protocol='egoav78_c13_v4', model='joint_relative_c13',
                                                   tasks_file='tasks.json', deployment_id=IDENTITY,
                                                   expected_experiment='relative')))
                c = open_client78(*e.server.server_address, remote_config=config, model='joint_relative_c13',
                                  task='book', tasks=tasks, embodiment='3b')
                c.close()
                with self.assertRaises(ValueError):
                    open_client78(*e.server.server_address, remote_config=config, model='joint_relative',
                                  task='book', tasks=tasks, embodiment='3b')
        finally:
            e.close()

    def test_original_profile_still_rejects_c13_contract(self):
        e = Endpoint()
        try:
            with self.assertRaises(ValueError):
                Deployed78Client(*e.server.server_address, model='joint_relative', task='book',
                                 tasks=registry(e.info), embodiment='3b', expected_deployment=IDENTITY)
            self.assertEqual([r['cmd'] for r in e.requests], ['info'])
        finally:
            e.close()


if __name__ == '__main__':
    unittest.main()
