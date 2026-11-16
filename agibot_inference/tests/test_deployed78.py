"""Exercise the real TCP client against captured deployment contracts."""
import copy
import json
from pathlib import Path
import socketserver
import sys
import tempfile
import threading
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT/'common' if (ROOT/'common').is_dir() else (
    ROOT/'src/infer/mixed78_common' if (ROOT/'src/infer/mixed78_common').is_dir() else ROOT/'scripts')
FIXTURES = ROOT/'evidence' if (ROOT/'evidence').is_dir() else ROOT/'tests/deployed78_info'
sys.path.insert(0, str(COMMON))
from _ours_protocol import send_message, recv_message
from egoav78_contract import WIRE_INFO


class Endpoint:
    def __init__(self, port=8868, info_edit=None, reply_edit=None):
        self.info = json.loads((FIXTURES / f'info_{port}.json').read_text())
        if info_edit: info_edit(self.info)
        self.requests = []
        outer = self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                while True:
                    try: req = recv_message(self.request)
                    except (OSError, ConnectionError): return
                    outer.requests.append(req)
                    if req['cmd'] == 'info': result = outer.info
                    elif req['cmd'] == 'reset':
                        result = dict(ok=True, session_id='session', request_id=req['request_id'],
                            deployment_id=outer.info['deployment_id'], protocol_version=4)
                    else:
                        # Strictly reproduce the actual server request boundary.
                        assert set(req['observation']) == {'state', 'head_rgb'}
                        assert req['state_frame'] == 'chunk_camera'
                        spec = outer.info['tasks'][req['task']]
                        assert req['embodiment'] == spec['embodiment']
                        state = req['observation']['state']
                        actions = np.tile(state, (32, 1)); actions[:, 0] = .25
                        result = dict(protocol_version=4, representation='egoav78_rot6d_columns',
                            request_id=req['request_id'], session_id='session', sample_ns=req['sample_ns'],
                            deployment_id=outer.info['deployment_id'], task=req['task'],
                            embodiment=req['embodiment'], action_units='physical',
                            action_frame='chunk_camera', action_mode='absolute', horizon=32, dt_s=.05,
                            action_mask=spec['mask'], anchor_state=state.copy(), actions=actions,
                            inference_seconds=.1)
                        if reply_edit: reply_edit(result)
                    send_message(self.request, result)
        self.server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def tasks():
    info = json.loads((FIXTURES / 'info_8869.json').read_text())
    return {short: dict(embodiment=info['tasks'][full]['embodiment'].lower(),
                       instruction=info['tasks'][full]['prompt'],
                       validity=list(map(int, info['tasks'][full]['mask'])))
            for short, full in info['task_aliases'].items()
            if short in ('basket', 'book', 'diguan', 'yifu', 'dengpao')}


class Deployed78Tests(unittest.TestCase):
    def client(self, endpoint, model='joint_relative', task='yifu', registry=None):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('client_deployed78'),
                             'Missing deployed78 protocol adapter')
        from client_deployed78 import Deployed78Client
        return Deployed78Client(*endpoint.server.server_address, model=model, task=task,
            tasks=registry or tasks(), embodiment='3b' if task in ('basket', 'book') else 'g2',
            expected_deployment=endpoint.info['deployment_id'])

    def state(self):
        state = np.zeros(78, np.float32)
        for start in (0, 9, 18): state[start+3:start+9] = [1,0,0,0,1,0]
        state[0] = .7
        return state

    def test_all_variants_and_tasks_no_double_relative_composition(self):
        for port, model in ((8868,'joint_relative'), (8869,'joint'), (8870,'tri')):
            e = Endpoint(port)
            try:
                for task, spec in tasks().items():
                    c = self.client(e, model, task)
                    try:
                        a = c.infer(self.state(), np.zeros((32,32,3),np.uint8),
                            task=task, embodiment=spec['embodiment'], sample_ns=123)
                        self.assertEqual(a.shape, (32,78))
                        np.testing.assert_array_equal(a[:,0], np.full(32,.25,np.float32))
                        self.assertEqual(c.spec['instruction'], spec['instruction'])
                        for key, value in WIRE_INFO.items():
                            self.assertEqual(c.execution_info[key], value)
                    finally: c.close()
            finally: e.close()

    def test_wrong_endpoint_instruction_mask_frame_and_layout_rejected_before_reset(self):
        mutations = [lambda i: i.update(model='tri'),
                     lambda i: i['tasks']['g2/yifu'].update(prompt='wrong'),
                     lambda i: i['tasks']['g2/yifu']['mask'].__setitem__(54,True),
                     lambda i: i.update(rotation='rows'),
                     lambda i: i['state_layout'].update(left_pose=[9,18])]
        for mutation in mutations:
            e = Endpoint(info_edit=mutation)
            try:
                with self.assertRaises(ValueError): self.client(e)
                self.assertEqual([r['cmd'] for r in e.requests], ['info'])
            finally: e.close()

    def test_bad_reply_rejected(self):
        mutations = [lambda r: r.update(action_frame='source'),
                     lambda r: r.update(action_mode='relative'),
                     lambda r: r.update(task='g2/diguan'),
                     lambda r: r.update(sample_ns=999),
                     lambda r: r['anchor_state'].__setitem__(0,9),
                     lambda r: r['action_mask'].__setitem__(54,True),
                     lambda r: r.update(actions=np.zeros((31,78),np.float32))]
        for mutation in mutations:
            e = Endpoint(reply_edit=mutation); c = self.client(e)
            try:
                with self.assertRaises(ValueError):
                    c.infer(self.state(), np.zeros((32,32,3),np.uint8),
                            task='yifu',embodiment='g2',sample_ns=123)
            finally: c.close(); e.close()

    def test_nonidentity_camera_and_wrong_task_never_sent(self):
        e = Endpoint(); c = self.client(e)
        try:
            for task, state in [('diguan',self.state()), ('yifu',self.state())]:
                if task == 'yifu': state[18] = 1
                with self.assertRaises(ValueError):
                    c.infer(state,np.zeros((32,32,3),np.uint8),task=task,embodiment='g2')
            self.assertEqual([r['cmd'] for r in e.requests], ['info','reset'])
        finally: c.close(); e.close()

    def test_explicit_factory_uses_pinned_training_prompt_without_changing_mask(self):
        from client_deployed78 import open_client78
        e=Endpoint(8869)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/'tasks.json').write_text(json.dumps(dict(
                    schema='egoav_78d_20261021_v1',tasks=tasks())))
                config=root/'endpoint.json'
                config.write_text(json.dumps(dict(wire_protocol='egoav78_v4',model='joint',
                    tasks_file='tasks.json',deployment_id=e.info['deployment_id'])))
                old=tasks();old['yifu']['instruction']='Old training text'
                c=open_client78(*e.server.server_address,remote_config=config,
                    model='joint',task='yifu',tasks=old,embodiment='g2')
                try:
                    self.assertTrue(c.spec['instruction'].startswith('Raise the right arm'))
                    self.assertEqual(c.spec['validity'],old['yifu']['validity'])
                finally:c.close()
                with self.assertRaises(ValueError):
                    open_client78(*e.server.server_address,remote_config=config,
                        model='tri',task='yifu',tasks=old,embodiment='g2')
        finally:e.close()

    def test_wrong_deployment_rejected_before_reset(self):
        from client_deployed78 import Deployed78Client
        e=Endpoint(8869)
        try:
            with self.assertRaises(ValueError):
                Deployed78Client(*e.server.server_address,model='joint',task='yifu',
                    tasks=tasks(),embodiment='g2',expected_deployment='not_this_server')
            self.assertEqual([r['cmd'] for r in e.requests],['info'])
        finally:e.close()


if __name__ == '__main__': unittest.main()
