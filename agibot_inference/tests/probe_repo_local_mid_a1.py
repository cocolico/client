"""Actual installed config/client with a loopback simulated model; no robot IO."""
from pathlib import Path
import json
import socket
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np

repo = Path('/home/descfly/code/agibot_inference')
bundle = repo/'model_servers/mid_a1_final'
sys.path[:0] = [str(repo/'scripts')]
from inference_connection import inference_connection
from _ours_protocol import recv_message, send_message
from client_deployed78 import open_client78
from paired_history import PairedHistory
from egoav78_contract import load_tasks

config_path = repo/'configs/deployed_mid_a1_final_local.json'
config = json.loads(config_path.read_text())
receipt = json.loads((bundle/'verification/LAST_TEST_SERVER.json').read_text())
# Old receipts remain immutable evidence after the directory move.
info_path = bundle/'logs'/Path(receipt['output']).name/'info.json'
info = json.loads(info_path.read_text())
info['simulated'] = True
tasks = load_tasks(config_path.parent/config['tasks_file'])
listener = socket.socket()
listener.settimeout(5)
listener.bind((config['server_host'], config['server_port']))
listener.listen()
errors, requests = [], []

def serve():
    try:
        with listener.accept()[0] as conn:
            conn.settimeout(5)
            while True:
                try:
                    request = recv_message(conn)
                except (OSError, EOFError, ConnectionError):
                    break
                requests.append(request)
                if request['cmd'] == 'info':
                    reply = info
                elif request['cmd'] == 'reset':
                    reply = dict(ok=True, request_id=request['request_id'], session_id='local-offline',
                                 protocol_version=4, deployment_id=info['deployment_id'])
                else:
                    observation = request['observation']
                    assert 'state_history' not in observation
                    assert len(observation['head_rgb_history']) == 5
                    assert request['task'] in ('g2/diguan','g2/yifu','g2/dengpao','g2/drink_head_only')
                    assert request['state_frame'] == 'chunk_camera'
                    reply = dict(protocol_version=4, representation='egoav78_rot6d_columns',
                        request_id=request['request_id'], session_id=request['session_id'],
                        sample_ns=request['sample_ns'], deployment_id=info['deployment_id'],
                        task=request['task'], embodiment='g2', action_frame='chunk_camera',
                        action_mode='absolute', action_units='physical',
                        action_mask=info['tasks'][request['task']]['mask'], horizon=32, dt_s=.05,
                        anchor_state=observation['state'], inference_seconds=0.,
                        observed_latent_shape=[1,48,2,20,20],
                        actions=np.repeat(observation['state'][None], 32, axis=0))
                send_message(conn, reply)
    except BaseException as error:
        errors.append(repr(error))

for task in ('diguan','yifu','dengpao','drink'):
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = None
    try:
        args = SimpleNamespace(inference='remote', remote_config=config_path)
        with inference_connection(args) as address:
            assert address == ('127.0.0.1', 19011)
            client = open_client78(*address, remote_config=config_path, model='c0_lr',
                                  task=task, tasks=tasks, embodiment='g2', allow_simulated=True, timeout=5)
            capture = PairedHistory(lambda: None)
            client.attach_history(capture)
            for index in range(2):
                state = np.zeros(78, np.float32)
                for pose in (0,9,18):
                    state[pose+3:pose+9] = [1,0,0,0,1,0]
                state[0] = .2 + .01*index
                state[32], state[55] = .1, .2
                rgb = np.zeros((480,640,3), np.uint8)
                stamp = time.monotonic_ns()
                capture.push(dict(state=state, head_rgb=rgb, sample_ns=stamp,
                                  source_id=index, T_source_camera=np.eye(4)))
                actions = client.infer(state, rgb, task=task, embodiment='g2', sample_ns=stamp)
                np.testing.assert_array_equal(actions, np.repeat(state[None],32,axis=0))
    finally:
        if client:
            client.close()
        thread.join(timeout=6)
        assert not thread.is_alive()
listener.close()
assert not errors, errors
assert [q['cmd'] for q in requests] == ['info','reset','infer','infer']*4
result = dict(passed=True, mode='simulated_model_actual_installed_client_and_config',
    tasks=['diguan','yifu','dengpao','drink'], rpc_requests=len(requests), inference_requests=8,
    action_shape=[32,78], transport='tcp', address='127.0.0.1:19011',
    checkpoint_sha256=config['checkpoint_sha256'], deployment_id=config['deployment_id'],
    robot_io=False, real_model_inference_verified=False, host_fault_unresolved=False)
(bundle/'REPO_LOCAL_RPC_ACCEPTANCE.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
