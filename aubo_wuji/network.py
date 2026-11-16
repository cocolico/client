"""Strict XFlow client; one persistent TCP session and an owned SSH tunnel."""
from contextlib import contextmanager
from pathlib import Path
import ipaddress
import json
import re
import shlex
import socket
import subprocess
import tempfile
import time
import uuid
import numpy as np
from vendor.wire_v4 import send_message, recv_message

PROTOCOL = 'xflow_kp21_chunk_v1'


def checked_host(host):
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', host):
        raise ValueError('Invalid host / SSH alias')
    return host


@contextmanager
def connection(cfg):
    if cfg['transport'] == 'tcp':
        yield checked_host(cfg['host']), int(cfg['port'])
        return
    if cfg['transport'] != 'ssh': raise ValueError('Unknown transport')
    alias = checked_host(cfg['ssh_alias'])
    host, port = cfg.get('host'), cfg.get('port')
    if not host:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'StrictHostKeyChecking=yes', alias, 'cat -- '+shlex.quote(cfg['endpoint_file'])],
            capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError('Cannot read XFlow endpoint; supply server.host/port or obtain file read permission: '+result.stderr.strip())
        endpoint = json.loads(result.stdout)
        if endpoint.get('protocol') != PROTOCOL: raise ValueError('Endpoint protocol mismatch')
        host, port = endpoint['host'], endpoint['port']
    host = checked_host(host)
    if type(port) is not int or not 1 <= port <= 65535: raise ValueError('Bad server port')
    with tempfile.TemporaryDirectory(prefix='aubo-xflow-') as directory:
        control = str(Path(directory)/'ssh.sock')
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0)); local = reservation.getsockname()[1]
        cmd = ['ssh', '-M', '-S', control, '-f', '-N', '-T', '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ControlPersist=no',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
               '-L', f'127.0.0.1:{local}:{host}:{port}', alias]
        try:
            # File stderr avoids a background SSH process retaining subprocess pipes.
            with open(Path(directory)/'ssh.log', 'w') as log:
                result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=log, timeout=20)
            if result.returncode:
                raise RuntimeError((Path(directory)/'ssh.log').read_text())
            yield '127.0.0.1', local
        finally:
            if Path(control).exists():
                subprocess.run(['ssh', '-S', control, '-O', 'exit', alias],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=10, check=True)


def validate_observation(obs):
    for key in ('head_rgb', 'history_rgb'):
        if key not in obs and key == 'history_rgb': continue
        x = np.asarray(obs[key])
        if x.dtype != np.uint8 or x.shape != (400, 640, 3):
            raise ValueError(key+' must be raw uint8 RGB [400,640,3]')
    x = np.asarray(obs['keypoints'])
    if x.shape != (2, 21, 3) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Expected finite float32 current keypoints [2,21,3]')


class Client:
    def __init__(self, host, port, cfg):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(cfg['timeout_s'])
        self.cfg = cfg; self.last_ns = -1
        try:
            self.info = self.call(dict(cmd='info'))
            if (self.info.get('protocol'), self.info.get('protocol_version'), self.info.get('horizon')) != (PROTOCOL, 4, 32):
                raise ValueError('Unsupported XFlow info')
            self.model = cfg.get('model') or self.info['default_model']
            self.spec = self.info['models'][self.model]
            if cfg.get('expected_run_name') and cfg['expected_run_name'] not in Path(self.spec['weights']).parts:
                raise ValueError('Model weights do not belong to requested training run')
            if self.spec['contract'] not in ('humanhand', 'realrobot'): raise ValueError('Unknown time contract')
            self.times = np.asarray(self.spec['keypoint_time_s'], float)
            if self.times.shape != (32,) or not np.isfinite(self.times).all() or self.times[0] <= 0 or np.any(np.diff(self.times) <= 0):
                raise ValueError('Invalid model prediction times')
            if type(self.spec['history_offset_frames']) is not int or not 0 <= self.spec['history_offset_frames'] <= 60:
                raise ValueError('Invalid history offset')
            expected = cfg.get('expected_deployment_id')
            if expected and self.spec['deployment_id'] != expected: raise ValueError('Unexpected model deployment')
            if cfg['task'] not in self.info['tasks']: raise ValueError('Task unavailable')
            request_id = uuid.uuid4().hex
            reset = self.call(dict(cmd='reset', request_id=request_id))
            if reset.get('request_id') != request_id or not isinstance(reset.get('session_id'), str) or not reset['session_id']:
                raise ValueError('Invalid reset response')
            self.session = reset['session_id']
        except BaseException:
            self.close(); raise

    def call(self, msg):
        send_message(self.sock, msg); response = recv_message(self.sock)
        if 'error' in response: raise RuntimeError('XFlow error: '+str(response['error']))
        return response

    def infer(self, observation, sample_ns):
        validate_observation(observation)
        if type(sample_ns) is not int or sample_ns <= self.last_ns: raise ValueError('sample_ns must increase')
        request_id = uuid.uuid4().hex
        # Consume timestamps even when a call fails; never retry the same sample.
        self.last_ns = sample_ns
        out = self.call(dict(cmd='infer', request_id=request_id, session_id=self.session,
            sample_ns=sample_ns, model=self.model, task=self.cfg['task'], seed=0,
            num_inference_steps=self.cfg['num_inference_steps'], observation=observation))
        expected = dict(protocol=PROTOCOL, protocol_version=4, request_id=request_id,
            session_id=self.session, sample_ns=sample_ns, model=self.model,
            deployment_id=self.spec['deployment_id'], weights=self.spec['weights'],
            contract=self.spec['contract'], task=self.cfg['task'], hand_order=['left','right'],
            frame='cam0_color_optical_frame', units='metres', horizon=32)
        for key, value in expected.items():
            if out.get(key) != value: raise ValueError('Response mismatch: '+key)
        kp = np.asarray(out['keypoints'])
        if kp.dtype != np.float32 or kp.shape != (32,2,21,3) or not np.isfinite(kp).all():
            raise ValueError('Invalid predicted keypoints')
        if not np.allclose(out['keypoint_time_s'], self.times, rtol=1e-6, atol=1e-8):
            raise ValueError('Prediction time contract changed')
        if np.asarray(out.get('keypoint_time_s')).shape != (32,): raise ValueError('Wrong time shape')
        if not np.array_equal(out.get('proprio_keypoints'), observation['keypoints']):
            raise ValueError('Server used different current keypoints')
        return out

    def close(self):
        self.sock.close()
