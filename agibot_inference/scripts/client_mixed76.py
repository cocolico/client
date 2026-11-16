"""Existing synchronous TCP client: absolute76 v4 or explicit camera71 v5."""
import socket
import time
import uuid
import numpy as np
from _ours_protocol import send_message, recv_message
from mixed76_contract import REPRESENTATION, embodiment_mask, validate_actions

CAMERA71_INFO = dict(protocol_version=5, representation='g2_eef71_chunk_first_camera_rot6d',
    action_dim=71, state_dim=71, action_mode='chunk_relative',
    coordinate_frame='chunk_first_camera_fixed', rotation_order='first_two_columns',
    head_frame='head_link3', position_units='meter', joint_units='radian')


class Mixed76Client:
    def __init__(self, host, port, *, model, timeout=300, checkpoint=None, representation='mixed76'):
        if representation not in ('mixed76', 'g2-camera71'):
            raise ValueError('Unknown explicitly selected representation')
        self.camera71 = representation == 'g2-camera71'
        self.dimension = 71 if self.camera71 else 76
        self.protocol_version = 5 if self.camera71 else 4
        self.representation = CAMERA71_INFO['representation'] if self.camera71 else REPRESENTATION
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.session_id = None
        try:
            self.info = self._rpc({'cmd': 'info'})
            expected = dict(protocol_version=4, model=model, representation=REPRESENTATION,
                            action_dim=76, state_dim=76, action_units='physical',
                            state_units='physical', action_mode='absolute', horizon=32,
                            dt_s=.05, camera_keys=['head_rgb'], weights_strict=True)
            if self.camera71:
                expected.update(CAMERA71_INFO)
            for key, value in expected.items():
                if self.info.get(key) != value:
                    raise ValueError(f'Server contract mismatch: {key}')
            for key in ('checkpoint', 'normalization_sha256', 'deployment_id'):
                if not isinstance(self.info.get(key), str) or not self.info[key]:
                    raise ValueError(f'Missing server identity: {key}')
            if checkpoint is not None and self.info['checkpoint'] != checkpoint:
                raise ValueError('Checkpoint mismatch')
            self.reset()
        except BaseException:
            self.close()
            raise

    def _rpc(self, message):
        send_message(self.socket, message)
        result = recv_message(self.socket)
        if 'error' in result:
            raise RuntimeError(str(result['error']))
        return result

    def close(self):
        self.socket.close()

    def reset(self):
        request_id = uuid.uuid4().hex
        result = self._rpc({'cmd': 'reset', 'request_id': request_id})
        for key, value in dict(protocol_version=self.protocol_version, request_id=request_id,
                               deployment_id=self.info['deployment_id'], ok=True).items():
            if result.get(key) != value:
                raise ValueError(f'Reset mismatch: {key}')
        if not isinstance(result.get('session_id'), str) or not result['session_id']:
            raise ValueError('Missing session identity')
        self.session_id = result['session_id']

    def infer(self, state76, head_rgb, *, task, embodiment, sample_ns=None):
        state = np.asarray(state76, np.float32)
        if self.camera71 and embodiment != 'g2':
            raise ValueError('camera71 supports G2 only')
        mask = np.ones(71, dtype=bool) if self.camera71 else embodiment_mask(embodiment)
        rgb = np.asarray(head_rgb)
        if state.shape != (self.dimension,) or not np.isfinite(state).all() or np.any(state[~mask] != 0):
            raise ValueError(f'Expected finite physical{self.dimension} state with zero inactive slots')
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Expected uint8 HWC RGB')
        if task not in self.info.get('prompts', {}) or embodiment not in self.info.get('embodiments', []):
            raise ValueError('Unsupported task or embodiment')
        request_id = uuid.uuid4().hex
        stamp = time.monotonic_ns() if sample_ns is None else sample_ns
        result = self._rpc(dict(cmd='infer', request_id=request_id, session_id=self.session_id,
                                sample_ns=stamp, task=task, embodiment=embodiment,
                                observation=dict(state=state, head_rgb=rgb)))
        expected = dict(protocol_version=self.protocol_version, request_id=request_id, session_id=self.session_id,
                        sample_ns=stamp, deployment_id=self.info['deployment_id'],
                        embodiment=embodiment, representation=self.representation,
                        action_units='physical', dt_s=.05)
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f'Inference response mismatch: {key}')
        if result.get('action_mask') != mask.tolist():
            raise ValueError('Response mask mismatch')
        seconds = result.get('inference_seconds')
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid inference timing')
        if self.camera71:
            actions = np.asarray(result['actions'], np.float32)
            if actions.shape != (self.info['horizon'],71) or not np.isfinite(actions).all():
                raise ValueError('Expected finite physical [horizon,71] actions')
        else:
            actions = validate_actions(result['actions'], embodiment, self.info['horizon'])
        self.last_response = result
        return actions
