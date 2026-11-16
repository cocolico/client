"""Pinned A/B/C/D posttrain client with real paired RGB/proprio history.

Server wire actions already contain absolute poses in the request camera frame.
The server alone converts its native73 layout and rotation-row representation.
"""
import copy
import json
import socket

import numpy as np

from client_deployed78 import Deployed78Client, LAYOUT
from egoav78_contract import WIRE_INFO, validate_tasks
from paired_history import PairedHistory


class ABCDClient(Deployed78Client):
    # Legacy history adapters reuse this RPC without calling this constructor.
    # 旧历史客户端复用RPC但自行初始化，默认保留其历史proprio行为。
    proprio_history = True
    task_training_frame = None

    def __init__(self, host, port, *, model, task, tasks, embodiment,
                 expected_config, timeout=300, checkpoint=None, allow_simulated=False):
        registry = validate_tasks(tasks)
        native_dim = dict(model_a=73, model_b=73, model_c=78, model_d=78).get(model)
        if (native_dim is None or task not in registry
                or registry[task]['embodiment'] != embodiment):
            raise ValueError('Unsupported A/B/C/D model/task/embodiment')
        if task == 'drink' and native_dim == 73:
            raise ValueError('A/B do not support drink/body actions')
        self._expect(expected_config, dict(model=model, model_native_dim=native_dim,
            expected_experiment='real_posttrain'), 'Local A/B/C/D config')
        # Explicit per-deployment conditioning; legacy configs keep their old mode.
        # 每个部署明确条件类型，旧配置仍沿用历史本体输入。
        self.proprio_history = expected_config.get('proprio_history', True)
        if type(self.proprio_history) is not bool:
            raise ValueError('proprio_history must be an explicit boolean')
        if not self.proprio_history:
            self._expect(expected_config, dict(conditioning='image_history_current_proprio'), 'Conditioning config')
        for key in ('deployment_id', 'checkpoint_sha256'):
            value = expected_config.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Pinned deployment/checkpoint SHA256 required: '+key)
        self.task, self.kind = task, embodiment
        self.spec = copy.deepcopy(registry[task])
        self.wire_kind = '3B' if embodiment == '3b' else 'g2'
        self.wire_task = self.wire_kind + '/' + ('drink_head_only' if task == 'drink' else task)
        self.history_capture = None
        self.last_sample_ns = -1
        self.socket = socket.create_connection((host, port), timeout=timeout)
        try:
            self.info = self._rpc(dict(cmd='info'))
            self._expect(self.info, dict(protocol='egoav78_chunk_v1', protocol_version=4,
                representation='egoav78_rot6d_columns', model='joint', experiment='real_posttrain',
                model_action_representation='relative', model_action_mode='relative_to_current_observed_pose',
                wire_action_mode='absolute_in_requested_state_frame', action_mode='absolute',
                state_dim=78, action_dim=78, model_state_dim=native_dim, model_action_dim=native_dim,
                horizon=32, dt_s=.05, state_units='physical', action_units='physical',
                camera_keys=['head_rgb'], rotation='first_two_matrix_columns',
                model_rotation='first_two_matrix_rows', xyz_unit='m', fingers_unit='rad',
                gripper_unit='mm', body_yaw_unit='rad', state_layout=LAYOUT,
                weights_strict=True, fresh_chunk_per_request=True, robot_io=False,
                robot_execution_accepted=False, checkpoint_step=30000, denoise_steps=10,
                compile=True, deployment_id=expected_config['deployment_id'],
                checkpoint_sha256=expected_config['checkpoint_sha256']), 'A/B/C/D server contract')
            if not self.proprio_history:
                self._expect(self.info, dict(conditioning='image_history_current_proprio',
                    proprio_history=False), 'Current-proprio checkpoint')
            elif self.info.get('proprio_history', True) is not True:
                raise ValueError('Legacy config requires historical proprio')
            history_contract = dict(required=True,
                offsets=[-16,-12,-8,-4,0], action_hz=20,
                observation_fields=['state_history','head_rgb_history','history_is_pad'],
                state_frame='all five states share the current request state_frame and current camera anchor',
                current_index=4, observed_latents=2)
            if not self.proprio_history:
                history_contract.update(proprio_history=False,
                    observation_fields=['head_rgb_history','history_is_pad'],
                    state_frame='current state uses the current request camera anchor; no past proprio')
            self._expect(self.info.get('history', {}), history_contract, 'Paired-history contract')
            if 'chunk_camera' not in self.info.get('state_frames', []):
                raise ValueError('Server does not accept chunk_camera')
            for key in ('checkpoint','normalization_sha256'):
                if not isinstance(self.info.get(key), str) or not self.info[key]:
                    raise ValueError('Missing server identity: '+key)
            if checkpoint is not None and checkpoint != self.info['checkpoint']:
                raise ValueError('Checkpoint mismatch')
            if self.info.get('simulated', False) and not allow_simulated:
                raise ValueError('Refusing simulated policy for execution')
            self._expect(self.info.get('tasks', {}).get(self.wire_task, {}),
                dict(embodiment=self.wire_kind, prompt=self.spec['instruction'],
                     mask=[bool(x) for x in self.spec['validity']],
                     source_frame='piper_station' if embodiment == '3b' else 'base_link'), 'Selected task')
            # Internal execution adapter only; this does not overwrite remote metadata.
            self.execution_info = dict(self.info, **WIRE_INFO)
            self.reset()
        except BaseException:
            self.close()
            raise

    def attach_history(self, capture):
        if not isinstance(capture, PairedHistory):
            raise TypeError('A/B/C/D require a real PairedHistory capture')
        self.history_capture = capture

    def _rpc(self, request):
        if request.get('cmd') != 'infer':
            return super()._rpc(request)
        if self.history_capture is None:
            raise ValueError('Paired RGB/proprio capture is required; no single-frame fallback')
        history = self.history_capture.prepare(request['sample_ns'])
        observation = request['observation']
        if (not np.array_equal(observation['state'], history['state'])
                or not np.array_equal(observation['head_rgb'], history['head_rgb'])):
            raise ValueError('Current observation must be the exact paired capture snapshot')
        # Boolean ndarray is deliberately avoided: the existing wire codec only
        # supports uint8/float32/float64 arrays, while this field is a JSON list.
        observation = dict(observation,
            head_rgb_history=history['head_rgb_history'],
            history_is_pad=history['history_is_pad'].tolist(),
            history_offsets=history['history_offsets'])
        if self.proprio_history:
            observation['state_history'] = history['state_history']
        from task_training_frame import prepare_request
        request = prepare_request(dict(request, observation=observation),
                                  self.task_training_frame, history)
        response = super()._rpc(request)
        if response.get('observed_latent_shape') != [1,48,2,20,20]:
            raise ValueError('Server did not confirm two encoded observed history latents')
        self.last_history = {key: history[key] for key in
            ('valid_history_frames','selected_sample_ns','timestamp_errors_s','padding')}
        print('ABCD_HISTORY '+json.dumps(self.last_history), flush=True)
        return response
