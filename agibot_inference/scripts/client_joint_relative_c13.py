"""Pinned C13/current-proprio v4 profile; reuse existing checked absolute-action RPCs.

History belongs to this persistent server connection. The model predicts relative
actions internally; the wire already returns absolute poses in chunk_camera.
"""
import copy
import socket

from client_deployed78 import Deployed78Client, LAYOUT
from egoav78_contract import WIRE_INFO, validate_tasks


class C13Client(Deployed78Client):
    def __init__(self, host, port, *, model, task, tasks, embodiment,
                 expected_deployment, expected_experiment, timeout=300,
                 checkpoint=None, allow_simulated=False):
        registry = validate_tasks(tasks)
        if (model != 'joint_relative_c13' or task not in registry
                or registry[task]['embodiment'] != embodiment):
            raise ValueError('Unsupported C13 model/task/embodiment')
        if expected_experiment != 'relative':
            raise ValueError('C13 current profile requires expected_experiment=relative')
        if not isinstance(expected_deployment, str) or not expected_deployment:
            raise ValueError('Pinned C13 deployment identity is required')
        self.task, self.kind = task, embodiment
        self.spec = copy.deepcopy(registry[task])
        self.wire_kind = '3B' if embodiment == '3b' else 'g2'
        self.wire_task = self.wire_kind + '/' + ('drink_head_only' if task == 'drink' else task)
        self.last_sample_ns = -1
        self.socket = socket.create_connection((host, port), timeout=timeout)
        try:
            self.info = self._rpc({'cmd': 'info'})
            expected = dict(
                protocol='egoav78_chunk_v1', protocol_version=4,
                representation='egoav78_rot6d_columns', model='joint',
                experiment=expected_experiment, variant='joint_relative_continuous13_current',
                deployment_id=expected_deployment, checkpoint_step=30000,
                state_dim=78, action_dim=78, horizon=32, dt_s=.05,
                state_units='physical', action_units='physical', action_mode='absolute',
                wire_action_mode='absolute_in_requested_state_frame',
                model_action_representation='relative', model_action_mode='relative_to_current_observed_pose',
                camera_keys=['head_rgb'], rotation='first_two_matrix_columns',
                xyz_unit='m', fingers_unit='rad', gripper_unit='mm', state_layout=LAYOUT,
                weights_strict=True, robot_io=False, history_owner='server_connection',
                history_offsets_s=[-.8, -.6, -.4, -.2, 0], history_proprio=False,
                denoise_steps=10, inference_compile=True, warmup_complete=True)
            self._expect(self.info, expected, 'C13 server contract')
            if 'chunk_camera' not in self.info.get('state_frames', []):
                raise ValueError('C13 server does not accept chunk_camera')
            for key in ('checkpoint', 'checkpoint_sha256', 'normalization_sha256'):
                if not isinstance(self.info.get(key), str) or not self.info[key]:
                    raise ValueError('Missing C13 identity: ' + key)
            if checkpoint is not None and checkpoint != self.info['checkpoint']:
                raise ValueError('C13 checkpoint mismatch')
            if self.info.get('simulated', False) and not allow_simulated:
                raise ValueError('Refusing simulated C13 policy for execution')
            self._expect(self.info.get('tasks', {}).get(self.wire_task, {}), dict(
                embodiment=self.wire_kind, prompt=self.spec['instruction'],
                mask=[bool(x) for x in self.spec['validity']],
                source_frame='piper_station' if embodiment == '3b' else 'base_link'), 'C13 selected task')
            # Internal execution contract only; do not alter the server metadata.
            self.execution_info = dict(self.info, **WIRE_INFO)
            self.reset()
        except BaseException:
            self.close()
            raise

    def capture_history(self, read_frame):
        if self.info.get('history_transport')!='timestamped_rgb_batch_v1':
            raise ValueError('C13 endpoint lacks continuous RGB transport')
        from c13_rgb_transport import RGBTransport
        capture=RGBTransport(read_frame)
        self.rgb_capture=capture
        return capture

    def _rpc(self, request):
        capture=getattr(self,'rgb_capture',None)
        if capture is not None and request.get('cmd')=='infer':
            request=dict(request,history_frames=capture.frames_before(request['sample_ns']))
        response=super()._rpc(request)
        if capture is not None and request.get('cmd')=='reset':capture.clear()
        if capture is not None and request.get('cmd')=='infer':
            if (capture.first_sample_ns is not None and request['sample_ns']-capture.first_sample_ns>=1000000000
                    and response.get('history',{}).get('valid_history_frames',0)!=4):
                raise RuntimeError('C13 continuous history coverage lost; action chunk discarded')
            import json
            print('C13_HISTORY '+json.dumps(dict(sample_ns=request['sample_ns'],**response.get('history',{}))),flush=True)
        return response
