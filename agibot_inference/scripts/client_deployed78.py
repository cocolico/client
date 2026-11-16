"""Explicit adapter for the reviewed EgoAV78 v4 deployments; no robot I/O.

Both relative-model and absolute-model servers return absolute poses expressed
in the observed chunk-camera frame. Never compose a second relative delta here.
"""
import copy
import socket
import time
import uuid
import numpy as np
from _ours_protocol import send_message, recv_message
from egoav78_contract import WIRE_INFO, validate78, validate_tasks

LAYOUT = dict(left_pose=[0,9], right_pose=[9,18], head_pose=[18,27], body=[27,32],
              left_fingers=[32,54], left_gripper=[54,55],
              right_fingers=[55,77], right_gripper=[77,78])


class Deployed78Client:
    def __init__(self, host, port, *, model, task, tasks, embodiment,
                 expected_deployment, timeout=300, checkpoint=None, allow_simulated=False):
        registry = validate_tasks(tasks)
        if (model not in ('joint_relative','joint','tri','tri_relative','alpha','pi05','groot') or task not in registry
                or registry[task]['embodiment'] != embodiment):
            raise ValueError('Unsupported deployed78 model/task/embodiment')
        if not isinstance(expected_deployment,str) or not expected_deployment:
            raise ValueError('Reviewed deployment identity is required')
        self.task, self.kind = task, embodiment
        self.spec = copy.deepcopy(registry[task])
        self.wire_kind = '3B' if embodiment == '3b' else 'g2'
        self.wire_task = self.wire_kind + '/' + ('drink_head_only' if task == 'drink' else task)
        self.socket = socket.create_connection((host,port),timeout=timeout)
        self.last_sample_ns = -1
        try:
            self.info = self._rpc({'cmd':'info'})
            relative = model in ('joint_relative', 'tri_relative')
            expected = dict(protocol_version=4,representation='egoav78_rot6d_columns',
                model=model.removesuffix('_relative') if relative else model, state_dim=78,action_dim=78,
                horizon=32,dt_s=.05,state_units='physical',action_units='physical',
                action_mode='absolute',wire_action_mode='absolute_in_requested_state_frame',
                model_action_mode='relative_to_current_observed_pose' if relative else 'absolute_in_chunk_camera',
                camera_keys=['head_rgb'],rotation='first_two_matrix_columns',xyz_unit='m',
                fingers_unit='rad',gripper_unit='mm',body_yaw_unit='rad',state_layout=LAYOUT,
                weights_strict=True,fresh_chunk_per_request=True,robot_io=False,
                deployment_id=expected_deployment)
            # Current-only GR00T retains the same native physical execution wire.
            if model=='groot':
                expected.update(experiment='aligned_camera_c0',model_action_mode='absolute_in_optical_chunk_camera',
                    model_pose_convention='aligned_eef_axes_optical_c0',wire_pose_convention='native_eef_axes_head_end_anchor',
                    conditioning='current_image_current_proprio',history_images=False,history_proprio=False)
            self._expect(self.info,expected,'Server contract')
            if self.info.get('protocol') not in ('egoav78_chunk_v1','egoav78_relative_chunk_v1'):
                raise ValueError('Unrecognized deployed78 protocol')
            if 'chunk_camera' not in self.info.get('state_frames',[]):
                raise ValueError('Server does not accept chunk_camera')
            for key in ('checkpoint','normalization_sha256'):
                if not isinstance(self.info.get(key),str) or not self.info[key]:
                    raise ValueError('Missing server identity: '+key)
            if checkpoint is not None and checkpoint != self.info['checkpoint']:
                raise ValueError('Checkpoint mismatch')
            if self.info.get('simulated',False) and not allow_simulated:
                raise ValueError('Refusing simulated policy for execution')
            remote = self.info.get('tasks',{}).get(self.wire_task,{})
            self._expect(remote,dict(embodiment=self.wire_kind,prompt=self.spec['instruction'],
                mask=[bool(x) for x in self.spec['validity']],
                source_frame='piper_station' if embodiment=='3b' else 'base_link'), 'Selected task')
            # Explicit internal adapter contract, NOT claimed remote metadata.
            # Geometry is untouched; only spelling/protocol differs from v6.
            self.execution_info = dict(self.info, **WIRE_INFO)
            self.reset()
        except BaseException:
            self.close(); raise

    @staticmethod
    def _expect(actual, expected, label):
        for key,value in expected.items():
            if actual.get(key) != value: raise ValueError(f'{label} mismatch: {key}')

    def _rpc(self, request):
        from task_image_preprocess import prepare_request
        request = prepare_request(request, getattr(self, "image_preprocess", None))
        if getattr(self, 'lossless_png', False):
            from rpc_png_transport import send_message as send_lossless
            send_lossless(self.socket, request)
        else:
            send_message(self.socket,request)
        response = recv_message(self.socket)
        if not isinstance(response,dict): raise ValueError('Expected response object')
        if 'error' in response: raise RuntimeError(str(response['error']))
        return response

    def reset(self):
        rid = uuid.uuid4().hex
        result = self._rpc(dict(cmd='reset',request_id=rid))
        self._expect(result,dict(ok=True,request_id=rid,protocol_version=4,
            deployment_id=self.info['deployment_id']),'Reset')
        if not isinstance(result.get('session_id'),str) or not result['session_id']:
            raise ValueError('Missing session identity')
        self.session_id = result['session_id']; self.last_sample_ns = -1

    def infer(self, state, head_rgb, *, task, embodiment, sample_ns=None):
        if task != self.task or embodiment != self.kind:
            raise ValueError('Session bound to selected task/embodiment')
        state = validate78(state,self.kind,task=self.task).copy()
        if state.shape != (78,): raise ValueError('Expected one state78')
        if not np.allclose(state[18:27],[0,0,0,1,0,0,0,1,0],atol=1e-5,rtol=0):
            raise ValueError('chunk_camera state requires identity observed head pose')
        rgb = np.asarray(head_rgb)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or not 32 <= min(rgb.shape[:2]) <= max(rgb.shape[:2]) <= 4096):
            raise ValueError('Expected raw uint8 RGB HWC, 32..4096 pixels')
        stamp = time.monotonic_ns() if sample_ns is None else sample_ns
        if type(stamp) is not int or stamp < 0 or stamp <= self.last_sample_ns:
            raise ValueError('Observation timestamp must strictly increase')
        rid = uuid.uuid4().hex
        # Task selects server-side cached training prompt and dimension mask.
        # Unlike v6, v4 requires EXACTLY state and head_rgb inside observation.
        result = self._rpc(dict(cmd='infer',request_id=rid,session_id=self.session_id,
            sample_ns=stamp,task=self.wire_task,embodiment=self.wire_kind,
            state_frame='chunk_camera',seed=42,observation=dict(state=state,head_rgb=rgb)))
        self._expect(result,dict(protocol_version=4,representation='egoav78_rot6d_columns',
            request_id=rid,session_id=self.session_id,sample_ns=stamp,
            deployment_id=self.info['deployment_id'],task=self.wire_task,embodiment=self.wire_kind,
            action_frame='chunk_camera',action_mode='absolute',action_units='physical',
            action_mask=[bool(x) for x in self.spec['validity']],horizon=32,dt_s=.05), 'Inference response')
        if not np.array_equal(result.get('anchor_state'),state):
            raise ValueError('Response observation anchor mismatch')
        seconds = result.get('inference_seconds')
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or seconds<0:
            raise ValueError('Invalid inference timing')
        actions = validate78(result.get('actions'),self.kind,task=self.task)
        if actions.shape != (32,78): raise ValueError('Invalid action horizon/shape')
        self.last_response = result; self.last_sample_ns = stamp
        return actions

    def close(self): self.socket.close()


def open_client78(host, port, *, remote_config=None, **kwargs):
    """Explicit config selection; legacy v6 remains unchanged and the default."""
    import json
    from pathlib import Path
    config = json.loads(Path(remote_config).read_text()) if remote_config else {}
    from rpc_png_transport import transport_config
    transport = transport_config(config)
    if transport is not None:
        if transport.get('codec') != 'rgb_png_lossless_v1' or config.get('wire_protocol') not in (
                'egoav78_v4', 'egoav78_c13_v4', 'egoav78_abcd_history_v4', 'egoav78_paired_history_v4',
                'egoav78_aligned_c0_v4', 'egoav78_native_post_v4'):
            raise ValueError('Unsupported lossless transport/dialect')
    def finish(client):
        if transport is not None:
            # Raw info/reset negotiates transport separately from model metadata.
            # Direct TCP users keep raw framing unless the CPU gateway confirms it.
            client.lossless_png = (getattr(client, 'info', {}).get('transport_timing', {}).get('codec')
                                   == transport['codec'])
        return client
    dialect = config.get('wire_protocol','native78_v6')
    if dialect == 'native78_v6':
        from client_egoav78 import EgoAV78Client
        return finish(EgoAV78Client(host,port,**kwargs))
    if dialect not in ('egoav78_v4','egoav78_c13_v4','egoav78_abcd_history_v4','egoav78_paired_history_v4','egoav78_aligned_c0_v4','egoav78_native_post_v4'): raise ValueError('Unknown wire_protocol')
    if config.get('model') != kwargs['model']: raise ValueError('Endpoint/model mismatch')
    from egoav78_contract import load_tasks
    selected = load_tasks(Path(remote_config).parent / config['tasks_file'])
    # Task wording may differ from old training, but layout cannot silently change.
    task = kwargs['task']; old = kwargs['tasks'][task]
    if any(selected[task][key] != old[key] for key in ('embodiment','validity')):
        raise ValueError('Deployment task changes existing embodiment/mask')
    kwargs['tasks'] = selected
    if dialect == 'egoav78_native_post_v4':
        from client_native_post78 import NativePost78Client
        return finish(NativePost78Client(host,port,expected_config=config,**kwargs))
    if dialect == 'egoav78_aligned_c0_v4':
        from client_aligned_c0 import AlignedC0Client
        return finish(AlignedC0Client(host,port,expected_config=config,**kwargs))
    if dialect == 'egoav78_paired_history_v4':
        from client_joint_relative_history import JointRelativeHistoryClient
        return finish(JointRelativeHistoryClient(host,port,expected_config=config,**kwargs))
    if dialect == 'egoav78_abcd_history_v4':
        from client_abcd_history import ABCDClient
        return finish(ABCDClient(host,port,expected_config=config,**kwargs))
    if dialect == 'egoav78_c13_v4':
        from client_joint_relative_c13 import C13Client
        return finish(C13Client(host,port,expected_deployment=config['deployment_id'],
            expected_experiment=config['expected_experiment'],**kwargs))
    return finish(Deployed78Client(host,port,expected_deployment=config['deployment_id'],**kwargs))
