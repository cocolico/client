"""Aligned78 metadata adapter; reuse history/RPC/execution / 只适配合同，复用原客户端。"""
import copy
import re
import socket
from client_abcd_history import ABCDClient
from client_deployed78 import LAYOUT
from egoav78_contract import WIRE_INFO,validate_tasks


class AlignedC0Client(ABCDClient):
    """Physical wire remains native axes/head anchor; server owns optical/Q conversion.

    继承原ABCD的历史采集、持久RPC与infer；不在客户端再旋转Q、再乘光学外参或缩放夹爪。
    """
    def __init__(self,host,port,*,model,task,tasks,embodiment,expected_config,
                 timeout=300,checkpoint=None,allow_simulated=False):
        registry=validate_tasks(tasks)
        if model not in ('c0_full','c0_lora','c0_lr') or task not in registry or registry[task]['embodiment']!=embodiment:
            raise ValueError('Unsupported aligned model/task/embodiment')
        self._expect(expected_config,dict(model=model,model_native_dim=78,expected_experiment='aligned_camera_c0',
            conditioning='image_history_current_proprio',proprio_history=False),'Local aligned config')
        for key in ('deployment_id','checkpoint_sha256','normalization_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',str(expected_config.get(key,''))):raise ValueError('Pinned SHA256 required: '+key)
        if type(expected_config.get('checkpoint_step')) is not int or expected_config['checkpoint_step']<=0:
            raise ValueError('Explicit positive checkpoint step required')
        self.task,self.kind=task,embodiment;self.spec=copy.deepcopy(registry[task])
        self.wire_kind='3B' if embodiment=='3b' else 'g2'
        self.wire_task=self.wire_kind+'/'+('drink_head_only' if task=='drink' else task)
        self.proprio_history=False;self.history_capture=None;self.last_sample_ns=-1
        self.socket=socket.create_connection((host,port),timeout=timeout)
        try:
            self.info=self._rpc(dict(cmd='info'))
            expected=dict(protocol='egoav78_chunk_v1',protocol_version=4,representation='egoav78_rot6d_columns',
                model='joint',requested_model=model,experiment='aligned_camera_c0',
                model_action_representation='camera_c0',model_action_mode='absolute_in_optical_chunk_camera',
                wire_action_mode='absolute_in_requested_state_frame',action_mode='absolute',
                wire_pose_convention='native_eef_axes_head_end_anchor',model_pose_convention='aligned_eef_axes_optical_c0',
                state_dim=78,action_dim=78,model_state_dim=78,model_action_dim=78,horizon=32,dt_s=.05,
                state_units='physical',action_units='physical',rotation='first_two_matrix_columns',model_rotation='first_two_matrix_rows',
                xyz_unit='m',fingers_unit='rad',gripper_unit='mm',body_yaw_unit='rad',state_layout=LAYOUT,
                camera_keys=['head_rgb'],weights_strict=True,fresh_chunk_per_request=True,robot_io=False,robot_execution_accepted=False,
                conditioning='image_history_current_proprio',proprio_history=False,
                denoise_steps=expected_config['denoise_steps'],compile=expected_config['compile'],
                checkpoint_step=expected_config['checkpoint_step'],checkpoint_sha256=expected_config['checkpoint_sha256'],
                deployment_id=expected_config['deployment_id'],normalization_sha256=expected_config['normalization_sha256'])
            self._expect(self.info,expected,'Aligned server contract')
            from task_image_preprocess import configure
            configure(self, expected_config.get("client_image_preprocess"))
            from task_training_frame import configure as configure_training_frame
            configure_training_frame(self, expected_config.get('task_training_frame'))
            self._expect(self.info.get('history',{}),dict(required=True,offsets=[-16,-12,-8,-4,0],action_hz=20,
                current_index=4,observed_latents=2,proprio_history=False,
                observation_fields=['head_rgb_history','history_is_pad'],
                state_frame='current state uses the current request camera anchor; no past proprio'),'Image history')
            if 'chunk_camera' not in self.info.get('state_frames',[]):raise ValueError('Missing chunk_camera wire frame')
            if not isinstance(self.info.get('checkpoint'),str) or not self.info['checkpoint']:raise ValueError('Missing checkpoint identity')
            if checkpoint is not None and checkpoint!=self.info['checkpoint']:raise ValueError('Checkpoint mismatch')
            if self.info.get('simulated',False) and not allow_simulated:raise ValueError('Refusing simulated policy for execution')
            self._expect(self.info.get('tasks',{}).get(self.wire_task,{}),dict(embodiment=self.wire_kind,
                prompt=self.spec['instruction'],mask=[bool(x) for x in self.spec['validity']],
                source_frame='piper_station' if embodiment=='3b' else 'base_link'),'Selected task')
            self.execution_info=dict(self.info,**WIRE_INFO)
            self.reset()
        except BaseException:
            self.close();raise
