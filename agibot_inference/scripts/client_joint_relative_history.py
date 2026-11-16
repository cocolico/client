"""Native Joint-relative history handshake; reuse existing paired RPC and capture.

仅新增原生历史模型的握手配置；采集、坐标转换、执行和安全检查全部复用。
"""
import copy
import socket
from client_abcd_history import ABCDClient
from client_deployed78 import LAYOUT
from egoav78_contract import WIRE_INFO, validate_tasks


class JointRelativeHistoryClient(ABCDClient):
    """Keep the proven paired-history transport, pin the native78/columns model.

    复用五帧图像与关节历史传输；不要套用A/B/C/D的内部行旋转训练约定。
    """
    def __init__(self, host, port, *, model, task, tasks, embodiment,
                 expected_config, timeout=300, checkpoint=None, allow_simulated=False):
        registry=validate_tasks(tasks)
        if (model!='joint_relative_history' or task not in registry
                or registry[task]['embodiment']!=embodiment):
            raise ValueError('Unsupported native-history model/task/embodiment')
        self._expect(expected_config,dict(model=model),'Local history config')
        for key in ('deployment_id','checkpoint_sha256'):
            value=expected_config.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Pinned history identity required: '+key)
        self.task,self.kind=task,embodiment
        self.spec=copy.deepcopy(registry[task])
        self.wire_kind='3B' if embodiment=='3b' else 'g2'
        self.wire_task=self.wire_kind+'/'+('drink_head_only' if task=='drink' else task)
        self.history_capture=None;self.last_sample_ns=-1
        self.socket=socket.create_connection((host,port),timeout=timeout)
        try:
            self.info=self._rpc(dict(cmd='info'))
            self._expect(self.info,dict(protocol='egoav78_chunk_v1',protocol_version=4,
                representation='egoav78_rot6d_columns',model='joint',experiment='relative',
                variant='joint_relative_history',requested_model=model,
                model_action_representation='relative',model_action_mode='relative_to_current_observed_pose',
                wire_action_mode='absolute_in_requested_state_frame',action_mode='absolute',
                state_dim=78,action_dim=78,horizon=32,dt_s=.05,state_units='physical',action_units='physical',
                camera_keys=['head_rgb'],rotation='first_two_matrix_columns',xyz_unit='m',fingers_unit='rad',
                gripper_unit='mm',state_layout=LAYOUT,weights_strict=True,robot_io=False,
                checkpoint_step=30000,denoise_steps=10,inference_compile=True,warmup_complete=True,
                history_proprio=True,history_offsets=[-16,-12,-8,-4,0],
                history_transport='explicit_five_frame_or_measured_stream',
                deployment_id=expected_config['deployment_id'],
                checkpoint_sha256=expected_config['checkpoint_sha256']),'Native history server contract')
            if 'chunk_camera' not in self.info.get('state_frames',[]):
                raise ValueError('History server must accept chunk_camera')
            for key in ('checkpoint','normalization_sha256'):
                if not isinstance(self.info.get(key),str) or not self.info[key]:
                    raise ValueError('Missing history model identity: '+key)
            if checkpoint is not None and checkpoint!=self.info['checkpoint']:
                raise ValueError('History checkpoint mismatch')
            if self.info.get('simulated',False) and not allow_simulated:
                raise ValueError('Refusing simulated history model for execution')
            self._expect(self.info.get('tasks',{}).get(self.wire_task,{}),dict(
                embodiment=self.wire_kind,prompt=self.spec['instruction'],
                mask=[bool(x) for x in self.spec['validity']],
                source_frame='piper_station' if embodiment=='3b' else 'base_link'),'History selected task')
            self.execution_info=dict(self.info,**WIRE_INFO)
            self.reset()
        except BaseException:
            self.close();raise
