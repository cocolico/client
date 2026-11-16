"""Current-only native posttrain metadata on the unchanged Deployed78 infer path.

只适配新原生模型的元数据与裁剪声明；复用原RPC、输出验证和真机执行接口。
"""
import copy,re,socket
from client_deployed78 import Deployed78Client,LAYOUT
from egoav78_contract import WIRE_INFO,validate_tasks

class NativePost78Client(Deployed78Client):
    def __init__(self,host,port,*,model,task,tasks,embodiment,expected_config,timeout=300,checkpoint=None,allow_simulated=False):
        registry=validate_tasks(tasks)
        if model not in ('pi05','alpha','groot','fastwam') or task not in registry or registry[task]['embodiment']!=embodiment:
            raise ValueError('Unsupported native78 posttrain model/task/embodiment')
        self._expect(expected_config,dict(model=model,wire_protocol='egoav78_native_post_v4',
            conditioning='current_image_current_proprio',proprio_history=False),'Native config')
        for key in ('deployment_id','checkpoint_sha256','normalization_sha256'):
            if not re.fullmatch('[0-9a-f]{64}',str(expected_config.get(key,''))):raise ValueError('Pinned SHA required: '+key)
        self.task,self.kind=task,embodiment;self.spec=copy.deepcopy(registry[task])
        self.wire_kind='3B' if embodiment=='3b' else 'g2'
        self.wire_task=self.wire_kind+'/'+('drink_head_only' if task=='drink' else task)
        self.last_sample_ns=-1
        self.socket=socket.create_connection((host,port),timeout=timeout)
        try:
            self.info=self._rpc(dict(cmd='info'))
            self._expect(self.info,dict(protocol='egoav78_chunk_v1',protocol_version=4,
                representation='egoav78_rot6d_columns',model=model,requested_model=model,experiment='aligned_camera_c0',
                model_action_representation='camera_c0',model_action_mode='absolute_in_optical_chunk_camera',
                wire_action_mode='absolute_in_requested_state_frame',action_mode='absolute',
                wire_pose_convention='native_eef_axes_head_end_anchor',model_pose_convention='aligned_eef_axes_optical_c0',
                state_dim=78,action_dim=78,model_state_dim=78,model_action_dim=78,horizon=32,dt_s=.05,
                state_units='physical',action_units='physical',rotation='first_two_matrix_columns',model_rotation='first_two_matrix_rows',
                xyz_unit='m',fingers_unit='rad',gripper_unit='mm',body_yaw_unit='rad',state_layout=LAYOUT,
                camera_keys=['head_rgb'],weights_strict=True,fresh_chunk_per_request=True,robot_io=False,robot_execution_accepted=False,
                conditioning='current_image_current_proprio',proprio_history=False,history_images=False,history_proprio=False,
                denoise_steps=expected_config['denoise_steps'],compile=expected_config['compile'],checkpoint_step=30000,
                checkpoint_sha256=expected_config['checkpoint_sha256'],deployment_id=expected_config['deployment_id'],
                normalization_sha256=expected_config['normalization_sha256']),'Native posttrain contract')
            self._expect(self.info.get('history',{}),dict(required=False,offsets=[0],current_index=0,
                proprio_history=False,observation_fields=['head_rgb','state']),'Current-only conditioning')
            if 'chunk_camera' not in self.info.get('state_frames',[]):raise ValueError('Missing chunk_camera wire frame')
            if checkpoint is not None and checkpoint!=self.info.get('checkpoint'):raise ValueError('Checkpoint mismatch')
            if self.info.get('simulated',False) and not allow_simulated:raise ValueError('Refusing simulated policy for execution')
            self._expect(self.info.get('tasks',{}).get(self.wire_task,{}),dict(embodiment=self.wire_kind,
                prompt=self.spec['instruction'],mask=[bool(x) for x in self.spec['validity']],
                source_frame='piper_station' if embodiment=='3b' else 'base_link'),'Selected task')
            from task_image_preprocess import configure
            configure(self,expected_config.get('client_image_preprocess'))
            self.execution_info=dict(self.info,**WIRE_INFO)
            self.reset()
        except BaseException:
            self.close();raise
