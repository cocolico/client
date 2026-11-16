"""Task-bound physical78 TCP client. No hardware imports or execution policy."""
import copy
import socket
import time
import uuid
import numpy as np
from _ours_protocol import send_message,recv_message
from egoav78_contract import WIRE_INFO,MODELS,validate_tasks,validate78,task_digest


class EgoAV78Client:
    def __init__(self,host,port,*,model,task,tasks,embodiment,timeout=300,
                 checkpoint=None,allow_simulated=False):
        tasks=validate_tasks(tasks)
        if model not in MODELS or task not in tasks or tasks[task]['embodiment']!=embodiment:
            raise ValueError('Unsupported model/task/embodiment combination')
        self.task=task;self.spec=copy.deepcopy(tasks[task]);self.kind=embodiment
        self.socket=socket.create_connection((host,port),timeout=timeout)
        self.session_id=None
        try:
            self.info=self._rpc({'cmd':'info'})
            for key,value in dict(WIRE_INFO,model=model,weights_strict=True).items():
                if self.info.get(key)!=value:raise ValueError(f'Server contract mismatch: {key}')
            if type(self.info.get('simulated')) is not bool:
                raise ValueError('Server must explicitly declare simulated status')
            if self.info['simulated'] and not allow_simulated:
                raise ValueError('Refusing simulated policy for live execution')
            if type(self.info.get('horizon')) is not int or not 1<=self.info['horizon']<=256:
                raise ValueError('Invalid policy horizon')
            for key in ('checkpoint','normalization_sha256','deployment_id'):
                if not isinstance(self.info.get(key),str) or not self.info[key]:
                    raise ValueError(f'Missing server identity: {key}')
            if checkpoint is not None and self.info['checkpoint']!=checkpoint:
                raise ValueError('Checkpoint mismatch')
            remote=validate_tasks(self.info.get('tasks'))
            if remote.get(task)!=self.spec:
                raise ValueError('Selected task instruction/mask/embodiment mismatch')
            self.reset()
        except BaseException:
            self.close();raise

    def _rpc(self,message):
        send_message(self.socket,message);result=recv_message(self.socket)
        if 'error' in result:raise RuntimeError(str(result['error']))
        return result

    def reset(self):
        rid=uuid.uuid4().hex
        result=self._rpc(dict(cmd='reset',request_id=rid,task=self.task,
            task_contract=task_digest(self.spec)))
        expected=dict(protocol_version=WIRE_INFO['protocol_version'],request_id=rid,
            deployment_id=self.info['deployment_id'],task=self.task,
            task_contract=task_digest(self.spec),ok=True)
        for key,value in expected.items():
            if result.get(key)!=value:raise ValueError(f'Reset mismatch: {key}')
        if not isinstance(result.get('session_id'),str) or not result['session_id']:
            raise ValueError('Missing session identity')
        self.session_id=result['session_id']

    def infer(self,state,head_rgb,*,task,embodiment,sample_ns=None):
        if task!=self.task or embodiment!=self.kind:
            raise ValueError('Session is bound to the selected task/embodiment')
        state=validate78(state,self.kind)
        if state.shape!=(78,):raise ValueError('Expected one state78')
        rgb=np.asarray(head_rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
            raise ValueError('Expected uint8 HWC RGB')
        stamp=time.monotonic_ns() if sample_ns is None else sample_ns
        if type(stamp) is not int or stamp<0:raise ValueError('Invalid observation timestamp')
        rid=uuid.uuid4().hex
        result=self._rpc(dict(cmd='infer',request_id=rid,session_id=self.session_id,
            sample_ns=stamp,task=task,embodiment=embodiment,
            instruction=self.spec['instruction'],task_contract=task_digest(self.spec),
            observation=dict(state=state,head_rgb=rgb,validity=np.asarray(self.spec['validity'],np.uint8))))
        expected=dict(protocol_version=WIRE_INFO['protocol_version'],request_id=rid,
            session_id=self.session_id,sample_ns=stamp,deployment_id=self.info['deployment_id'],
            task=task,task_contract=task_digest(self.spec),embodiment=embodiment,
            representation=WIRE_INFO['representation'],schema=WIRE_INFO['schema'],
            action_units='physical',dt_s=.05)
        for key,value in expected.items():
            if result.get(key)!=value:raise ValueError(f'Inference response mismatch: {key}')
        if result.get('action_mask')!=self.spec['validity']:
            raise ValueError('Response task mask mismatch')
        seconds=result.get('inference_seconds')
        if isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not np.isfinite(seconds) or seconds<0:
            raise ValueError('Invalid inference timing')
        actions=validate78(result.get('actions'),self.kind)
        if actions.shape!=(self.info['horizon'],78):raise ValueError('Invalid action horizon/shape')
        self.last_response=result
        return actions

    def close(self):self.socket.close()
