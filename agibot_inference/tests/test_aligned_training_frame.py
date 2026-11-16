"""Offline persistent RPC with real client implementation / 原客户端离线持久连接验收。"""
import importlib
import socket
import threading
import time
import os
import tempfile
import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
sys.path.insert(0,str(COMMON))
import numpy as np
from _ours_protocol import recv_message,send_message
from egoav78_contract import load_tasks
from paired_history import PairedHistory


def test_new_client_exists():
    try:importlib.import_module('client_aligned_c0')
    except ModuleNotFoundError:raise AssertionError('aligned C0 client adapter not implemented')


def check_rpc(tmp_path,model,kind,task,local_contract,remote_contract):
    import json
    from client_deployed78 import open_client78,LAYOUT
    from egoav78_contract import embodiment_mask
    mask=embodiment_mask(kind).astype(bool)
    if task=='drink':mask[27]=True
    mask=mask.tolist();wire_task=('3B' if kind=='3b' else 'g2')+'/'+('drink_head_only' if task=='drink' else task)
    tasks={task:dict(embodiment=kind,instruction='trained English instruction',validity=[int(x) for x in mask])}
    (tmp_path/'tasks.json').write_text(json.dumps(dict(schema='egoav_78d_20261021_v1',tasks=tasks)))
    config=dict(wire_protocol='egoav78_aligned_c0_v4',model=model,tasks_file='tasks.json',expected_experiment='aligned_camera_c0',
        model_native_dim=78,deployment_id='a'*64,checkpoint_sha256='b'*64,checkpoint_step=50000 if model=='c0_full' else 30000,
        normalization_sha256='c'*64,conditioning='image_history_current_proprio',proprio_history=False,denoise_steps=10,compile=True)
    if local_contract is not None:config['task_training_frame']=local_contract
    (tmp_path/'config.json').write_text(json.dumps(config))
    info=dict(protocol='egoav78_chunk_v1',protocol_version=4,representation='egoav78_rot6d_columns',model='joint',requested_model=model,
        experiment='aligned_camera_c0',model_action_representation='camera_c0',model_action_mode='absolute_in_optical_chunk_camera',
        wire_action_mode='absolute_in_requested_state_frame',action_mode='absolute',wire_pose_convention='native_eef_axes_head_end_anchor',
        model_pose_convention='aligned_eef_axes_optical_c0',state_dim=78,action_dim=78,model_state_dim=78,model_action_dim=78,
        horizon=32,dt_s=.05,state_units='physical',action_units='physical',rotation='first_two_matrix_columns',
        model_rotation='first_two_matrix_rows',state_frames=['chunk_camera'],xyz_unit='m',fingers_unit='rad',gripper_unit='mm',body_yaw_unit='rad',
        state_layout=LAYOUT,camera_keys=['head_rgb'],weights_strict=True,fresh_chunk_per_request=True,robot_io=False,robot_execution_accepted=False,
        checkpoint='/test/weights',checkpoint_step=config['checkpoint_step'],checkpoint_sha256='b'*64,deployment_id='a'*64,
        normalization_sha256='c'*64,conditioning='image_history_current_proprio',proprio_history=False,denoise_steps=10,compile=True,
        history=dict(required=True,offsets=[-16,-12,-8,-4,0],action_hz=20,current_index=4,observed_latents=2,
            proprio_history=False,observation_fields=['head_rgb_history','history_is_pad'],state_frame='current state uses the current request camera anchor; no past proprio'),
        tasks={wire_task:dict(embodiment='3B' if kind=='3b' else 'g2',prompt=tasks[task]['instruction'],mask=mask,
            source_frame='piper_station' if kind=='3b' else 'base_link')},simulated=True)
    if remote_contract is not None:info['task_training_frame']=remote_contract
    received=[];errors=[];listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
    def serve():
        try:
            with listener.accept()[0] as conn:
                while True:
                    try:q=recv_message(conn)
                    except (OSError,EOFError,ConnectionError):break
                    received.append(q)
                    if q['cmd']=='info':r=info
                    elif q['cmd']=='reset':r=dict(ok=True,request_id=q['request_id'],session_id='one-session',protocol_version=4,deployment_id='a'*64)
                    else:
                        obs=q['observation'];assert 'state_history' not in obs
                        assert obs['head_rgb_history'].shape==(5,40,40,3)
                        assert obs['history_is_pad']==[True,True,True,True,False]
                        r=dict(protocol_version=4,representation='egoav78_rot6d_columns',request_id=q['request_id'],
                            session_id=q['session_id'],sample_ns=q['sample_ns'],deployment_id='a'*64,task=wire_task,
                            embodiment=q['embodiment'],action_frame='chunk_camera',action_mode='absolute',action_units='physical',
                            action_mask=mask,horizon=32,dt_s=.05,anchor_state=obs['state'],inference_seconds=.01,
                            observed_latent_shape=[1,48,2,20,20],actions=np.repeat(obs['state'][None],32,axis=0))
                    send_message(conn,r)
        except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=serve,daemon=True);worker.start();client=None
    try:
        client=open_client78(*listener.getsockname(),remote_config=tmp_path/'config.json',model=model,task=task,
            tasks=tasks,embodiment=kind,allow_simulated=True)
        h=PairedHistory(lambda:None);client.attach_history(h)
        for i in range(2):
            s=np.zeros(78,np.float32)
            for p in (0,9,18):s[p+3:p+9]=[1,0,0,0,1,0]
            s[0]=.2+i*.01;rgb=np.zeros((40,40,3),np.uint8);stamp=time.monotonic_ns()
            h.push(dict(state=s,head_rgb=rgb,sample_ns=stamp,source_id=i,T_source_camera=np.eye(4)))
            a=client.infer(s,rgb,task=task,embodiment=kind,sample_ns=stamp)
            np.testing.assert_array_equal(a,np.repeat(s[None],32,axis=0))
        assert [r['cmd'] for r in received]==['info','reset','infer','infer']
    finally:
        if client:client.close()
        listener.close();worker.join(timeout=3)
    assert not errors,errors


from test_task_training_frame import CONTRACT
class AlignedTrainingFrameTests(unittest.TestCase):
    def test_final_contract_all_tasks_actual_handshake(self):
        for kind,task in [('g2','diguan'),('g2','yifu'),('g2','dengpao'),('g2','drink'),('3b','basket'),('3b','book')]:
            if os.environ.get('EGOAV_TEST_EMBODIMENT') and kind!=os.environ['EGOAV_TEST_EMBODIMENT']:continue
            with self.subTest(kind=kind,task=task),tempfile.TemporaryDirectory() as tmp:
                check_rpc(Path(tmp),'c0_lr',kind,task,CONTRACT,CONTRACT)

    def test_actual_handshake_rejects_missing_or_unpinned_metadata_for_g2_and_piper(self):
        import copy
        bad=copy.deepcopy(CONTRACT);bad['default']='changed'
        for kind,task in [('g2','diguan'),('3b','basket')]:
            for local,remote in [(CONTRACT,None),(None,CONTRACT),(bad,bad),(CONTRACT,bad)]:
                with self.subTest(kind=kind,local=local,remote=remote),tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(ValueError,'contract'):
                        check_rpc(Path(tmp),'c0_lr',kind,task,local,remote)

if __name__=='__main__':unittest.main()
