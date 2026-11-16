"""Offline real-posttrain wire tests; no model, camera or robot is initialized."""
import importlib.util
import json
from pathlib import Path
import socketserver
import sys
import threading
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
sys.path.insert(0,str(COMMON))
from _ours_protocol import send_message,recv_message
from paired_history import PairedHistory


def fixture(port):
    return json.loads((ROOT/f'tests/fixtures/info_{port}.json').read_text())


class Endpoint:
    def __init__(self,info,reply_edit=None):
        self.info=info;self.requests=[];outer=self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                while True:
                    try:req=recv_message(self.request)
                    except (OSError,ConnectionError):return
                    outer.requests.append(req)
                    if req['cmd']=='info':result=outer.info
                    elif req['cmd']=='reset':
                        result=dict(ok=True,session_id='offline',request_id=req['request_id'],protocol_version=4,
                                    deployment_id=outer.info['deployment_id'])
                    else:
                        obs=req['observation'];state=obs['state']
                        if outer.info.get('proprio_history') is False:
                            assert 'state_history' not in obs and 'proprio_history' not in obs
                        else:
                            assert obs['state_history'].shape==(5,78)
                            np.testing.assert_array_equal(obs['state_history'][-1],state)
                            assert np.ptp(obs['state_history'][:,0])>0
                        assert obs['head_rgb_history'].shape==(5,32,32,3)
                        np.testing.assert_array_equal(obs['head_rgb_history'][-1],obs['head_rgb'])
                        assert obs['history_offsets']==[-16,-12,-8,-4,0]
                        assert obs['history_is_pad']==[False]*5
                        assert len(np.unique(obs['head_rgb_history'][:,0,0,0]))==5
                        actions=np.tile(state,(32,1));actions[:,0]=.25
                        result=dict(protocol_version=4,representation='egoav78_rot6d_columns',
                            request_id=req['request_id'],session_id='offline',sample_ns=req['sample_ns'],
                            deployment_id=outer.info['deployment_id'],task=req['task'],embodiment=req['embodiment'],
                            action_units='physical',action_frame='chunk_camera',action_mode='absolute',
                            horizon=32,dt_s=.05,action_mask=outer.info['tasks'][req['task']]['mask'],
                            anchor_state=state,actions=actions,inference_seconds=.01,
                            observed_latent_shape=[1,48,2,20,20])
                        if reply_edit:reply_edit(result)
                    send_message(self.request,result)
        self.server=socketserver.ThreadingTCPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()


class ProtocolTests(unittest.TestCase):
    def client(self,e,port=8881,task='book',current_only=False):
        self.assertIsNotNone(importlib.util.find_spec('client_abcd_history'),'ABCD adapter missing')
        from client_abcd_history import ABCDClient
        original=fixture(port);model='model_'+chr(ord('a')+port-8881)
        tasks={name:dict(embodiment=spec['embodiment'].lower(),instruction=spec['prompt'],
                         validity=list(map(int,spec['mask'])))
               for name in ('basket','book','diguan','yifu','dengpao')
               for spec in [original['tasks'][original['task_aliases'][name]]]}
        if task=='drink':
            spec=fixture(8883)['tasks']['g2/drink_head_only']
            tasks[task]=dict(embodiment='g2',instruction=spec['prompt'],validity=list(map(int,spec['mask'])))
        config=dict(model=model,model_native_dim=original['model_state_dim'],
                    deployment_id=original['deployment_id'],checkpoint_sha256=original['checkpoint_sha256'],
                    expected_experiment='real_posttrain')
        if current_only:config.update(proprio_history=False,conditioning='image_history_current_proprio')
        return ABCDClient(*e.server.server_address,model=model,task=task,tasks=tasks,
                          embodiment='3b' if task in ('basket','book') else 'g2',expected_config=config,timeout=2)

    def history(self):
        capture=PairedHistory(lambda:None)
        base=time.monotonic_ns()-1_050_000_000
        for i in range(6):
            state=np.zeros(78,np.float32)
            for p in (0,9,18):state[p+3:p+9]=[1,0,0,0,1,0]
            state[0]=.3+i*.01
            capture.push(dict(sample_ns=base+i*200_000_000,source_id=i,state=state,
                head_rgb=np.full((32,32,3),i,np.uint8),T_source_camera=np.eye(4)))
        return capture

    def infer(self,c,capture):
        packet=capture.snapshot();c.attach_history(capture)
        return c.infer(packet['state'],packet['head_rgb'],task='book',embodiment='3b',sample_ns=packet['sample_ns'])

    def test_all_four_send_real_paired_history_and_keep_absolute_actions(self):
        for port in (8881,8882,8883,8884):
            e=Endpoint(fixture(port))
            try:
                c=self.client(e,port)
                try:
                    result=self.infer(c,self.history())
                    np.testing.assert_array_equal(result[:,0],np.full(32,.25,np.float32))
                    self.assertEqual([r['cmd'] for r in e.requests],['info','reset','infer'])
                finally:c.close()
            finally:e.close()

    def test_new_all_four_send_image_history_and_current_proprio_only(self):
        for port in (8881,8882,8883,8884):
            info=fixture(port)
            info.update(proprio_history=False,conditioning='image_history_current_proprio')
            info['history'].update(proprio_history=False,
                observation_fields=['head_rgb_history','history_is_pad'],
                state_frame='current state uses the current request camera anchor; no past proprio')
            e=Endpoint(info)
            try:
                c=self.client(e,port,current_only=True)
                try:
                    result=self.infer(c,self.history())
                    np.testing.assert_array_equal(result[:,0],np.full(32,.25,np.float32))
                    self.assertEqual(e.requests[-1]['observation']['state'].shape,(78,))
                    self.assertNotIn('state_history',e.requests[-1]['observation'])
                finally:c.close()
            finally:e.close()

    def test_new_profile_rejects_legacy_history_checkpoint_before_reset(self):
        e=Endpoint(fixture(8881))
        try:
            with self.assertRaises(ValueError):self.client(e,current_only=True)
            self.assertEqual([r['cmd'] for r in e.requests],['info'])
        finally:e.close()

    def test_wrong_identity_dimension_frame_or_experiment_fails_before_reset(self):
        for key,value in [('deployment_id','wrong'),('checkpoint_sha256','wrong'),('model_state_dim',78),
                          ('rotation','first_two_matrix_rows'),('experiment','relative')]:
            data=fixture(8881);data[key]=value;e=Endpoint(data)
            try:
                with self.assertRaises(ValueError):self.client(e)
                self.assertEqual([r['cmd'] for r in e.requests],['info'])
            finally:e.close()

    def test_missing_capture_never_sends_infer(self):
        e=Endpoint(fixture(8881))
        try:
            c=self.client(e)
            try:
                packet=self.history().snapshot()
                with self.assertRaises(ValueError):
                    c.infer(packet['state'],packet['head_rgb'],task='book',embodiment='3b',sample_ns=packet['sample_ns'])
                self.assertEqual([r['cmd'] for r in e.requests],['info','reset'])
            finally:c.close()
        finally:e.close()

    def test_server_must_confirm_two_observed_latents_and_original_anchor(self):
        for edit in (lambda r:r.update(observed_latent_shape=[1,48,1,20,20]),
                     lambda r:r.update(action_frame='source'),
                     lambda r:r.update(anchor_state=np.zeros(78,np.float32))):
            e=Endpoint(fixture(8881),reply_edit=edit)
            try:
                c=self.client(e)
                try:
                    with self.assertRaises(ValueError):self.infer(c,self.history())
                finally:c.close()
            finally:e.close()


if __name__=='__main__':unittest.main()
