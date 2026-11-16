"""CPU-only paired sample training-frame contract and persistent RPC regressions."""
import copy
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
if not COMMON.exists(): COMMON=Path(__file__).parent/'work'
sys.path.insert(0,str(COMMON))
from paired_history import PairedHistory
from client_abcd_history import ABCDClient
from _ours_protocol import send_message,recv_message

CONTRACT=dict(schema='task_training_frame_v1',default='optical_chunk_camera',
    tasks={'g2/diguan':'optical_base_legacy_labels'},requires_source_from_chunk_camera_tasks=['g2/diguan'])

def state():
    s=np.zeros(78,np.float32)
    for start in (0,9,18):s[start+3:start+9]=[1,0,0,0,1,0]
    return s

def packet(i,stamp):
    anchor=np.eye(4,dtype=np.float64);anchor[:3,3]=[i*.1,i*.2,-i*.05]
    angle=i*.15;c,s=np.cos(angle),np.sin(angle)
    anchor[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    return dict(state=state(),head_rgb=np.full((40,40,3),i,np.uint8),
        sample_ns=stamp,source_id=i,T_source_camera=anchor)

def helper():
    try:from task_training_frame import configure,prepare_request
    except ImportError:raise AssertionError('task_training_frame helper is not implemented')
    return configure,prepare_request

class TrainingFrameTests(unittest.TestCase):
    def test_prepare_anchor_is_exact_sample_and_readonly(self):
        h=PairedHistory(lambda:None);now=time.monotonic_ns()-10_000_000
        old,new=packet(1,now),packet(2,now+1)
        h.push(old);h.push(new)
        prepared=h.prepare(now)
        self.assertIn('source_from_chunk_camera',prepared)
        a=prepared['source_from_chunk_camera']
        np.testing.assert_array_equal(a,old['T_source_camera'])
        self.assertEqual(a.dtype,np.float64)
        self.assertFalse(a.flags.writeable)
        with self.assertRaises(ValueError):a.flags.writeable=True
        np.testing.assert_array_equal(h.prepare(now+1)['source_from_chunk_camera'],new['T_source_camera'])

    def test_contract_is_explicit_exact_and_copied_for_every_embodiment(self):
        configure,_=helper()
        class Client:pass
        for kind in ('g2','3b'):
            c=Client();c.kind=kind;c.info={'task_training_frame':copy.deepcopy(CONTRACT)}
            configure(c,CONTRACT)
            self.assertEqual(c.task_training_frame,CONTRACT)
            self.assertIsNot(c.task_training_frame,CONTRACT)
            c.info={}
            with self.assertRaisesRegex(ValueError,'contract'):configure(c,CONTRACT)
            c.info={'task_training_frame':CONTRACT}
            with self.assertRaisesRegex(ValueError,'contract'):configure(c,None)
            bad=copy.deepcopy(CONTRACT);bad['default']='other'
            with self.assertRaisesRegex(ValueError,'contract'):configure(c,bad)
            c.info={};configure(c,None);self.assertIsNone(c.task_training_frame)

    def test_missing_and_malformed_anchor_rejected_before_network(self):
        configure,_=helper()
        for bad in (None,np.eye(3),np.full((4,4),np.nan),np.diag([1,1,-1,1]),np.diag([2,1,1,1]),np.zeros((4,4))):
            h=PairedHistory(lambda:None);p=packet(1,time.monotonic_ns());h.push(p)
            prepared=h.prepare(p['sample_ns'])
            if bad is None:prepared.pop('source_from_chunk_camera')
            else:prepared['source_from_chunk_camera']=bad
            client=ABCDClient.__new__(ABCDClient);client.info={'task_training_frame':CONTRACT};configure(client,CONTRACT)
            client.history_capture=h;client.proprio_history=False;client.socket=object()
            q=dict(cmd='infer',task='g2/diguan',sample_ns=p['sample_ns'],observation=dict(state=p['state'],head_rgb=p['head_rgb']))
            with patch.object(h,'prepare',return_value=prepared),patch('client_deployed78.send_message') as send:
                with self.assertRaises((ValueError,KeyError)):client._rpc(q)
                send.assert_not_called()

    def test_persistent_rpc_same_sample_not_newest_then_next_chunk(self):
        configure,_=helper()
        for final in (False,True):
            for task in ('g2/diguan','g2/yifu','g2/dengpao','g2/drink_head_only','3B/basket','3B/book'):
                with self.subTest(final=final,task=task):
                    contract=CONTRACT if final else None
                    client=ABCDClient.__new__(ABCDClient);client.info={'task_training_frame':contract} if final else {}
                    configure(client,contract);client.proprio_history=False
                    h=PairedHistory(lambda:None);client.attach_history(h)
                    now=time.monotonic_ns()-10_000_000
                    packets=[packet(i,now+i) for i in range(1,4)]
                    h.push(packets[0]);h.push(packets[1]);received=[];errors=[]
                    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
                    def serve():
                        try:
                            with listener.accept()[0] as conn:
                                for _ in range(2):
                                    q=recv_message(conn);received.append(q)
                                    send_message(conn,{'observed_latent_shape':[1,48,2,20,20]})
                        except BaseException as exc:errors.append(exc)
                    worker=threading.Thread(target=serve,daemon=True);worker.start()
                    client.socket=socket.create_connection(listener.getsockname())
                    try:
                        with patch.object(h,'snapshot',side_effect=AssertionError('must never read newest anchor')):
                            for p in (packets[0],packets[2]):
                                if p is packets[2]:h.push(p)
                                q=dict(cmd='infer',task=task,sample_ns=p['sample_ns'],observation=dict(state=p['state'],head_rgb=p['head_rgb']))
                                client._rpc(q)
                        worker.join(3);self.assertFalse(worker.is_alive());self.assertFalse(errors,errors)
                        self.assertEqual(len(received),2)
                        for q,p in zip(received,(packets[0],packets[2])):
                            if final and task=='g2/diguan':
                                np.testing.assert_array_equal(q['source_from_chunk_camera'],p['T_source_camera'])
                                self.assertEqual(q['source_from_chunk_camera'].dtype,np.float64)
                            else:self.assertNotIn('source_from_chunk_camera',q)
                            self.assertNotIn('source_from_chunk_camera',q['observation'])
                            np.testing.assert_array_equal(q['observation']['state'],p['state'])
                            np.testing.assert_array_equal(q['observation']['head_rgb'],p['head_rgb'])
                    finally:client.close();listener.close();worker.join(3)

    def test_final_crop_and_anchor_share_original_paired_sample(self):
        from task_image_preprocess import CONTRACT as CROP,configure as configure_crop
        configure,_=helper()
        p=packet(1,time.monotonic_ns());p['head_rgb']=np.arange(480*640*3,dtype=np.uint32).reshape(480,640,3).astype(np.uint8)
        h=PairedHistory(lambda:None);h.push(p)
        client=ABCDClient.__new__(ABCDClient);client.history_capture=h;client.proprio_history=False;client.socket=object()
        client.info={'task_training_frame':CONTRACT,'client_image_preprocess':CROP}
        configure(client,CONTRACT);configure_crop(client,CROP)
        q=dict(cmd='infer',task='g2/diguan',sample_ns=p['sample_ns'],observation=dict(state=p['state'],head_rgb=p['head_rgb']))
        sent=[]
        with patch('client_deployed78.send_message',side_effect=lambda sock,q:sent.append(q)),patch('client_deployed78.recv_message',return_value={'observed_latent_shape':[1,48,2,20,20]}):
            client._rpc(q)
        np.testing.assert_array_equal(sent[0]['source_from_chunk_camera'],p['T_source_camera'])
        np.testing.assert_array_equal(sent[0]['observation']['head_rgb'],p['head_rgb'][130:450,218:538])
        self.assertEqual(sent[0]['observation']['head_rgb_history'].shape,(5,320,320,3))
        self.assertEqual(h.prepare(p['sample_ns'])['head_rgb'].shape,(480,640,3))

    def test_paired_mismatch_rejected_before_anchor_processing(self):
        configure,_=helper()
        h=PairedHistory(lambda:None);p=packet(1,time.monotonic_ns());h.push(p)
        client=ABCDClient.__new__(ABCDClient);client.history_capture=h;client.info={'task_training_frame':CONTRACT}
        configure(client,CONTRACT)
        q=dict(cmd='infer',task='g2/diguan',sample_ns=p['sample_ns'],observation=dict(state=p['state']+1,head_rgb=p['head_rgb']))
        with self.assertRaisesRegex(ValueError,'exact paired'):client._rpc(q)

if __name__=='__main__':unittest.main()
