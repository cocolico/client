"""CPU-only tests of the actual shared RPC/history path; no robot imports."""
import sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
sys.path.insert(0,str(COMMON))
from task_image_preprocess import CONTRACT,prepare_request,configure

class CropTests(unittest.TestCase):
    def request(self,task='g2/diguan'):
        frame=np.arange(480*640*3,dtype=np.uint32).reshape(480,640,3).astype(np.uint8)
        history=np.stack([frame+i for i in range(5)])
        return dict(cmd='infer',task=task,sample_ns=123,observation=dict(state=np.zeros(78,np.float32),head_rgb=history[-1],head_rgb_history=history))
    def test_exact_pixels_all_history_no_mutation(self):
        q=self.request();r=prepare_request(q,CONTRACT)
        np.testing.assert_array_equal(r['observation']['head_rgb'],q['observation']['head_rgb'][130:450,218:538])
        np.testing.assert_array_equal(r['observation']['head_rgb_history'],q['observation']['head_rgb_history'][:,130:450,218:538])
        np.testing.assert_array_equal(r['observation']['head_rgb'],r['observation']['head_rgb_history'][-1])
        self.assertEqual(q['observation']['head_rgb'].shape,(480,640,3))
        self.assertIs(r['observation']['state'],q['observation']['state'])
        self.assertEqual(r['sample_ns'],123)
    def test_legacy_and_all_other_tasks_unchanged(self):
        q=self.request();self.assertIs(prepare_request(q,None),q)
        for task in ['g2/yifu','g2/dengpao','g2/drink_head_only','3B/basket','3B/book']:
            q=self.request(task);self.assertIs(prepare_request(q,CONTRACT),q)
    def test_double_crop_wrong_resolution_and_dtype_rejected(self):
        for shape,dtype in [((320,320,3),np.uint8),((720,1280,3),np.uint8),((480,640,3),np.float32)]:
            q=self.request();q['observation']['head_rgb']=np.zeros(shape,dtype)
            with self.assertRaises(ValueError):prepare_request(q,CONTRACT)
    def test_handshake_requires_exact_crop_contract(self):
        class Client:pass
        c=Client();c.info={}
        with self.assertRaises(ValueError):configure(c,CONTRACT)
        c.info={'client_image_preprocess':CONTRACT};configure(c,CONTRACT)
        self.assertEqual(c.image_preprocess,CONTRACT)
    def test_actual_history_rpc_crops_after_pair_validation(self):
        from client_abcd_history import ABCDClient
        q=self.request();obs=q['observation']
        history=dict(state=obs['state'],head_rgb=obs['head_rgb'],head_rgb_history=obs['head_rgb_history'],
            history_is_pad=np.zeros(5,bool),history_offsets=[-16,-12,-8,-4,0],
            valid_history_frames=4,selected_sample_ns=[1,2,3,4,5],timestamp_errors_s=[0]*5,padding=[False]*5)
        class Capture:
            def prepare(self,stamp):return history
        c=ABCDClient.__new__(ABCDClient);c.history_capture=Capture();c.proprio_history=False;c.socket=object();c.image_preprocess=CONTRACT
        sent=[]
        with patch('client_deployed78.send_message',side_effect=lambda sock,r:sent.append(r)),patch('client_deployed78.recv_message',return_value={'observed_latent_shape':[1,48,2,20,20]}):c._rpc(q)
        self.assertEqual(sent[0]['observation']['head_rgb_history'].shape,(5,320,320,3))
        np.testing.assert_array_equal(sent[0]['observation']['head_rgb_history'][:,0,0],obs['head_rgb_history'][:,130,218])
        self.assertNotIn('state_history',sent[0]['observation'])
        self.assertEqual(history['head_rgb'].shape,(480,640,3))

if __name__=='__main__':unittest.main()
