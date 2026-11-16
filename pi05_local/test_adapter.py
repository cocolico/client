"""No GPU/robot tests for ordering, wire identity, timestamps and rejection paths."""
import tempfile
import threading
import unittest
import sys
from pathlib import Path
import numpy as np
from runtime import MODEL_NAME,TIMES,state_right_left,actions_left_right
from server import Adapter,Server,Handler

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'aubo_wuji'))
from network import Client


class FakeRuntime:
    deployment_id='test-pi05'
    def describe(self):
        return dict(weights='/test/pi05_posttrain_humanhand/8408/model.safetensors',
            deployment_id=self.deployment_id,contract='humanhand',history_offset_frames=0,
            keypoint_time_s=TIMES.tolist())
    def infer(self,observation,prompt,steps,seed):
        state=state_right_left(observation['keypoints'])
        actions=np.tile(state,(50,1));actions+=np.arange(1,51,dtype=np.float32)[:,None]*.001
        return actions_left_right(actions),dict(inference_seconds=.01,native_actions_right_left=actions)


class Tests(unittest.TestCase):
    def test_handedness_round_trip(self):
        keypoints=np.arange(126,dtype=np.float32).reshape(2,21,3)
        state=state_right_left(keypoints)
        np.testing.assert_array_equal(state[:63],keypoints[1].ravel())
        np.testing.assert_array_equal(actions_left_right(np.tile(state,(50,1)))[0],keypoints)

    def test_invalid_actions(self):
        with self.assertRaises(ValueError):actions_left_right(np.zeros((32,126)))
        with self.assertRaises(ValueError):actions_left_right(np.full((50,126),np.nan))

    def test_real_tcp_client_and_raw_fifty_steps(self):
        with tempfile.TemporaryDirectory() as tmp,Server(('127.0.0.1',0),Handler) as server:
            server.adapter=Adapter(FakeRuntime(),tmp)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            cfg=dict(model=MODEL_NAME,task='tidying',timeout_s=5,num_inference_steps=10,
                expected_deployment_id='test-pi05',expected_run_name='pi05_posttrain_humanhand')
            client=Client(*server.server_address,cfg)
            try:
                observation=dict(head_rgb=np.zeros((400,640,3),np.uint8),keypoints=np.zeros((2,21,3),np.float32))
                response=client.infer(observation,1)
                self.assertEqual(response['keypoints'].shape,(32,2,21,3))
                self.assertEqual(response['native_keypoints'].shape,(50,2,21,3))
                self.assertAlmostEqual(response['keypoint_time_s'][-1],32/30,places=6)
                self.assertEqual(client.spec['history_offset_frames'],0)
                np.testing.assert_allclose(response['keypoints'][0],.001)
                self.assertEqual(len(list(Path(tmp).glob('*.npz'))),1)
                with self.assertRaises(ValueError):client.infer(observation,1)
            finally:
                client.close();server.shutdown();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
