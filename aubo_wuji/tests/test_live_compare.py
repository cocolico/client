"""Monitor tests use synthetic feedback and never connect to a robot/camera."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen
import numpy as np
from common import load_config,replay_imports,read_json,ROOT
from live_compare import LiveCompare,bend_degrees,project
from tests.test_pipeline import setup_case


class LiveCompareTests(unittest.TestCase):
    def test_bend_angles_distinguish_extended_and_bent(self):
        points=np.zeros((2,21,3))
        for f in range(5):
            points[:,1+4*f:5+4*f,0]=np.arange(4)
        np.testing.assert_allclose(bend_degrees(points),0)
        points[:,3,0]=1.;points[:,3,1]=1.
        self.assertAlmostEqual(bend_degrees(points)[0,0],90.)

    def test_projection_marks_points_behind_camera(self):
        info=dict(K=np.eye(3).tolist(),D=[0]*5)
        xy=project(np.array([[1.,2.,1.],[0.,0.,-1.]]),info)
        np.testing.assert_allclose(xy[0],[1,2]);self.assertTrue(np.isnan(xy[1]).all())

    def test_http_feedback_and_video_without_hardware(self):
        cfg,models,frame=setup_case();replay_imports(cfg)
        import device_backends,dh_kinematics
        cfg['monitor']=dict(port=0)
        hands={s:dict(positions_rad=frame['initial'][s]['q'],received_monotonic=time.monotonic()) for s in models}
        fake_reader=lambda config:SimpleNamespace(
            state=SimpleNamespace(getJointPositions=lambda:[0.]*6),
            config=SimpleNamespace(getKinematicsParam=lambda real:None),close=lambda:None)
        hardware=SimpleNamespace(base_T_camera={s:np.eye(4) for s in models},
            replay={'sides':{s:{'flange_T_wuji_base':np.eye(4).tolist()} for s in models}},
            hands=SimpleNamespace(snapshot=lambda:hands),geometry=None)
        camera=SimpleNamespace(condition=threading.Condition(),
            frames=[(0,time.monotonic_ns(),frame['head_rgb'])],
            info=read_json(ROOT/'vendor/cam0_color_camera_info.json'))
        pose=np.eye(4);pose[2,3]=.6
        with tempfile.TemporaryDirectory() as tmp,patch.object(device_backends,'Aubo',side_effect=fake_reader),patch.object(dh_kinematics,'fk',return_value=pose):
            directory=Path(tmp)/'cycle_0000';directory.mkdir()
            np.savez(directory/'prediction.npz',keypoints=np.tile(frame['keypoints'],(32,1,1,1)))
            np.savez(directory/'observation.npz',head_rgb=frame['head_rgb'],keypoints=frame['keypoints'],sample_ns=frame['sample_ns'])
            monitor=LiveCompare(cfg,hardware,camera,models,Path(tmp))
            try:
                target=np.r_[np.zeros(6),hands['left']['positions_rad'],np.zeros(6),hands['right']['positions_rad']]
                monitor.command(target,4,dict(trajectory=dict(path=str(directory/'runtime.npz'))))
                deadline=time.monotonic()+4.
                while 'actual_image' not in monitor.packet and time.monotonic()<deadline:time.sleep(.02)
                with urlopen(monitor.url+'/state',timeout=2.) as response:data=json.load(response)
                self.assertIn('actual_image',data,msg=data)
                self.assertEqual(data['step'],4)
                np.testing.assert_allclose(data['joint_error_deg']['right'],0)
                with urlopen(monitor.url,timeout=2.) as response:self.assertIn('本地重定向'.encode(),response.read())
            finally:monitor.close()
            cap=__import__('cv2').VideoCapture(str(Path(tmp)/'monitor/comparison.avi'))
            ok,image=cap.read();cap.release()
            self.assertTrue(ok);self.assertEqual(image.shape,(400,1280,3))
            self.assertTrue((Path(tmp)/'monitor/feedback.jsonl').read_text().strip())


if __name__=='__main__':unittest.main()
