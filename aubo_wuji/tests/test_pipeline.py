"""Hardware-free tests: real TCP framing, independent FK, IK and coordinates."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT, SIDES, load_config, read_json, transform, write_json
from hand_kinematics import HandModel, camera_points, normalized_goals, retarget_chunk
from network import Client, PROTOCOL, validate_observation
from planning import arrays_from_chunk, check_age, unchanged_start
from run import main, save_observation
from vendor.wire_v4 import send_message, recv_message


def setup_case():
    cfg = load_config()
    cfg['server'].update(model='hh_a12',expected_run_name='pt-hh-a12-g64-1114')
    cfg['server']['expected_deployment_id'] = None  # Mock identities are never production identities.
    models = {s:HandModel(cfg['hand_model_root'],s) for s in SIDES}
    initial = {}; points = []
    for i,side in enumerate(SIDES):
        pose = np.eye(4); pose[:3,:3] = Rotation.from_euler('xyz',[.1,-.2,.15*(-1 if i else 1)]).as_matrix()
        pose[:3,3] = [(-.2 if i==0 else .2),0,.6]
        q = np.clip(np.full(20,.12),models[side].lo+.01,models[side].hi-.01)
        initial[side] = dict(q=q,camera_T_root=pose,arm_q=np.zeros(6))
        points.append(camera_points(models[side],q,pose))
    obs = dict(head_rgb=np.zeros((400,640,3),np.uint8),history_rgb=np.ones((400,640,3),np.uint8),
               keypoints=np.asarray(points,np.float32),initial=initial,sample_ns=time.monotonic_ns())
    return cfg,models,obs


class FakeServer:
    """A real TCP server with explicit model identity; never imports a device SDK."""
    def __init__(self, mutation=None):
        self.listener = socket.socket(); self.listener.bind(('127.0.0.1',0)); self.listener.listen()
        self.host,self.port = self.listener.getsockname(); self.mutation = mutation
        self.times = np.arange(1,33,dtype=np.float32)/120
        self.spec = dict(weights='/checkpoints/pt-hh-a12-g64-1114/checkpoints/step_100.pt',
            contract='humanhand',history_offset_frames=4,keypoint_time_s=self.times.tolist(),deployment_id='test-pinned')
        self.messages = []; self.error = None
        self.thread = threading.Thread(target=self.loop,daemon=True); self.thread.start()

    def loop(self):
        try:
            peer,_ = self.listener.accept()
            with peer:
                while True:
                    try: req = recv_message(peer)
                    except ConnectionError: break
                    self.messages.append(req)
                    if req['cmd'] == 'info':
                        out = dict(protocol=PROTOCOL,protocol_version=4,horizon=32,default_model='hh_a12',
                                   models={'hh_a12':self.spec},tasks={'flower':{'instruction':'Arrange flowers'}})
                    elif req['cmd'] == 'reset': out = dict(request_id=req['request_id'],session_id='test-session')
                    else:
                        kp = np.tile(req['observation']['keypoints'],(32,1,1,1))
                        kp[...,0] += np.linspace(0,.001,32,dtype=np.float32)[:,None,None]
                        out = dict(protocol=PROTOCOL,protocol_version=4,horizon=32,
                            request_id=req['request_id'],session_id=req['session_id'],sample_ns=req['sample_ns'],
                            model='hh_a12',deployment_id='test-pinned',weights=self.spec['weights'],
                            contract='humanhand',task='flower',hand_order=['left','right'],
                            frame='cam0_color_optical_frame',units='metres',keypoints=kp,
                            keypoint_time_s=self.times,proprio_keypoints=req['observation']['keypoints'])
                        if self.mutation: self.mutation(out)
                    send_message(peer,out)
        except BaseException as error: self.error = error

    def close(self):
        self.listener.close(); self.thread.join(timeout=3)
        if self.thread.is_alive(): raise AssertionError('Mock server did not stop')
        if self.error: raise self.error


@contextmanager
def served(mutation=None):
    server = FakeServer(mutation)
    try: yield server
    finally: server.close()


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cfg,cls.models,cls.obs = setup_case()

    def test_fk_matches_independent_pinocchio(self):
        import pinocchio as pin
        rng = np.random.default_rng(92)
        for hand in self.models.values():
            model = pin.buildModelFromUrdf(str(hand.path)); data = model.createData()
            for _ in range(3):
                q = rng.uniform(hand.lo,hand.hi)
                reference = np.zeros(model.nq)
                for name,value in zip(hand.names,q): reference[model.joints[model.getJointId(name)].idx_q] = value
                pin.forwardKinematics(model,data,reference); pin.updateFramePlacements(model,data)
                points = np.array([data.oMf[model.getFrameId(n)].translation for n in hand.points])
                np.testing.assert_allclose(hand.fk(q),points,atol=1e-12)

    def test_analytic_joint_jacobian(self):
        for hand in self.models.values():
            q = (hand.lo+hand.hi)/2; _,jac = hand.fk(q,True)
            for i in range(20):
                delta = np.eye(20)[i]*1e-6
                finite = (hand.fk(q+delta)-hand.fk(q-delta))/2e-6
                np.testing.assert_allclose(jac[:,:,i],finite,atol=1e-9)

    def test_retarget_roundtrip_and_timeline(self):
        prediction = dict(keypoints=np.tile(self.obs['keypoints'],(32,1,1,1)),
                          keypoint_time_s=np.arange(1,33)/120)
        out = retarget_chunk(self.models,prediction,self.obs['initial'],self.cfg['execution'],3)
        np.testing.assert_allclose(out['time_s'],[0,1/120,2/120,3/120])
        for side in SIDES:
            np.testing.assert_allclose(out[side]['poses'][0],self.obs['initial'][side]['camera_T_root'])
            self.assertEqual(out[side]['joints'].shape,(4,20))
            self.assertLess(out[side]['fit_error_m'].max(),.004)

    def test_missing_bones_rejected(self):
        with self.assertRaises(ValueError): normalized_goals(np.zeros((2,21,3)),self.models['left'].fk(np.zeros(20)))

    def test_motion_envelope_rejected(self):
        points = np.tile(self.obs['keypoints'],(32,1,1,1)); points[...,0] += 1
        with self.assertRaisesRegex(ValueError,'translation'):
            retarget_chunk(self.models,dict(keypoints=points,keypoint_time_s=np.arange(1,33)/120),
                           self.obs['initial'],dict(self.cfg['execution'],max_root_translation_m=.08),1)

    def test_root_rotation_limit_is_warning_only(self):
        prediction = dict(keypoints=np.tile(self.obs['keypoints'],(32,1,1,1)),
                          keypoint_time_s=np.arange(1,33)/120)
        out = retarget_chunk(self.models,prediction,self.obs['initial'],
            dict(self.cfg['execution'],max_root_rotation_rad=1e-12),1)
        for side in SIDES:
            diagnostic = out[side]['diagnostics'][0]
            self.assertTrue(diagnostic['root_rotation_warning'])
            self.assertEqual(diagnostic['root_rotation_warning_threshold_rad'],1e-12)

    def test_transform_and_mount_roundtrip(self):
        from observation import calibration_chain
        from replay_pipeline import config
        c = config(self.cfg['replay_config']); base = calibration_chain(self.cfg,c)
        chunk = dict(time_s=np.array([0.,.1]))
        mounts = {s:transform(c['sides'][s]['flange_T_wuji_base']) for s in SIDES}
        for side in SIDES: chunk[side] = dict(poses=np.tile(self.obs['initial'][side]['camera_T_root'],(2,1,1)),joints=np.zeros((2,20)))
        arrays = arrays_from_chunk(chunk,base,mounts,self.models)
        for side in SIDES:
            back = np.linalg.inv(base[side]) @ arrays[side+'_aubo_base_T_flange'] @ mounts[side]
            np.testing.assert_allclose(back,chunk[side]['poses'],atol=1e-12)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cfg,cls.models,cls.obs = setup_case()

    def test_persistent_handshake_and_multiple_samples(self):
        with served() as server:
            client = Client(server.host,server.port,self.cfg['server'])
            try:
                for stamp in (10,11):
                    out = client.infer({k:self.obs[k] for k in ('head_rgb','history_rgb','keypoints')},stamp)
                    self.assertEqual(out['keypoints'].shape,(32,2,21,3))
                with self.assertRaises(ValueError): client.infer(self.obs,11)
            finally: client.close()
            self.assertEqual([m['cmd'] for m in server.messages],['info','reset','infer','infer'])

    def test_invalid_responses_never_return_actions(self):
        cases = [('request_id','wrong'),('session_id','wrong'),('deployment_id','wrong'),
                 ('weights','wrong'),('hand_order',['right','left']),('units','mm'),
                 ('frame','base'),('task','lemon'),('sample_ns',-10),
                 ('keypoints',np.zeros((32,2,21,3),np.float64)),
                 ('keypoint_time_s',np.arange(1,33,dtype=np.float32))]
        for key,value in cases:
            with self.subTest(field=key),served(lambda out,k=key,v=value:out.update({k:v})) as server:
                client = Client(server.host,server.port,self.cfg['server'])
                try:
                    with self.assertRaises(ValueError): client.infer(self.obs,10)
                finally: client.close()

    def test_server_error_rejected(self):
        with served(lambda out:out.update(error=dict(code='STALE_SESSION',message='expired'))) as server:
            client = Client(server.host,server.port,self.cfg['server'])
            try:
                with self.assertRaisesRegex(RuntimeError,'STALE_SESSION'): client.infer(self.obs,10)
            finally: client.close()

    def test_wrong_training_run_rejected_at_handshake(self):
        with served() as server:
            cfg = copy.deepcopy(self.cfg['server']); cfg['expected_run_name'] = 'another_run'
            with self.assertRaisesRegex(ValueError,'training run'): Client(server.host,server.port,cfg)

    def test_wrong_pinned_deployment_rejected_at_handshake(self):
        with served() as server:
            cfg = copy.deepcopy(self.cfg['server']); cfg['expected_deployment_id'] = 'unexpected'
            with self.assertRaisesRegex(ValueError,'deployment'): Client(server.host,server.port,cfg)

    def test_wrong_input_resolution_rejected(self):
        obs = dict(self.obs,head_rgb=np.zeros((480,640,3),np.uint8))
        with self.assertRaises(ValueError): validate_observation(obs)

    def test_recorded_cli_real_tcp_to_real_retarget(self):
        with tempfile.TemporaryDirectory() as tmp,served() as server:
            root = Path(tmp); cfg = copy.deepcopy(self.cfg)
            cfg['server'].update(transport='tcp',host=server.host,port=server.port)
            cfg['output_root'] = str(root/'runs'); write_json(root/'config.json',cfg)
            save_observation(root/'observation.npz',self.obs)
            main(['--config',str(root/'config.json'),'infer','--observation',str(root/'observation.npz'),'--steps','2'])
            results = list((root/'runs').glob('*/retarget.npz')); self.assertEqual(len(results),1)
            with np.load(results[0],allow_pickle=False) as z:
                self.assertEqual(z['left_joints'].shape,(3,20)); self.assertEqual(z['right_poses'].shape,(3,4,4))
            state = read_json(results[0].parent/'session.json'); self.assertTrue(state['read_only'])


class CameraAndExecutionTests(unittest.TestCase):
    def test_network_history_uses_real_time_not_frame_count(self):
        from observation import FrameHistory
        cfg = load_config()['camera']; now = time.monotonic_ns()
        buf = FrameHistory(cfg); buf.info = {'test_only':True}
        rgb = np.zeros((2,2,3),np.uint8)
        # Two 10fps frames: target is 4/30 seconds earlier, not four 10fps frames.
        buf.frames.extend([(now-100_000_000,now-100_000_000,rgb),(now,now,rgb)])
        out = buf.capture(4,now-1,timeout=.01)
        self.assertAlmostEqual(out['history_delta_s'],.1)
        self.assertAlmostEqual(out['history_error_s'],1/30,places=7)
        buf.cfg = dict(cfg,maximum_history_error_s=.025)
        with self.assertRaises(TimeoutError): buf.capture(4,now-1,timeout=.001)

    def test_recorded_fractional_forecast_times(self):
        from check_recorded import compare_recording
        keypoints = np.zeros((2,21,3),np.float32)
        future = np.zeros((32,2,21,3),np.float32)
        future[...,0] = np.arange(1,33)[:,None,None]*.001
        times = np.arange(1,33)/120
        predicted = np.zeros_like(future); predicted[...,0] = (times*30*.001)[:,None,None]
        report = compare_recording(dict(keypoints=predicted,keypoint_time_s=times),
            dict(keypoints=keypoints),dict(future_keypoints_30fps=future))
        self.assertLess(report['recorded_mpjpe_mm'],1e-6)
        self.assertAlmostEqual(report['hold_pose_mpjpe_mm'],4.125,places=5)

    def test_cli_requires_execute_and_reobserves_each_cycle(self):
        from common import replay_imports
        cfg,models,frame = setup_case(); replay_imports(cfg)
        import observation, planning, replay_control
        from types import SimpleNamespace
        for execute,persistent in ((False,False),(True,False),(True,True)):
            with self.subTest(execute=execute,persistent=persistent),tempfile.TemporaryDirectory() as tmp,served() as server:
                local = copy.deepcopy(cfg)
                local['execution']['persistent_servo'] = persistent
                local['server'].update(transport='tcp',host=server.host,port=server.port)
                local['output_root'] = str(Path(tmp)/'runs'); write_json(Path(tmp)/'config.json',local)
                captured = []
                def capture(camera,offset):
                    sample = copy.deepcopy(frame); sample['sample_ns'] = time.monotonic_ns()
                    sample['camera_info'] = {'test_only':True}; captured.append(sample['sample_ns'])
                    return sample
                hardware = SimpleNamespace(capture=capture,close=lambda:None)
                fake_camera = SimpleNamespace(close=lambda:None)
                def execute_mock(*args,**kwargs):
                    hook = kwargs.get('loop_hook')
                    if hook:
                        deadline = time.monotonic()+5.
                        while hook() is not False:
                            if time.monotonic()>deadline: raise TimeoutError('Persistent test callback stalled')
                            time.sleep(.001)
                with patch.object(observation,'Hardware',return_value=hardware),\
                     patch.object(observation,'create_camera',return_value=fake_camera),\
                     patch.object(replay_control,'acquire_replay_lock',return_value=SimpleNamespace(close=lambda:None)),\
                     patch.object(planning,'plan_chunk',return_value=('test',)),\
                     patch.object(planning,'execute_plan',side_effect=execute_mock) as executor:
                    argv = ['--config',str(Path(tmp)/'config.json'),'run','--steps','1']
                    if execute: argv += ['--execute','--iterations','2']
                    main(argv)
                    self.assertEqual(executor.call_count,(1 if persistent else 2) if execute else 0)
                    self.assertEqual(len(captured),2 if execute else 1)
                    if execute: self.assertLess(captured[0],captured[1])

    def test_canonical_reprojection_preserves_rays(self):
        import cv2
        from camera_geometry import Cam0Geometry
        cfg = load_config()
        from common import replay_imports
        replay_imports(cfg)
        from replay_pipeline import config
        c = config(cfg['replay_config'])
        meta = read_json(c['sides']['left']['handeye'])['metadata']
        source = dict(width=1280,height=800,K=meta['K'],D=meta['D'])
        target = read_json(ROOT/'vendor/cam0_color_camera_info.json')
        g = Cam0Geometry(source,target); self.assertGreater(g.coverage,.99)
        pixels = np.array([[30.,40.],[320.,200.],[600.,360.]]).reshape(-1,1,2)
        rays = cv2.undistortPoints(pixels,np.array(target['K']).reshape(3,3),np.array(target['D']))
        mapped = np.array([[g.map_x[int(y),int(x)],g.map_y[int(y),int(x)]] for x,y in pixels[:,0]]).reshape(-1,1,2)
        source_rays = cv2.undistortPoints(mapped,np.array(source['K']).reshape(3,3),np.array(source['D']))
        np.testing.assert_allclose(rays,source_rays,atol=1e-6)

    def test_current_replay_camera_profile_validated(self):
        cfg = load_config()
        from common import replay_imports
        replay_imports(cfg)
        from replay_pipeline import config
        from observation import check_camera_info,calibrated_source_profile
        c = config(cfg['replay_config']); meta = read_json(c['sides']['left']['handeye'])['metadata']
        info = dict(**calibrated_source_profile(cfg['camera'],meta),
                    frame_id='camera_color_optical_frame',distortion_model='plumb_bob')
        check_camera_info(cfg['camera'],info,c)
        with self.assertRaises(ValueError): check_camera_info(cfg['camera'],dict(info,width=info['width']+1),c)
        wrong = copy.deepcopy(info); wrong['K'][0][0] += 20
        with self.assertRaises(ValueError): check_camera_info(cfg['camera'],wrong,c)

    def test_expired_and_future_samples_rejected(self):
        with self.assertRaises(TimeoutError): check_age(time.monotonic_ns()-10**10,1)
        with self.assertRaises(TimeoutError): check_age(time.monotonic_ns()+10**10,1)

    def test_control_timing_override_preserves_points_and_raw_contract(self):
        from run import control_prediction_timing
        raw = dict(keypoints=np.zeros((32,2,21,3),np.float32),keypoint_time_s=np.arange(1,33)/120)
        original = raw['keypoint_time_s'].copy()
        changed = control_prediction_timing(raw,dict(prediction_horizon_s=1.04))
        self.assertIs(changed['keypoints'],raw['keypoints'])
        np.testing.assert_array_equal(raw['keypoint_time_s'],original)
        np.testing.assert_allclose(changed['keypoint_time_s'],np.arange(1,33)*.0325)

    def test_source_waypoints_are_not_interpolated(self):
        from common import replay_imports
        cfg,models,frame = setup_case(); replay_imports(cfg)
        import aubo_arms_replay as planner
        from replay_pipeline import config
        c = config(cfg['replay_config'])
        times = np.arange(33)*.0325
        controllers = {s:dict(joint_positions_rad=np.zeros(6),joint_min_rad=[-3]*6,
            joint_max_rad=[3]*6,joint_max_speed_rad_s=[10]*6,
            joint_max_acceleration_rad_s2=[100]*6) for s in SIDES}
        hands = {s:dict(positions_rad=frame['initial'][s]['q']) for s in SIDES}
        arrays = {s+'_wuji_joint_positions_rad':np.tile(hands[s]['positions_rad'],(33,1)) for s in SIDES}
        settings = dict(speed=.3,rate_hz=100,arms_only=False,arm_velocity_fraction=1.,
            arm_acceleration_fraction=1.,hand_velocity_rad_s=5.,hand_acceleration_rad_s2=100.,
            trajectory_sampling='source_waypoints',collision_check_policy='disabled_by_operator')
        solution = (times,np.zeros((33,6)),None)
        with patch.object(planner,'sample_replay',side_effect=AssertionError('Unexpected interpolation')),\
             patch.object(planner,'Scene',side_effect=AssertionError('Unexpected collision query')):
            trajectory,report = planner.runtime_pair(c,controllers,hands,solution,solution,settings,arrays)
        self.assertEqual(trajectory.shape,(33,52))
        self.assertAlmostEqual(report['duration_s'],1.04/.3)
        self.assertAlmostEqual(report['command_interval_s'],.0325/.3)
        self.assertEqual(report['predicted_waypoints'],32)

    def test_camera_bad_frame_warning_is_nonfatal(self):
        from observation import FrameHistory
        history = FrameHistory(dict())
        history.warn_frame('camera_exposure_timestamp_invalid',age_s=1.)
        self.assertIsNone(history.error)
        self.assertEqual(history.frame_warnings[-1]['action'],'drop_frame_and_continue')

    def test_observation_waits_and_recaptures_after_motion(self):
        from observation import Hardware,ArmsNotSettled
        h = object.__new__(Hardware); h.cancel_wait=threading.Event()
        h.wait_until_steady = unittest.mock.Mock()
        expected = {'fresh':True}
        h._capture_once = unittest.mock.Mock(side_effect=[ArmsNotSettled('moving'),expected])
        self.assertIs(h.capture(None,4),expected)
        self.assertEqual(h.wait_until_steady.call_count,2)
        self.assertEqual(h._capture_once.call_count,2)

    def test_stationary_wait_can_be_cancelled(self):
        from observation import Hardware
        from common import replay_imports
        replay_imports(load_config())
        h = object.__new__(Hardware); h.cancel_wait=threading.Event();h.cancel_wait.set()
        with self.assertRaises(InterruptedError): h.wait_until_steady()

    def test_arm_start_drift_is_warning_only(self):
        cfg,models,frame = setup_case()
        arms = {s:dict(joint_positions_rad=np.zeros(6)) for s in SIDES}
        hands = {s:dict(positions_rad=frame['initial'][s]['q']) for s in SIDES}
        unchanged_start(frame['initial'],arms,hands,cfg['execution'])
        arms['right']['joint_positions_rad'] = np.ones(6)
        warnings = unchanged_start(frame['initial'],arms,hands,cfg['execution'])
        self.assertEqual(len(warnings),1)
        self.assertEqual(warnings[0]['type'],'arm_moved_since_observation')
        self.assertEqual(warnings[0]['side'],'right')
        self.assertEqual(warnings[0]['action'],'warn_only')

    def test_hand_drift_is_warning_only(self):
        cfg,models,frame = setup_case()
        arms = {s:dict(joint_positions_rad=np.zeros(6)) for s in SIDES}
        hands = {s:dict(positions_rad=frame['initial'][s]['q'].copy()) for s in SIDES}
        hands['left']['positions_rad'][7] += .01
        warnings = unchanged_start(frame['initial'],arms,hands,cfg['execution'])
        self.assertEqual(len(warnings),1)
        self.assertEqual(warnings[0]['side'],'left')
        self.assertEqual(warnings[0]['joint_index'],7)
        self.assertEqual(warnings[0]['action'],'warn_only')


if __name__ == '__main__': unittest.main()
