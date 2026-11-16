"""Read-only network/ROS image capture, measured Aubo/Wuji FK and calibration."""
from collections import deque
from pathlib import Path
import threading
import time
import numpy as np
from common import ROOT, SIDES, read_json, replay_imports, transform
from hand_kinematics import camera_points


def calibration_chain(cfg, replay):
    # Real-time data is in the current physical camera: no offline table/task mapping.
    return {side:transform(read_json(replay['sides'][side]['handeye'])['T_base_camera']) for side in SIDES}


def calibrated_source_profile(camera,metadata):
    height,width = metadata['image_size']
    selected_width = int(camera.get('width',width)); selected_height = int(camera.get('height',height))
    sx,sy = selected_width/width,selected_height/height
    if not np.isclose(sx,sy,rtol=0,atol=1e-9):
        raise ValueError('Camera profile changes calibrated aspect ratio/crop')
    k = np.asarray(metadata['K'],float).reshape(3,3).copy()
    k[0,:] *= sx;k[1,:] *= sy
    return dict(width=selected_width,height=selected_height,K=k.tolist(),D=metadata['D'])


def check_camera_info(camera, info, replay):
    expected = read_json(ROOT/'vendor/cam0_color_camera_info.json')
    if info['frame_id'] != camera['frame_id']: raise ValueError('Wrong CameraInfo optical frame')
    if info.get('distortion_model') not in ('plumb_bob','rational_polynomial'):
        raise ValueError('Unsupported camera distortion model')
    k = np.asarray(info['K'],float).reshape(3,3); d = np.asarray(info['D'],float)
    def check(k2,d2,label):
        d2 = np.asarray(d2,float)
        n = max(len(d),len(d2)); a = np.pad(d,(0,n-len(d))); b = np.pad(d2,(0,n-len(d2)))
        if not np.allclose(k,np.asarray(k2).reshape(3,3),rtol=0,atol=camera['intrinsics_tolerance_px']):
            raise ValueError(label+' intrinsics differ from current camera')
        if not np.allclose(a,b,rtol=0,atol=camera['distortion_tolerance']):
            raise ValueError(label+' distortion differs from current camera')
    mode = camera['geometry_mode']
    if mode == 'strict_raw':
        if (info['width'],info['height']) != (640,400): raise ValueError('strict_raw requires native 640x400')
        check(expected['K'],expected['D'],'XFlow cam0')
    elif mode != 'canonical_cam0': raise ValueError('Unknown camera geometry mode')
    for side in SIDES:
        handeye = read_json(replay['sides'][side]['handeye'])
        if handeye.get('mode') != 'eye_to_hand': raise ValueError('Expected fixed camera handeye')
        if handeye.get('parent_frame') != side+'_aubo_base': raise ValueError('Wrong handeye arm frame')
        if handeye.get('child_frame') != camera['frame_id']: raise ValueError('Wrong handeye optical frame')
        profile = calibrated_source_profile(camera,handeye['metadata'])
        if (profile['width'],profile['height']) != (info['width'],info['height']):
            raise ValueError('Handeye selected camera resolution differs')
        # Accept scaled calibration only when actual SDK K/D also match it.
        check(profile['K'],profile['D'],side+' handeye')


class CameraPairTimeout(TimeoutError):
    """No valid pair before the observation deadline."""


class FrameHistory:
    def __init__(self,cfg):
        self.cfg = cfg; self.error = None
        self.frames = deque(maxlen=90); self.info = None; self.condition = threading.Condition()
        self.frame_warnings = deque(maxlen=100); self._last_frame_warning_s = 0.

    def warn_frame(self,kind,**details):
        warning = dict(type=kind,action='drop_frame_and_continue',**details)
        with self.condition:
            self.frame_warnings.append(warning)
            self.condition.notify_all()
        now = time.monotonic()
        if now-self._last_frame_warning_s >= 1.:
            print('WARNING',warning,flush=True); self._last_frame_warning_s = now

    def capture(self,offset,after_ns,timeout=5.):
        deadline = time.monotonic()+timeout
        best_error_s = None
        with self.condition:
            while True:
                if self.error: raise self.error
                earliest = max(after_ns, time.monotonic_ns()-round(self.cfg['maximum_frame_age_s']*1e9))
                for current in reversed(self.frames):
                    if not self.info or current[1] < earliest: break
                    target = current[0]-round(offset/self.cfg['fps']*1e9)
                    history = min(self.frames,key=lambda row:abs(row[0]-target))
                    error_s = abs(history[0]-target)/1e9
                    best_error_s = error_s if best_error_s is None else min(best_error_s,error_s)
                    if error_s <= self.cfg['maximum_history_error_s']:
                        return dict(head_rgb=current[2].copy(),history_rgb=history[2].copy(),
                            sample_ns=int(current[1]),camera_info=dict(self.info),
                            history_delta_s=(current[0]-history[0])/1e9,history_error_s=error_s)
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    age = (time.monotonic_ns()-self.frames[-1][1])/1e9 if self.frames else None
                    raise CameraPairTimeout(f'No valid camera pair in {timeout:g}s: buffered={len(self.frames)}, '
                        f'camera_info={bool(self.info)}, latest_age_s={age}, best_history_error_s={best_error_s}')
                self.condition.wait(min(.1,remaining))


class RosCamera(FrameHistory):
    def __init__(self, cfg):
        super().__init__(cfg)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        self.rclpy = rclpy
        self.context = rclpy.context.Context(); rclpy.init(context=self.context)
        self.node = rclpy.create_node('aubo_wuji_xflow_camera',context=self.context)
        self.node.create_subscription(Image,cfg['image_topic'],self.on_image,qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo,cfg['info_topic'],self.on_info,qos_profile_sensor_data)
        self.executor = SingleThreadedExecutor(context=self.context); self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin,daemon=True); self.thread.start()

    def on_info(self,msg):
        with self.condition:
            self.info = dict(width=msg.width,height=msg.height,K=list(msg.k),D=list(msg.d),
                frame_id=msg.header.frame_id,distortion_model=msg.distortion_model)
            self.condition.notify_all()

    def on_image(self,msg):
        try:
            if msg.encoding not in ('rgb8','bgr8'):
                raise ValueError('Expected uncompressed raw rgb8/bgr8 image')
            if msg.header.frame_id != self.cfg['frame_id']: raise ValueError('Wrong image frame')
            header_ns = int(msg.header.stamp.sec)*10**9+int(msg.header.stamp.nanosec)
            age_ns = self.node.get_clock().now().nanoseconds-header_ns
            if not 0 <= age_ns <= self.cfg['maximum_frame_age_s']*1e9:
                raise ValueError('Stale/future image header or incompatible ROS clock')
            sample_ns = time.monotonic_ns()-age_ns
            pixels = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
            rgb = pixels[:,:msg.width*3].reshape(msg.height,msg.width,3)
            if msg.encoding == 'bgr8': rgb = rgb[...,::-1]
            with self.condition:
                if self.frames and header_ns <= self.frames[-1][0]: raise ValueError('Image timestamps not increasing')
                self.frames.append((header_ns,sample_ns,np.ascontiguousarray(rgb)))
                self.condition.notify_all()
        except Exception as error:
            with self.condition: self.error = error; self.condition.notify_all()

    def close(self):
        self.executor.shutdown(timeout_sec=2.); self.thread.join(timeout=2.)
        self.node.destroy_node(); self.rclpy.shutdown(context=self.context)


def load_orbbec():
    import sys
    import importlib
    directory = ROOT/'vendor/camera_runtime'
    if directory.is_dir(): sys.path.insert(0,str(directory))
    return importlib.import_module('pyorbbecsdk')


class OrbbecNetCamera(FrameHistory):
    """Direct network camera; RGB profile and SDK global exposure timestamps."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.stop = threading.Event(); self.pipeline = None; self.thread = None
        ob = load_orbbec()
        ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
        self.context = ob.Context()
        try:
            self.device = self.context.create_net_device(cfg['ip'],int(cfg['port']))
            identity = self.device.get_device_info()
            serial = identity.get_serial_number()
            if serial != cfg['serial']: raise ValueError('Network camera serial mismatch')
            self.pipeline = ob.Pipeline(self.device)
            profile = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
                int(cfg['width']),int(cfg['height']),ob.OBFormat.RGB,int(cfg['capture_fps']))
            k = profile.get_intrinsic(); d = profile.get_distortion()
            self.info = dict(width=int(k.width),height=int(k.height),
                K=[[k.fx,0.,k.cx],[0.,k.fy,k.cy],[0.,0.,1.]],
                D=[d.k1,d.k2,d.p1,d.p2,d.k3,d.k4,d.k5,d.k6],distortion_model='rational_polynomial',
                frame_id=cfg['frame_id'],serial=serial,device_name=identity.get_name(),
                capture_fps=int(profile.get_fps()),capture_format='RGB',
                timestamp_source='SDK get_global_timestamp_us; camera exposure mapped to host clock',
                model_history_reference_fps=cfg['fps'])
            streams = ob.Config(); streams.enable_stream(profile)
            self.pipeline.start(streams)
            self.thread = threading.Thread(target=self.loop,daemon=True); self.thread.start()
        except BaseException:
            self.close(); raise

    def loop(self):
        try:
            while not self.stop.is_set():
                # This wheel holds the Python GIL during its native wait. A long
                # blocking wait starves the Wuji feedback and controller reader.
                # Keep native waits short and yield between empty polls.
                frames = self.pipeline.wait_for_frames(1)
                if frames is None:
                    self.stop.wait(.002)
                    continue
                color = frames.get_color_frame()
                if color is None: continue
                stamp = int(color.get_global_timestamp_us())*1000
                age_ns = time.time_ns()-stamp
                if stamp <= 0 or not 0 <= age_ns <= self.cfg['maximum_frame_age_s']*1e9:
                    self.warn_frame('camera_exposure_timestamp_invalid',timestamp_ns=stamp,
                        age_s=float(age_ns/1e9),maximum_age_s=float(self.cfg['maximum_frame_age_s']))
                    continue
                sample = time.monotonic_ns()-age_ns
                rgb = np.asarray(color.get_data(),np.uint8).reshape(self.info['height'],self.info['width'],3).copy()
                with self.condition:
                    if self.frames and stamp <= self.frames[-1][0]:
                        previous = self.frames[-1][0]
                    else: previous = None
                if previous is not None:
                    self.warn_frame('camera_exposure_timestamp_not_increasing',timestamp_ns=stamp,
                        previous_timestamp_ns=int(previous))
                    continue
                with self.condition:
                    self.frames.append((stamp,sample,rgb)); self.condition.notify_all()
        except Exception as error:
            with self.condition: self.error = error; self.condition.notify_all()

    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=1.5)
        if self.pipeline is not None:
            try: self.pipeline.stop()
            finally: self.pipeline = None
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.)
        self.device = None; self.context = None


def create_camera(cfg):
    backend = cfg.get('backend','ros')
    if backend == 'ros': return RosCamera(cfg)
    if backend == 'orbbec_net': return OrbbecNetCamera(cfg)
    raise ValueError('Unsupported camera backend: '+backend)


class ArmsNotSettled(RuntimeError):
    """Transient observation failure; wait and capture a fresh observation."""


class Hardware:
    def __init__(self,cfg,models):
        replay_imports(cfg)
        from replay_pipeline import config
        from device_backends import Aubo
        from read_hands import Hands
        self.cfg = cfg; self.models = models; self.arms = {}; self.hands = None
        self.cancel_wait = threading.Event()
        self.geometry = None; self.geometry_info = None
        self.replay = config(cfg['replay_config']); self.runtime = read_json(cfg['runtime_config'])
        self.base_T_camera = calibration_chain(cfg,self.replay)
        if cfg['camera']['geometry_mode'] == 'canonical_cam0':
            from camera_geometry import Cam0Geometry
            meta = read_json(self.replay['sides']['left']['handeye'])['metadata']
            source = calibrated_source_profile(cfg['camera'],meta)
            # Precompute before creating high-rate hand subscriptions. Live K/D
            # still must match this calibration before any image is transformed.
            self.geometry = Cam0Geometry(source,read_json(ROOT/'vendor/cam0_color_camera_info.json'))
        try:
            for side in SIDES:
                if not self.replay['sides'][side].get('mount_confirmed'): raise ValueError('Unconfirmed hand mount')
                self.arms[side] = Aubo(self.replay['sides'][side])
            self.hands = Hands(self.runtime['hands'])
        except BaseException: self.close(); raise

    def measured(self):
        from read_hands import check_current_limit
        start = time.monotonic_ns()
        arms = {side:arm.snapshot() for side,arm in self.arms.items()}
        hands = self.hands.snapshot()
        check_current_limit(hands,1.5)
        return dict(controllers=arms,hands=hands,read_begin_ns=start,read_end_ns=time.monotonic_ns())

    def capture(self,camera,offset):
        retries = 0
        while True:
            self.wait_until_steady()
            try:
                return self._capture_once(camera,offset)
            except CameraPairTimeout as error:
                if retries >= self.cfg['camera'].get('pair_timeout_retries',0): raise
                retries += 1
                print('WARNING',str(error),'; reacquiring image and joint state',flush=True)
            except ArmsNotSettled as error:
                print('WARNING',str(error),'; waiting and recapturing fresh state/image',flush=True)
                if self.cancel_wait.wait(.05): raise InterruptedError('Observation wait cancelled')

    def wait_until_steady(self):
        last_warning = 0.
        from read_hands import check_current_limit
        while True:
            if self.cancel_wait.is_set(): raise InterruptedError('Observation wait cancelled')
            moving = [side for side,arm in self.arms.items() if not arm.state.isSteady()]
            if not moving: return
            check_current_limit(self.hands.snapshot(),1.5)
            now = time.monotonic()
            if now-last_warning>=1.:
                print('WARNING arms not steady:',','.join(moving),'; waiting without advancing trajectory',flush=True)
                last_warning = now
            self.cancel_wait.wait(.05)

    def _capture_once(self,camera,offset):
        if self.geometry_info is None and self.cfg['camera']['geometry_mode'] == 'canonical_cam0':
            # Build the pixel map and prime the camera BEFORE the synchronized
            # measurement interval; then capture a fresh image with real state.
            first = camera.capture(offset,time.monotonic_ns())
            check_camera_info(self.cfg['camera'],first['camera_info'],self.replay)
            self.geometry_info = dict(first['camera_info'])
        start = time.monotonic_ns()
        before = {s:np.asarray(a.state.getJointPositions(),float) for s,a in self.arms.items()}
        hands_before = self.hands.snapshot()
        frame = camera.capture(offset,start)
        check_camera_info(self.cfg['camera'],frame['camera_info'],self.replay)
        if self.cfg['camera']['geometry_mode'] == 'canonical_cam0':
            from camera_geometry import Cam0Geometry
            if self.geometry_info != frame['camera_info']: raise ValueError('Camera calibration changed during session')
            frame['head_rgb'] = self.geometry.apply(frame['head_rgb'])
            frame['history_rgb'] = self.geometry.apply(frame['history_rgb'])
            frame['camera_info']['canonical_cam0_coverage'] = self.geometry.coverage
        hands = self.hands.snapshot()
        after = {s:np.asarray(a.state.getJointPositions(),float) for s,a in self.arms.items()}
        end = time.monotonic_ns()
        if (end-frame['sample_ns'])/1e9 > self.cfg['camera']['maximum_sync_span_s']:
            raise ValueError(f'Image and measured state synchronization span exceeded: {(end-frame["sample_ns"])/1e9:.4f}s')
        from read_hands import check_current_limit
        check_current_limit(hands,1.5)
        initial = {}; kp = []; warnings = []
        for side in SIDES:
            arm = self.arms[side]
            if arm.robot.getMotionControl().isServoModeEnabled() and not getattr(self,'owns_servo',False):
                raise ValueError('Servo is controlled by another command source')
            if not arm.state.isSteady():
                raise ArmsNotSettled('Arm not steady during observation: '+side)
            q = np.asarray(hands[side]['positions_rad'],float)
            if np.max(abs(before[side]-after[side])) > self.cfg['execution']['stationary_arm_tolerance_rad']:
                raise ArmsNotSettled('Arm moved during observation: '+side)
            hand_delta = abs(q-np.asarray(hands_before[side]['positions_rad']))
            hand_joint = int(np.argmax(hand_delta)); hand_max = float(hand_delta[hand_joint])
            if hand_max > self.cfg['execution']['stationary_hand_tolerance_rad']:
                warning = dict(type='hand_moved_during_observation',side=side,
                    joint_index=hand_joint,max_delta_rad=hand_max,
                    threshold_rad=float(self.cfg['execution']['stationary_hand_tolerance_rad']),
                    action='warn_only')
                warnings.append(warning)
                print('WARNING',warning,flush=True)
            base_T_root = arm.fk(after[side]) @ transform(self.replay['sides'][side]['flange_T_wuji_base'])
            camera_T_root = np.linalg.inv(self.base_T_camera[side]) @ base_T_root
            initial[side] = dict(q=q,camera_T_root=camera_T_root,arm_q=after[side])
            kp.append(camera_points(self.models[side],q,camera_T_root))
        frame['keypoints'] = np.asarray(kp,np.float32); frame['initial'] = initial
        frame['warnings'] = warnings
        return frame

    def close(self):
        self.cancel_wait.set()
        if self.hands is not None: self.hands.close()
        for arm in self.arms.values(): arm.close()
