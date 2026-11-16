"""AgiBot 观测、初始化和动作执行。主客户端只需调用这三个接口。"""
from pathlib import Path
import struct
import socket
import time

import cv2
import numpy as np

from replay import BODY_NAMES, JOINT_NAMES, SELECT_JOINTS, head_target, move_to_first_frame, read_positions, send_servo
from genie_msgs_pb.msg.TrajectoryTrackingControl_pb2 import TrajectoryTrackingControl
from genie_msgs_pb.msg.RetargetInfo_pb2 import RetargetInfo

DEFAULT_EPISODE = Path(__file__).resolve().parents[1] / 'data/raw_data/clothes/0912_20260912112854'
# 模型顺序：左臂7、左夹爪、右臂7、右夹爪、头部3。
from agibot_tasks import MODEL_JOINT_NAMES, SHARPA_LEFT, SHARPA_RIGHT, SHARPA61_EEF_HEAD_INDICES
MOTOR_INDICES = list(range(7)) + list(range(8, 15)) + [16, 17, 18]


def _slerp_xyzw(a, b, alpha):
    """Shortest-arc interpolation of already-normalized quaternions; not IK."""
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, 0., 1.)
    if dot > .9995:
        q = (1-alpha)*a + alpha*b
    else:
        theta = np.arccos(dot)
        q = (np.sin((1-alpha)*theta)*a + np.sin(alpha*theta)*b) / np.sin(theta)
    return q / np.linalg.norm(q)


def read_pbdat(path: Path, message_type):
    """按官方文件格式逐帧读取；只在离线测试、读取初始化姿态时使用。"""
    with path.open('rb') as f:
        count, = struct.unpack('<q', f.read(8))
        if count <= 0:
            raise ValueError(f'Empty recording: {path}')
        for _ in range(count):
            timestamp, _, _, _, size = struct.unpack('<5q', f.read(40))
            if size <= 0 or size > 1024 * 1024:
                raise ValueError('Invalid PBDAT payload size')
            payload = f.read(size)
            if len(payload) != size:
                raise ValueError('Truncated PBDAT payload')
            yield timestamp, message_type.FromString(payload)


def _nearest(path: Path, message_type, timestamp: int):
    best, best_distance = None, float('inf')
    for t, msg in read_pbdat(path, message_type):
        distance = abs(t - timestamp)
        if distance < best_distance:
            best, best_distance = msg, distance
        if t >= timestamp:
            break
    if best is None or best_distance > 100_000_000:
        raise ValueError(f'No matching state within 100 ms: {path}')
    return best


def recorded_observation(episode: Path, frame: int = 0) -> dict:
    """读取指定头部 RGB 帧，按纳秒时间戳匹配关节和两个夹爪。不会连接机器人。"""
    from genie_msgs_pb.msg.JointState_pb2 import JointState
    from genie_msgs_pb.msg.EndState_pb2 import EndState
    camera_dir = episode / 'camera/head_color'
    timestamps = (camera_dir / 'head_color.txt').read_text().splitlines()
    if not 0 <= frame < len(timestamps):
        raise ValueError(f'frame must be in [0,{len(timestamps)-1}]')
    timestamp = int(timestamps[frame].split()[0])
    cap = cv2.VideoCapture(str(camera_dir / 'head_color.h265'))
    try:
        # 裸 H265 不依赖不可靠的随机 seek，顺序解码到选中的帧。
        for _ in range(frame + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError('Cannot decode recorded head_color video')
    finally:
        cap.release()
    joints = _nearest(episode / 'record/-hal-joint_state.pbdat', JointState, timestamp)
    left = _nearest(episode / 'record/-hal-left_ee_data.pbdat', EndState, timestamp)
    right = _nearest(episode / 'record/-hal-right_ee_data.pbdat', EndState, timestamp)
    by_name = dict(zip(joints.name, joints.motor_position))
    by_name[MODEL_JOINT_NAMES[7]] = left.end_state[0].position
    by_name[MODEL_JOINT_NAMES[15]] = right.end_state[0].position
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return {'active_rgb': cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR),
            'state': np.asarray([by_name[n] for n in MODEL_JOINT_NAMES], dtype=np.float32)}


class AgiBotRobot:
    def __init__(self, *, with_eef=False) -> None:
        # 发现服务不可达时 SDK 可能崩溃，先检查连接；该检查不发送运动指令。
        try:
            with socket.create_connection(('10.42.1.101', 2379), timeout=2):
                pass
        except OSError as exc:
            raise RuntimeError('机器人发现服务 10.42.1.101:2379 不可达；请检查开机、网线和 SDK 服务') from exc
        # 延迟导入：离线 test 不初始化 GDK，也不会连接机器人。
        import agibot_gdk as gdk
        self.gdk = gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError('GDK initialization failed')
        self.robot, self.camera = gdk.Robot(), gdk.Camera()
        self.tf = gdk.TF() if with_eef else None
        self.last_image_timestamp = None
        self.last_image_timestamps = {}
        time.sleep(2)

    def get_observation(self) -> dict:
        """读取头部彩色相机 + 实测19维状态，图像统一转为 RGB 640x480。"""
        gdk = self.gdk
        image = self.camera.get_latest_image(gdk.CameraType.kHeadColor, 1000.0)
        if image is None:
            raise RuntimeError('No head RGB image')
        # GDK 的相机/关节时间戳使用相对时钟，不能与 Unix 时间直接相减。
        # 首次读取时等一帧，确认相机确实在更新；之后要求时间戳递增。
        if self.last_image_timestamp is None:
            first_timestamp = image.timestamp_ns
            time.sleep(0.1)
            image = self.camera.get_latest_image(gdk.CameraType.kHeadColor, 1000.0)
            if image is None or image.timestamp_ns <= first_timestamp:
                raise RuntimeError('No fresh head RGB image')
        elif image.timestamp_ns <= self.last_image_timestamp:
            raise RuntimeError('No fresh head RGB image')
        raw = np.asarray(image.data, dtype=np.uint8)
        if image.encoding in (gdk.Encoding.JPEG, gdk.Encoding.PNG):
            bgr = cv2.imdecode(raw.reshape(-1), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError('Cannot decode live image')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif image.encoding == gdk.Encoding.UNCOMPRESSED:
            rgb = raw.reshape(image.height, image.width, 3)
            if image.color_format == gdk.ColorFormat.BGR:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            elif image.color_format != gdk.ColorFormat.RGB:
                raise ValueError(f'Unsupported color format: {image.color_format}')
        else:
            raise ValueError(f'Unsupported encoding: {image.encoding}')
        joint_state = self.robot.get_joint_states()
        if abs(joint_state['timestamp'] - image.timestamp_ns) > 500_000_000:
            raise RuntimeError('Image and joint state differ by more than 500 ms')
        states = joint_state['states']
        by_name = {s['name']: s['motor_position'] for s in states}
        end = self.robot.get_end_state()
        by_name[MODEL_JOINT_NAMES[7]] = end['left_end_state']['end_states'][0]['position']
        by_name[MODEL_JOINT_NAMES[15]] = end['right_end_state']['end_states'][0]['position']
        state = np.asarray([by_name[n] for n in MODEL_JOINT_NAMES], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError('Non-finite robot state')
        self.last_image_timestamp = image.timestamp_ns
        return {'active_rgb': cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR), 'state': state}

    def get_observation61(self, hands) -> dict:
        """Head RGB + measured base_link EEF, native hand qpos, head qpos.

        Keep increasing timestamps and the 1 second image/joint check.
        Wait for fresh head RGB.
        No old gripper reads and no movement/IK in observation assembly.
        """
        if self.tf is None:
            raise RuntimeError('61D observations require AgiBotRobot(with_eef=True)')
        gdk = self.gdk
        images, stamps = {}, {}
        cameras = (('head_color', gdk.CameraType.kHeadColor),)
        def progress(stage, **detail):
            self.capture_progress = dict(stage=stage, started_ns=time.monotonic_ns(), **detail)

        previous = dict(self.last_image_timestamps)
        started = time.monotonic()
        raw_images, errors = {}, {}
        while True:
            for name, camera_type in cameras:
                try:
                    progress('camera_read', camera=name, previous_timestamps=previous,
                             latest_timestamps={k: v.timestamp_ns for k,v in raw_images.items()})
                    image = self.camera.get_latest_image(camera_type, 1000.0)
                    if image is None:
                        raise RuntimeError('null image')
                    raw_images[name] = image
                    previous.setdefault(name, image.timestamp_ns)
                    errors.pop(name, None)
                except RuntimeError as exc:
                    raw_images.pop(name, None)
                    errors[name] = str(exc)
            pending = [name for name, _ in cameras if name not in raw_images
                       or raw_images[name].timestamp_ns <= previous[name]]
            if not pending:
                break
            elapsed = time.monotonic() - started
            if elapsed >= 1.0:
                import json
                detail = dict(pending_cameras=pending, waited_s=elapsed,
                              previous_timestamps=previous,
                              latest_timestamps={k: v.timestamp_ns for k,v in raw_images.items()},
                              read_errors=errors)
                error = RuntimeError('No fresh RGB image: ' + ', '.join(pending)
                                     + '; ' + json.dumps(detail))
                error.diagnostic = detail
                raise error
            progress('camera_wait', pending_cameras=pending, waited_s=elapsed,
                     previous_timestamps=previous,
                     latest_timestamps={k: v.timestamp_ns for k,v in raw_images.items()},
                     read_errors=dict(errors))
            time.sleep(min(.01, 1.0-elapsed))
        progress('image_decode')
        for name, _ in cameras:
            image = raw_images[name]
            raw = np.asarray(image.data, dtype=np.uint8)
            if image.encoding in (gdk.Encoding.JPEG, gdk.Encoding.PNG):
                bgr = cv2.imdecode(raw.reshape(-1), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f'Cannot decode live image: {name}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif image.encoding == gdk.Encoding.UNCOMPRESSED:
                rgb = raw.reshape(image.height, image.width, 3)
                if image.color_format == gdk.ColorFormat.BGR:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                elif image.color_format != gdk.ColorFormat.RGB:
                    raise ValueError(f'Unsupported color format: {name}: {image.color_format}')
            else:
                raise ValueError(f'Unsupported encoding: {name}: {image.encoding}')
            images[name] = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            stamps[name] = image.timestamp_ns
        progress('joint_read_or_sdk_lock')
        joint_state = self.robot.get_joint_states()
        for name, stamp in stamps.items():
            if abs(joint_state['timestamp'] - stamp) > 1_000_000_000:
                raise RuntimeError(f'Image and joint state differ by more than 1 second: {name}')
        joints = {s['name']: s['motor_position'] for s in joint_state['states']}
        progress('hand_read_or_sdk_lock')
        hand_qpos = hands.read_qpos()
        parts = []
        for frame, qpos in zip(('arm_l_end_link', 'arm_r_end_link'), hand_qpos):
            progress('tf_read_or_sdk_lock', frame=frame)
            transform = self.tf.get_tf_from_base_link(frame)
            p, q = transform.translation, transform.rotation
            parts.extend((p.x, p.y, p.z, q.x, q.y, q.z, q.w, *qpos))
        parts.extend(joints[n] for n in JOINT_NAMES[:3])
        state = np.asarray(parts, dtype=np.float32)
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError('Non-finite or invalid measured 61D robot state')
        # Preserve measured named arm/head joints for the existing drift check.
        if not np.isfinite([joints[n] for n in JOINT_NAMES]).all():
            raise ValueError('Non-finite robot joint state')
        self.last_image_timestamps = stamps
        return {'images': images, 'state': state, 'joints': joints}

    def reset(self, episode: Path = DEFAULT_EPISODE) -> None:
        """复用已验证的 replay 初始化方式，并恢复两个夹爪的首帧角度。"""
        from genie_msgs_pb.msg.JointState_pb2 import JointState
        from genie_msgs_pb.msg.EndState_pb2 import EndState
        _, msg = next(read_pbdat(episode / 'record/-hal-joint_state.pbdat', JointState))
        by_name = dict(zip(msg.name, msg.motor_position))
        body = tuple(by_name[n] for n in BODY_NAMES)
        target = SELECT_JOINTS(by_name)
        left = next(read_pbdat(episode / 'record/-hal-left_ee_data.pbdat', EndState))[1].end_state[0].position
        right = next(read_pbdat(episode / 'record/-hal-right_ee_data.pbdat', EndState))[1].end_state[0].position
        if not np.isfinite(body + target + (left, right)).all():
            raise ValueError('Invalid reset pose')
        print(f'自动复位到 {episode.name} 的首帧', flush=True)
        # 启动前先确认所有相关关节反馈有效，再发送任何复位指令。
        all_current = read_positions(self.robot, BODY_NAMES + JOINT_NAMES)
        current = all_current[:5]
        request = self.gdk.JointControlReq()
        request.joint_names, request.joint_positions = list(BODY_NAMES), list(body)
        request.joint_velocities = [0.15] * 5
        delta = max(abs(a-b) for a,b in zip(current,body))
        request.life_time = max(10.0, 2 * delta / .15 + 5)
        if self.robot.joint_control_request(request) != 0:
            raise RuntimeError('Body reset failed')
        actual = read_positions(self.robot, BODY_NAMES)
        if max(abs(a-b) for a,b in zip(actual,body)) > .05:
            raise RuntimeError('Body did not reach reset pose')
        move_to_first_frame(self.robot, target, (left, right))

    def execute(self, actions: np.ndarray, start_state: np.ndarray, check=None) -> None:
        """20Hz 模型动作插成 100Hz servo：24 个目标执行 1.2 秒。"""
        # 交互客户端传入键盘检查；原有调用无需改动。
        check = check or (lambda: None)
        check()
        actions = np.asarray(actions, dtype=np.float64)
        start_state = np.asarray(start_state, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 19 or not 1 <= len(actions) <= 32:
            raise ValueError('Expected 1..32 actions with 19 dimensions')
        if start_state.shape != (19,) or not np.isfinite(actions).all() or not np.isfinite(start_state).all():
            raise ValueError('Non-finite or invalid action/state')
        self._execute_joint_path(actions,start_state,check,grippers=True)

    def execute_eef(self, actions: np.ndarray, start_eef: np.ndarray,
                    start_joints: dict, check=None) -> None:
        """Send absolute dual-arm EEF to GDK; head remains three-joint qpos.

        actions [1..32,17] and start_eef [17]: left xyz+xyzw, right xyz+xyzw,
        head qpos. EEF is in base_link, metres; head is radians. Both initial
        poses and named head/arm start_joints must be measured with the
        observation used for inference, NOT obtained from the model output.

        Retains 20 Hz -> 100 Hz timing, cancellation, observation drift and
        head joint checks. GDK handles arm IK internally; there is no local
        arm target-qpos step check. No gripper/Sharpa commands are sent.
        Existing 19D client routing is unchanged. Hardware coexistence of
        head servo and dual-arm EEF requires separate real-machine validation.
        """
        self._execute_eef_path(actions, start_eef, start_joints, check)

    def execute_sharpa61(self, actions, start_state, start_joints, hands, check=None, *,
                         startup_transition=False):
        """Execute 1..50 absolute 61D targets with explicitly connected hands.

        State/action: left EEF7, left hand22, right EEF7, right hand22, head3.
        start_state must be measured feedback associated with inference input.
        No automatic connection/reset/enable and no changes to the 19D client.
        Caller owns connect()/close() in try/finally; errors stop this loop.
        """
        actions = np.asarray(actions, dtype=float)
        start_state = np.asarray(start_state, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 61 or not 1 <= len(actions) <= 50:
            raise ValueError('Expected 1..50 Sharpa actions with 61 dimensions')
        if start_state.shape != (61,) or not np.isfinite(actions).all() or not np.isfinite(start_state).all():
            raise ValueError('Non-finite or invalid Sharpa action/state')
        if hands is None:
            raise ValueError('61D execution requires a connected dual-hand adapter')
        hands.require_connected()  # Local lifecycle state only; no SDK mode query.
        eef = list(SHARPA61_EEF_HEAD_INDICES)
        hand_actions = np.concatenate([actions[:, SHARPA_LEFT], actions[:, SHARPA_RIGHT]], axis=1)
        start_hands = np.r_[start_state[SHARPA_LEFT], start_state[SHARPA_RIGHT]]
        self._execute_eef_path(actions[:, eef], start_state[eef], start_joints, check,
                               hands=hands, hand_actions=hand_actions, start_hands=start_hands,
                               max_frames=50, startup_transition=startup_transition)

    def _send_combined_eef_head(self, target, tick, *, waist_q=None):
        """One official GDK trajectory sample; arm IK stays in the factory controller.

        GDK 3.3.8's dict adapter adds zero-valued elbow reference poses. Build
        its official message without those additional IK references instead.
        Publish on /gdk (NOT /wbc) so the factory arbitrator remains in the path.
        Head model order is idx11/12/13 = yaw/roll/pitch, not yaw/pitch/roll.
        """
        if waist_q is not None and (len(waist_q) != 5 or not np.isfinite(waist_q).all()):
            raise ValueError('Waist target requires five finite joint positions')
        msg = TrajectoryTrackingControl(trajectory_reference_time=.01)
        stamp = time.time_ns()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
        msg.header.frame_id = 'base_link'
        frame = msg.retarget_groups_trajectory.add()
        for group, name, offset in ((RetargetInfo.GROUP_LEFT_ARM, 'arm_l_end_link', 0),
                                     (RetargetInfo.GROUP_RIGHT_ARM, 'arm_r_end_link', 7)):
            part = frame.retarget_infos.add(group_id=group, input_type=RetargetInfo.INPUT_GDK,
                                            control_type=RetargetInfo.ABS_POSE, frame_id='base_link')
            part.target_frame_names.append(name)
            pose = part.target_frame_poses.add()
            pose.position.x, pose.position.y, pose.position.z = map(float, target[offset:offset+3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, target[offset+3:offset+7])
        for group, value in zip((RetargetInfo.GROUP_HEAD_YAW, RetargetInfo.GROUP_HEAD_ROLL,
                                  RetargetInfo.GROUP_HEAD_PITCH), target[14:17]):
            part = frame.retarget_infos.add(group_id=group, input_type=RetargetInfo.INPUT_GDK,
                                            control_type=RetargetInfo.ABS_JOINT, frame_id='base_link')
            part.target_joint_positions.append(float(value))
        if waist_q is not None:
            part = frame.retarget_infos.add(group_id=RetargetInfo.GROUP_WAIST,
                input_type=RetargetInfo.INPUT_VR, control_type=RetargetInfo.ABS_JOINT,
                frame_id='base_link')
            part.target_joint_positions.extend(map(float,waist_q))
        info = self.gdk.PubMsgInfo()
        info.source_ts_ns = stamp
        if not self._eef_head_publisher.publish(msg, info):
            raise RuntimeError(f'Tick {tick}: combined head qpos / dual-arm EEF command failed')

    def _execute_eef_path(self, actions, start_eef, start_joints, check=None, *,
                          hands=None, hand_actions=None, start_hands=None, max_frames=32,
                          startup_transition=False):
        check = check or (lambda: None)
        check()
        actions = np.asarray(actions, dtype=float)
        start_eef = np.asarray(start_eef, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 17 or not 1 <= len(actions) <= max_frames:
            raise ValueError(f'Expected 1..{max_frames} EEF actions with 17 dimensions')
        if start_eef.shape != (17,) or not np.isfinite(actions).all() or not np.isfinite(start_eef).all():
            raise ValueError('Non-finite or invalid EEF action/state')
        start_q = np.asarray([start_joints[n] for n in JOINT_NAMES], dtype=float)
        if not np.isfinite(start_q).all():
            raise ValueError('Non-finite initial joint state')
        current = np.asarray(read_positions(self.robot, JOINT_NAMES), dtype=float)
        if np.max(np.abs(current - start_q)) > .3:
            raise RuntimeError('Robot moved while inference was running; reacquire observation')
        path = np.concatenate([start_eef[None], actions], axis=0)
        # Head interpolation starts from the actual named joint feedback.
        path[0, 14:17] = start_q[:3]
        if np.max(np.abs(np.diff(path[:, 14:17], axis=0))) > .35:
            raise RuntimeError('Predicted head joint step exceeds 0.35 rad; no actions sent')
        # Model regression need not produce unit-length quaternions. Normalize
        # before SLERP/GDK; reject only unusable zero/non-finite norms here.
        # This changes representation, not the represented rotation or position.
        for offset in (3, 10):
            norms = np.linalg.norm(path[:, offset:offset+4], axis=1)
            invalid = ~np.isfinite(norms) | (norms == 0.)
            if np.any(invalid):
                import json
                index = int(np.flatnonzero(invalid)[0])
                detail = dict(side='left' if offset == 3 else 'right',
                              source='measured_start' if index == 0 else 'prediction',
                              frame_index=None if index == 0 else index-1,
                              quaternion_xyzw=path[index, offset:offset+4].tolist(),
                              norm=float(norms[index]), reason='zero_or_nonfinite_norm')
                error = ValueError('Invalid EEF quaternion: ' + json.dumps(detail))
                error.diagnostic = detail
                raise error
            path[:, offset:offset+4] /= norms[:, None]
        for frame in path:
            frame[14:17] = head_target(tuple(frame[14:17]))
        hand_path = (np.concatenate([start_hands[None], hand_actions], axis=0)
                     if hands is not None else None)
        startup_ticks = 0
        if startup_transition:
            # A one-off approach to the first model target, not a slower rollout.
            # Smoothstep has peak slope 1.5. Extend the three-second minimum
            # for larger moves: 0.1 m/s EEF translation, 0.3 rad/s rotation/qpos.
            duration = max(3., 1.5 * np.max(np.abs(path[1, 14:17]-path[0, 14:17])) / .3)
            for offset in (0, 7):
                distance = np.linalg.norm(path[1, offset:offset+3]-path[0, offset:offset+3])
                dot = abs(float(np.dot(path[0, offset+3:offset+7], path[1, offset+3:offset+7])))
                angle = 2 * np.arccos(np.clip(dot, 0., 1.))
                duration = max(duration, 1.5 * distance / .1, 1.5 * angle / .3)
            if hand_path is not None:
                duration = max(duration, 1.5 * np.max(np.abs(hand_path[1]-hand_path[0])) / .3)
            startup_ticks = int(np.ceil(duration * 100))
            print(f'STARTUP_TRANSITION: {startup_ticks / 100:.2f}s smooth entry; '
                  'then full chunk at 20Hz / 100Hz interpolation. Existing checks retained.', flush=True)
        if hands is not None and not hasattr(self, '_eef_head_publisher'):
            # Create the writer before starting the interpolation clock, reuse
            # it across chunks. The legacy 19D/17D paths never create this writer.
            self._eef_head_publisher = self.gdk.Publisher(
                '/gdk/model_predict', TrajectoryTrackingControl.DESCRIPTOR.full_name, self.gdk.QoS())
        started = time.monotonic()
        for tick in range(1, startup_ticks + len(actions)*5 + 1):
            check()
            delay = started + tick*.01 - time.monotonic()
            if delay < -.1:
                raise RuntimeError('Action execution is over 100 ms late')
            time.sleep(max(0., delay))
            check()
            if tick <= startup_ticks:
                index, fraction = -1, tick-1
                u = tick / startup_ticks
                alpha = u*u*(3-2*u)
                left_index, right_index = 0, 1
            else:
                index, fraction = divmod(tick-startup_ticks-1, 5)
                alpha = (fraction+1)/5
                # Already at frame 0 after entry: do not return to measured start.
                left_index = 1 if startup_ticks and index == 0 else index
                right_index = index+1
            a, b = path[left_index], path[right_index]
            target = a*(1-alpha) + b*alpha
            for offset in (3, 10):
                target[offset:offset+4] = _slerp_xyzw(a[offset:offset+4], b[offset:offset+4], alpha)
            if hands is not None:
                check()
                self._send_combined_eef_head(target, tick)
            else:
                command = self.gdk.EndEffectorPose()
                command.group = self.gdk.EndEffectorControlGroup.kBothArms
                command.life_time = .05
                for pose, offset in ((command.left_end_effector_pose, 0),
                                     (command.right_end_effector_pose, 7)):
                    pose.position.x, pose.position.y, pose.position.z = map(float, target[offset:offset+3])
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, target[offset+3:offset+7])
                check()
                if self.robot.move_head_joint_servo(target[14:17].tolist(), .02, False) != 0:
                    raise RuntimeError(f'Tick {tick}: head qpos command failed')
                check()
                if self.robot.end_effector_pose_control(command) != 0:
                    raise RuntimeError(f'Tick {tick}: dual-arm EEF command failed')
            if hands is not None:
                check()
                hand_q = hand_path[left_index]*(1-alpha) + hand_path[right_index]*alpha
                hands.send_qpos(hand_q[:22], hand_q[22:], frame_index=index,
                                subframe_index=fraction)

    def _execute_joint_path(self, actions, start_state, check, *, grippers):
        """Legacy checks/timing shared by 19D legacy and 17D head/arm targets."""
        motor_indices=MOTOR_INDICES if grippers else list(range(17))
        # 推理期间若实机关节被人移动，不用陈旧的观测作为插值起点。
        current = read_positions(self.robot, JOINT_NAMES)
        current_model_order = np.asarray(current[3:] + current[:3])
        if np.max(np.abs(current_model_order - start_state[motor_indices])) > .1:
            raise RuntimeError('Robot moved while inference was running; reacquire observation')
        path = np.concatenate([start_state[None], actions], axis=0)
        if np.max(np.abs(np.diff(path[:, motor_indices], axis=0))) > .35:
            raise RuntimeError('Predicted joint step exceeds 0.35 rad; no actions sent')
        started = time.monotonic()
        for tick in range(1, len(actions) * 5 + 1):
            check()
            delay = started + tick * .01 - time.monotonic()
            if delay < -.1:
                raise RuntimeError('Action execution is over 100 ms late')
            time.sleep(max(0.0, delay))
            check()
            index, fraction = divmod(tick - 1, 5)
            alpha = (fraction + 1) / 5
            q = path[index] * (1-alpha) + path[index+1] * alpha
            # GDK 不允许 move_ee_pos 与 servo 混用：19 维一次发送，含左右夹爪。
            motor_q = (tuple(q[16:19]) + tuple(q[:7]) + tuple(q[8:15]) if grippers
                       else tuple(q[14:17]) + tuple(q[:14]))
            # 保留模型输出的夹爪饱和处理：-0.785 张开，0 闭合。
            grips = tuple(np.clip(q[[7, 15]], -0.785, 0.0)) if grippers else None
            check()
            send_servo(self.robot, motor_q, grips)
