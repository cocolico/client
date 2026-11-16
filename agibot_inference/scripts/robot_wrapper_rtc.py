"""AgiBot 观测、初始化和动作执行。主客户端只需调用这三个接口。"""
from pathlib import Path
import struct
import socket
import time

import cv2
import numpy as np

from replay import BODY_NAMES, JOINT_NAMES, SELECT_JOINTS, move_to_first_frame, read_positions, send_servo

DEFAULT_EPISODE = Path(__file__).resolve().parents[1] / 'data/raw_data/clothes/0912_20260912112854'
# 模型顺序：左臂7、左夹爪、右臂7、右夹爪、头部3。
from agibot_tasks import MODEL_JOINT_NAMES
MOTOR_INDICES = list(range(7)) + list(range(8, 15)) + [16, 17, 18]


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
    def __init__(self) -> None:
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
        self.last_image_timestamp = None
        time.sleep(2)

    def get_observation(self) -> dict:
        """读取头部彩色相机 + 实测19维状态，图像统一转为 RGB 640x480。"""
        gdk = self.gdk
        image = self.camera.get_latest_image(gdk.CameraType.kHeadColor, 1000.0)
        if image is None or image.timestamp_ns == self.last_image_timestamp:
            raise RuntimeError('No fresh head RGB image')
        if not -0.1 <= (time.time_ns() - image.timestamp_ns) / 1e9 <= 1.0:
            raise RuntimeError('Head image is stale or camera/system clocks differ')
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

    def execute(self, actions: np.ndarray, start_state: np.ndarray, check=None,
                on_frame=None) -> None:
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
        # 推理期间若实机关节被人移动，不用陈旧的观测作为插值起点。
        current = read_positions(self.robot, JOINT_NAMES)
        current_model_order = np.asarray(current[3:] + current[:3])
        if np.max(np.abs(current_model_order - start_state[MOTOR_INDICES])) > .1:
            raise RuntimeError('Robot moved while inference was running; reacquire observation')
        path = np.concatenate([start_state[None], actions], axis=0)
        if np.max(np.abs(np.diff(path[:, MOTOR_INDICES], axis=0))) > .35:
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
            motor_q = tuple(q[16:19]) + tuple(q[:7]) + tuple(q[8:15])
            # 保留模型输出的夹爪饱和处理：-0.785 张开，0 闭合。
            grips = np.clip(q[[7, 15]], -0.785, 0.0)
            check()
            send_servo(self.robot, motor_q, tuple(grips))
            if on_frame is not None and tick % 5 == 0:
                on_frame(tick // 5)
