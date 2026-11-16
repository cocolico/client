#!/usr/bin/env python3
"""读取一段采集数据，按原时间节奏回放头部、双臂和左右夹爪（单位：弧度）。

在 real 上先加载 SDK 和 Python 环境：
source ~/.cache/agibot/app/gdk/scripts/env.sh ~/.cache/agibot/app
conda activate gdk-py310

传入某一条 episode 的目录，不是 raw_data 总目录：
python replay.py /path/to/raw_data/clothes/EPISODE            # 只读取、检查数据
python replay.py /path/to/raw_data/clothes/EPISODE --execute  # 真正驱动机器人

先读取并检查整条轨迹，再用一个主循环：取一帧 → 等待时间 → 发送目标。
回放的是实际采集的 motor_position，不是模型预测动作，也不是末端笛卡尔位姿。
采集约定：每条轨迹都从右手 VR 手柄 A 键复位后开始，因此以本条首帧作为初始姿态。
启动顺序：躯干恢复到首帧 → 头部、双臂与夹爪平滑插值到首帧 → 按原时间节奏回放。
躯干只在启动时恢复；头部、双臂、夹爪合并为一条约 100 Hz 的联合伺服指令。
夹爪文件按时间戳对齐，允许录制末尾缺少约 1 秒，缺少部分保持最后目标。
"""

import argparse
import math
import struct
import time
from bisect import bisect_left
from operator import itemgetter
from pathlib import Path

from genie_msgs_pb.msg.JointState_pb2 import JointState
from genie_msgs_pb.msg.EndState_pb2 import EndState

# SDK 要求的目标顺序：头部 3 个 → 左臂 7 个 → 右臂 7 个。
# 按名字取角度，避免录制文件中的关节排列顺序不同导致控制错位。
JOINT_NAMES = (
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7",
)
# itemgetter 创建一个取值函数：传入 {关节名: 角度}，返回按上面顺序排列的 17 个角度。
SELECT_JOINTS = itemgetter(*JOINT_NAMES)


BODY_NAMES = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))

# 当前实机 GDK 日志：head_joint3 的 0.523580 可通过，0.523676 被拒绝。
# 使用已验证范围内的保守上界；仅收敛 <= 0.001 rad 的录制反馈边界误差。
# 这是回放目标适配，不修改机器人限位，也不绕过 GDK 校验。
HEAD3_MAX = 0.5235
BOUNDARY_TOLERANCE = 0.001
GRIPPER_NAMES = ("idx31_gripper_l_inner_joint1", "idx71_gripper_r_inner_joint1")


def bounded(value: float, low: float, high: float, label: str) -> float:
    """微小边界超出收敛到合法目标；大幅越界、NaN、Inf 直接拒绝。"""
    if not math.isfinite(value) or value < low - BOUNDARY_TOLERANCE or value > high + BOUNDARY_TOLERANCE:
        raise ValueError(f"{label}: {value} 超出允许目标范围 [{low}, {high}]")
    return min(high, max(low, value))


def head_target(q: tuple[float, ...]) -> tuple[float, ...]:
    # 这里只处理本次确认的 head_joint3 上边界；其余限位仍由 GDK 检查。
    return q[:2] + (bounded(q[2], -math.inf, HEAD3_MAX, JOINT_NAMES[2]),) + q[3:]


def read_pbdat(path: Path, message_type):
    """PBDAT: 8 字节总帧数，每帧 40 字节头（时间戳等）+ protobuf。"""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: 缺少文件头")
        total, = struct.unpack("<q", header)
        if total <= 0 or total > (file_size - 8) // 40:
            raise ValueError(f"{path}: 无效帧数 {total}")
        previous = None
        for i in range(total):
            header = f.read(40)
            if len(header) != 40:
                raise ValueError(f"{path}: 第 {i} 帧头部不完整")
            timestamp, _, _, _, size = struct.unpack("<5q", header)
            if size <= 0 or size > min(1024 * 1024, file_size - f.tell()):
                raise ValueError(f"{path}: 第 {i} 帧长度无效")
            if previous is not None and timestamp <= previous:
                raise ValueError(f"{path}: 第 {i} 帧时间戳未递增")
            previous = timestamp
            yield timestamp, message_type.FromString(f.read(size))
        if f.read(1):
            raise ValueError(f"{path}: 声明帧数之后仍有数据")


def load_episode(episode: Path):
    """在任何运动之前检查完整文件，生成 (时间戳, 头/臂17维, 左右夹爪2维)。"""
    record = episode / "record"
    frames, body = [], None
    clipped_head = 0
    for i, (timestamp, msg) in enumerate(read_pbdat(record / "-hal-joint_state.pbdat", JointState)):
        if len(msg.name) != len(msg.motor_position) or len(set(msg.name)) != len(msg.name):
            raise ValueError(f"关节第 {i} 帧: 名称/角度数量不匹配或名称重复")
        by_name = dict(zip(msg.name, msg.motor_position))
        q = SELECT_JOINTS(by_name)
        if not all(map(math.isfinite, q)):
            raise ValueError(f"关节第 {i} 帧: 无效角度")
        try:
            target = head_target(q)
        except ValueError as exc:
            raise ValueError(f"关节第 {i} 帧: {exc}") from exc
        clipped_head += target != q
        frames.append((timestamp, target))
        if body is None:
            body = tuple(by_name[n] for n in BODY_NAMES)
            if not all(map(math.isfinite, body)):
                raise ValueError("躯干首帧角度无效")
    aligned = []
    for side, name in zip(("left", "right"), GRIPPER_NAMES):
        samples = []
        clipped_grip = 0
        for timestamp, msg in read_pbdat(record / f"-hal-{side}_ee_data.pbdat", EndState):
            if list(msg.name) != [name] or len(msg.end_state) != 1 or msg.type != 1:
                raise ValueError(f"{side}: 夹爪名称、类型或关节数不符合 omnipicker")
            value = msg.end_state[0].position
            target = bounded(value, -0.785, 0.0, f"{side} gripper")
            clipped_grip += target != value
            samples.append((timestamp, target))
        times = [t for t, _ in samples]
        tail = max(0, frames[-1][0] - times[-1]) / 1e9
        if tail > 1.05:
            raise ValueError(f"{side}: 夹爪末尾缺失 {tail:.3f}s，超过允许的 1.05s")
        values = []
        for timestamp, _ in frames:
            index = bisect_left(times, timestamp)
            candidates = {max(0, index - 1), min(index, len(times) - 1)}
            nearest = min(candidates, key=lambda j: (abs(times[j] - timestamp), j))
            # 末尾允许保持最后目标；首部/中间找不到 100ms 内反馈则报错。
            if timestamp <= times[-1] and abs(times[nearest] - timestamp) > 100_000_000:
                raise ValueError(f"{side}: 时间戳 {timestamp} 附近 100ms 内无夹爪数据")
            values.append(samples[nearest][1])
        aligned.append(values)
        print(f"{side} 夹爪: {len(samples)} 帧，范围 [{min(values):.6f}, {max(values):.6f}] rad，"
              f"末尾保持 {tail:.3f}s，微小边界修正 {clipped_grip} 帧", flush=True)
    print(f"head_joint3 上界 {HEAD3_MAX} rad，微小边界修正 {clipped_head} 帧", flush=True)
    return [(t, q, (left, right)) for (t, q), left, right in zip(frames, *aligned)], body


def wait_grippers(robot, targets: tuple[float, float], timeout: float = 2.0) -> None:
    """只读等待夹爪到位，不重发命令；实测全程张开约需 0.75 秒。"""
    deadline = time.monotonic() + timeout
    while True:
        end = robot.get_end_state()
        errors = []
        for side, target in zip(("left", "right"), targets):
            state = end[f"{side}_end_state"]["end_states"][0]
            actual = state["position"]
            if not math.isfinite(actual) or state.get("err_code", 0) != 0 or state.get("enable", True) is False:
                raise RuntimeError(f"{side} 夹爪反馈无效、未使能或存在故障: {state}")
            errors.append(abs(actual - target))
        if max(errors) <= 0.05:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"夹爪 {timeout:.1f}s 内未到位，左右误差 {errors} rad；停止")
        time.sleep(0.05)


def read_positions(robot, names: tuple[str, ...]) -> tuple[float, ...]:
    """按指定名字顺序读取实机关节角度。"""
    states = robot.get_joint_states()["states"]
    by_name = {s["name"]: s["motor_position"] for s in states}
    positions = tuple(by_name[name] for name in names)
    if not all(map(math.isfinite, positions)):
        raise RuntimeError("机器人反馈中含无效角度")
    return positions


def send_servo(robot, q: tuple[float, ...], grippers: tuple[float, float] | None = None) -> None:
    """一条普通通道 servo 同时发送头/双臂/夹爪，不能混用 move_ee_pos。"""
    import agibot_gdk as gdk
    if len(q) != 17 or not all(map(math.isfinite, q)):
        raise ValueError("头部/双臂目标必须为有限的 17 维角度")
    q = head_target(tuple(q))
    names = JOINT_NAMES
    if grippers is not None:
        if len(grippers) != 2:
            raise ValueError("夹爪目标必须为左右 2 维")
        q += tuple(bounded(float(v), -0.785, 0.0, n) for n, v in zip(GRIPPER_NAMES, grippers))
        names += GRIPPER_NAMES
    req = gdk.JointServoControlReq()
    req.control_period = 0.02
    req.joint_names, req.joint_positions = list(names), list(q)
    if robot.joint_servo_control(req, False) != 0:
        raise RuntimeError("头部/双臂/夹爪联合伺服指令发送失败")


def move_to_first_frame(robot, target: tuple[float, ...], grippers: tuple[float, float] | None = None) -> None:
    """100 Hz 平滑插值到首帧；可把两个夹爪一起纳入同一伺服请求。"""
    if len(target) != 17 or not all(map(math.isfinite, target)):
        raise ValueError("首帧头部/双臂目标必须为有限的 17 维角度")
    if grippers is not None and len(grippers) != 2:
        raise ValueError("首帧夹爪目标必须为左右 2 维")
    target = head_target(tuple(target))
    start = read_positions(robot, JOINT_NAMES)
    if grippers is not None:
        target += tuple(bounded(v, -0.785, 0.0, n) for n, v in zip(GRIPPER_NAMES, grippers))
        end = robot.get_end_state()
        for side, name in zip(("left", "right"), GRIPPER_NAMES):
            info = end[f"{side}_end_state"]
            if info['names'] != [name] or len(info['end_states']) != 1:
                raise ValueError(f"{side} 夹爪反馈名称/数量不匹配")
            state = info['end_states'][0]
            if not state.get('enable', True) or state.get('err_code', 0):
                raise RuntimeError(f"{side} 夹爪未使能或存在故障")
            start += (bounded(state['position'], -0.785, 0.0, name),)
    distance = max(abs(a - b) for a, b in zip(start, target))
    # smoothstep 最大斜率为 1.5；含夹爪的所有目标角速度均不超过 0.3 rad/s。
    steps = math.ceil(max(3.0, 1.5 * distance / 0.3) * 100)
    print(f"插值到录制首帧：最大变化 {distance:.3f} rad，耗时 {steps / 100:.2f} 秒", flush=True)
    started = time.monotonic()
    # 插值结束后继续发送首帧目标 1 秒，让实际关节跟上。
    for step in range(steps + 101):
        delay = started + step * 0.01 - time.monotonic()
        if delay < -0.1:
            raise RuntimeError("首帧插值落后超过 100 ms，停止发送")
        time.sleep(max(0.0, delay))
        u = min(step / steps, 1.0)
        weight = u * u * (3.0 - 2.0 * u)  # 起点、终点速度均为 0。
        q = tuple(a + weight * (b - a) for a, b in zip(start, target))
        send_servo(robot, q[:17], q[17:] if grippers is not None else None)
    actual = read_positions(robot, JOINT_NAMES)
    error = max(abs(a - b) for a, b in zip(actual, target[:17]))
    if error > 0.05:
        raise RuntimeError(f"首帧未到位，最大误差 {error:.3f} rad；不开始回放")
    if grippers is not None:
        wait_grippers(robot, target[17:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path, help="包含 record/ 的 episode 目录")
    parser.add_argument("--execute", action="store_true", help="真正驱动机器人；默认只检查数据")
    args = parser.parse_args()
    if args.execute:
        parser.error('当前本体没有旧夹爪；历史轨迹只能离线校验，不能实机 replay')

    # 1. 所有文件先读完、检查和对齐。任何数据错误都发生在机器人运动之前。
    frames, body_q = load_episode(args.episode)
    first_ts, first_q, first_grips = frames[0]
    duration = (frames[-1][0] - first_ts) / 1e9
    print(f"{args.episode}: {len(frames)} frames; head={first_q[:3]}; "
          f"arms={first_q[3:]}; grippers={first_grips}", flush=True)
    if not args.execute:
        print(f"Validated {len(frames)} frames, {duration:.3f} seconds; 未连接机器人")
        return

    # 2. 恢复躯干、头部与双臂、夹爪的首帧姿态；全部到位后再开始计时。
    import agibot_gdk as gdk
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    robot = gdk.Robot()
    time.sleep(1)
    print("恢复首帧的躯干姿态（数采 A 键初始姿态）", flush=True)
    current = read_positions(robot, BODY_NAMES)
    distance = max(abs(a - b) for a, b in zip(current, body_q))
    request = gdk.JointControlReq()
    request.joint_names, request.joint_positions = list(BODY_NAMES), list(body_q)
    request.joint_velocities = [0.15] * len(BODY_NAMES)
    request.life_time = max(10.0, 2.0 * distance / 0.15 + 5.0)
    if robot.joint_control_request(request) != 0:
        raise RuntimeError("躯干初始姿态复位请求失败")
    actual = read_positions(robot, BODY_NAMES)
    error = max(abs(a - b) for a, b in zip(actual, body_q))
    if error > 0.05:
        raise RuntimeError(f"躯干未到位，误差 {error:.3f} rad；不开始手臂运动")
    move_to_first_frame(robot, first_q, first_grips)
    print("初始姿态已到位，开始 replay（头部、双臂、左右夹爪）", flush=True)
    started = time.monotonic()

    # 3. 唯一回放主循环：按记录时间一次发送头部、双臂和左右夹爪的 19 维目标。
    for i, (timestamp, q, grips) in enumerate(frames):
        delay = started + (timestamp - first_ts) / 1e9 - time.monotonic()
        if delay < -0.1:
            raise RuntimeError(f"Frame {i}: replay 落后超过 100ms，停止发送")
        time.sleep(max(0.0, delay))
        try:
            send_servo(robot, q, grips)  # 普通通道；不与独立夹爪接口混用。
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Frame {i}, t={(timestamp-first_ts)/1e9:.3f}s, "
                               f"head={q[:3]}, grippers={grips}: {exc}") from exc
    print(f"Replayed {len(frames)} frames, {duration:.3f} seconds; "
          f"发送 {len(frames)} 条头/双臂/夹爪联合伺服目标（另有首帧复位）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C 后退出循环，不再发送新目标；这里没有发送回零或断使能指令。
        print("Replay interrupted; stopped sending commands.")
