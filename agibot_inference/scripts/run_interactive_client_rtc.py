#!/usr/bin/env python3
"""常驻键盘推理：空格开始/暂停；S/F 标记结果并复位；Q 退出。

默认仅推理，--execute 才运动。启动及每次标记后恢复 clothes 首帧。
按键无需回车；长按会过滤重复键，请点按并松开约 0.7 秒。
空格停止继续发送目标，不是硬件急停；已经发送的短周期目标仍可能执行。
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
import uuid

import numpy as np
from agibot_tasks import canonical_task, FAMILIES
from replay import BODY_NAMES, JOINT_NAMES, GRIPPER_NAMES, bounded, load_episode, read_positions
from robot_wrapper_rtc import AgiBotRobot, DEFAULT_EPISODE
from run_ours_client import OursClient


class KeyAction(Exception):
    def __init__(self, key: str):
        self.key = key


class Keyboard:
    """主线程非阻塞读取终端，退出时恢复原终端设置。"""
    def __init__(self, fd: int):
        self.fd = fd
        self.last_key = None
        self.last_time = -math.inf
        self.stopping = False

    def __enter__(self):
        if not os.isatty(self.fd):
            raise RuntimeError('需要交互终端；远程运行请使用 ssh -t real')
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def check(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return
        raw = os.read(self.fd, 4096)
        if not raw:
            raise KeyAction('q')
        keys = [chr(c).lower() for c in raw if chr(c).lower() in ' sfq']
        if not keys:
            return
        # 同一批输入优先响应退出/停止，不缓存成下一次运动。
        key = 'q' if 'q' in keys else ' ' if ' ' in keys else keys[0]
        now = time.monotonic()
        repeated = key == self.last_key and now - self.last_time < .7
        self.last_key, self.last_time = key, now
        # 停止优先：即使刚按空格开始，立即再按也必须停；
        # 暂停后的长按重复键仍不能触发恢复运动。
        if not repeated or key == 'q' or (key == ' ' and self.stopping):
            raise KeyAction(key)


def wait_checked(seconds: float, check) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check()
        time.sleep(min(.02, max(0., end - time.monotonic())))
    check()


class Inference:
    """只有网络推理放后台。机器人始终由主线程控制，不会消费已暂停轮次的结果。"""
    def __init__(self, factory):
        self.factory = factory
        self.client = factory()
        self.info = dict(self.client.server_info)
        self.pending = None

    def wait_idle(self, check):
        if self.pending is not None:
            # 暂停前已提交的推理仍可完成，但它的结果一律丢弃。
            while not self.pending['done'].wait(.02):
                check()
            check()
            self.pending = None

    def infer(self, observation: dict, check) -> dict:
        if self.pending is not None:
            raise RuntimeError('Previous inference must be drained before reading a new observation')
        job = {'done': threading.Event()}
        self.pending = job
        def run():
            try:
                if self.client is None:
                    self.client = self.factory()
                job['result'] = self.client.policy_infer(observation)
            except Exception as exc:
                job['error'] = exc
                if self.client is not None:
                    self.client._disconnect()
                    self.client = None
            finally:
                job['done'].set()
        threading.Thread(target=run, daemon=True, name='policy-inference').start()
        while not job['done'].wait(.02):
            check()
        check()
        self.pending = None
        if 'error' in job:
            raise job['error']
        return job['result']

    def start_observe(self, observe) -> None:
        """RTC: 在发当前动作时读取新观测并请求下一块动作。"""
        if self.pending is not None:
            raise RuntimeError('RTC inference is already pending')
        job = {'done': threading.Event()}
        self.pending = job
        def run():
            try:
                observation = observe()
                job['observed_at'] = time.monotonic()
                job['result'] = self.client.policy_infer(observation)
            except Exception as exc:
                job['error'] = exc
                if self.client is not None:
                    self.client._disconnect()
                    self.client = None
            finally:
                job['done'].set()
        threading.Thread(target=run, daemon=True, name='rtc-observe-infer').start()

    def take_ready(self):
        job = self.pending
        if job is None or not job['done'].is_set():
            return None
        self.pending = None
        if 'error' in job:
            raise job['error']
        return job

    def close(self):
        # 退出时关闭 socket 解除等待；后台线程从不发送机器人动作。
        client = self.client
        sock = client.socket if client is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


def action_window(actions: np.ndarray, start_frame: int, steps: int) -> np.ndarray:
    """帧号从 1 计数；例如第 3 帧开始取 24 帧即输出索引 2..25。"""
    return np.asarray(actions)[start_frame - 1:start_frame - 1 + steps]


class RTC:
    """20Hz 动作块预取：按观测后的已过帧数跳过旧预测，并平滑接续。"""
    def __init__(self, robot, inference: Inference, steps: int, start_frame: int = 1):
        self.robot, self.inference, self.steps = robot, inference, steps
        self.start_index = start_frame - 1
        self.actions = None
        self.index = 0
        self.state = None

    def clear(self):
        self.actions = None
        self.index = 0
        self.state = None

    def step(self, check) -> dict:
        if self.actions is None:
            self.inference.wait_idle(check)
            observation = self.robot.get_observation()
            prediction = self.inference.infer(observation, check)
            self.actions = np.asarray(action_window(prediction['actions'],
                                                    self.start_index + 1, 32), dtype=np.float64)
            self.state = np.asarray(observation['state'], dtype=np.float64)
            self.index = 0
        count = min(self.steps, len(self.actions) - self.index)
        if count <= 0:
            self.clear()
            return {'frames': 0, 'switch': 'expired'}
        segment = self.actions[self.index:self.index + count]
        started = False
        trigger = max(1, min(8, count // 2))
        def on_frame(frame):
            nonlocal started
            if frame == trigger and self.inference.pending is None:
                self.inference.start_observe(self.robot.get_observation)
                started = True
        self.robot.execute(segment, self.state, check=check, on_frame=on_frame)
        self.state = segment[-1].copy()
        self.index += count
        job = self.inference.take_ready()
        if job is None and self.index == len(self.actions):
            # 旧块已用完时不再发送陈旧目标；等待下一块或重新采样。
            while self.inference.pending is not None and job is None:
                check()
                job = self.inference.take_ready()
                if job is None:
                    time.sleep(.02)
        switched = False
        if job is not None:
            age_frames = max(self.start_index, math.ceil((time.monotonic() - job['observed_at']) * 20))
            predicted = np.asarray(job['result']['actions'], dtype=np.float64)
            if predicted.shape != (32, 19) or not np.isfinite(predicted).all():
                raise ValueError('RTC prediction must be finite [32,19]')
            if age_frames >= len(predicted):
                self.clear()
            else:
                aligned = predicted[age_frames:].copy()
                # 最初 4 帧从旧计划的剩余目标渐变到新计划，防止接续跳变。
                for i in range(min(4, len(aligned))):
                    old_index = min(self.index + i, len(self.actions) - 1)
                    alpha = (i + 1) / min(4, len(aligned))
                    aligned[i] = (1 - alpha) * self.actions[old_index] + alpha * aligned[i]
                self.actions, self.index = aligned, 0
                switched = True
        elif self.index == len(self.actions):
            self.clear()
        return {'frames': count, 'switch': 'aligned' if switched else 'waiting',
                'prefetched': started}


def gripper_positions(robot) -> tuple:
    end = robot.get_end_state()
    values = []
    for side, name in zip(('left', 'right'), GRIPPER_NAMES):
        info = end[f'{side}_end_state']
        if info['names'] != [name] or len(info['end_states']) != 1:
            raise RuntimeError(f'{side} 夹爪反馈名称/数量异常')
        state = info['end_states'][0]
        if state.get('err_code', 0) or not state.get('enable', True):
            raise RuntimeError(f'{side} 夹爪故障或未使能')
        values.append(bounded(state['position'], -.785, 0., name))
    return tuple(values)


def ramp(robot: AgiBotRobot, names: tuple, start: tuple, target: tuple, speed: float, check) -> None:
    """100Hz 平滑复位；每次发送前检查键盘，普通通道保留 SDK 保护。"""
    distance = max(abs(a-b) for a,b in zip(start,target))
    steps = math.ceil(max(3., 1.5 * distance / speed) * 100)
    print(f'复位 {len(names)} 个关节，插值 {steps/100:.2f}s + 保持 1s', flush=True)
    started = time.monotonic()
    for tick in range(steps + 101):
        check()
        delay = started + tick * .01 - time.monotonic()
        if delay < -.1:
            raise RuntimeError('复位发送落后超过 100ms，已停止')
        time.sleep(max(0., delay))
        check()
        u = min(tick / steps, 1.)
        w = u*u*(3.-2.*u)
        req = robot.gdk.JointServoControlReq()
        req.control_period = .02
        req.joint_names = list(names)
        req.joint_positions = [a + w*(b-a) for a,b in zip(start,target)]
        if robot.robot.joint_servo_control(req, False) != 0:
            raise RuntimeError('复位伺服发送失败')


def reset_interruptible(robot: AgiBotRobot, pose: tuple, check) -> None:
    """保留先躯干、再头/双臂/夹爪的顺序；不调用无法响应按键的阻塞规划接口。"""
    body, motors, grips = pose
    check()
    start_body = read_positions(robot.robot, BODY_NAMES)
    # 先检查所有反馈，避免已开始运动后才发现夹爪故障。
    read_positions(robot.robot, JOINT_NAMES)
    gripper_positions(robot.robot)
    check()
    ramp(robot, BODY_NAMES, start_body, body, .15, check)
    actual = read_positions(robot.robot, BODY_NAMES)
    if max(abs(a-b) for a,b in zip(actual,body)) > .05:
        raise RuntimeError('躯干未到达首帧，停止复位')
    start = read_positions(robot.robot, JOINT_NAMES) + gripper_positions(robot.robot)
    # 首帧头部反馈的微小上界误差与 replay 使用同一适配规则。
    from replay import head_target
    start = head_target(start[:17]) + start[17:]
    ramp(robot, JOINT_NAMES + GRIPPER_NAMES, start, motors + grips, .3, check)
    actual = read_positions(robot.robot, JOINT_NAMES) + gripper_positions(robot.robot)
    if max(abs(a-b) for a,b in zip(actual,motors+grips)) > .05:
        raise RuntimeError('头部/手臂/夹爪未到首帧，停止复位')
    check()


class Results:
    def __init__(self, path: Path, metadata: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        # 每次启动独立日志，绝不覆盖之前记录；每条结果立刻 fsync。
        self.file = path.open('x', encoding='utf-8')
        self.metadata = metadata
        self.active = None

    def start(self):
        if self.active is None:
            self.active = {'trial_id': uuid.uuid4().hex, 'started_at': self.now(),
                           'chunks_completed': 0, '_start': time.monotonic()}

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def finish(self, outcome: str):
        if self.active is None:
            return False
        row = {**self.metadata, **self.active, 'outcome': outcome, 'ended_at': self.now()}
        row['elapsed_seconds'] = time.monotonic() - row.pop('_start')
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.active = None
        print(f'本条已保存：{outcome}', flush=True)
        return True

    def close(self):
        self.file.close()


def session(robot, inference: Inference, keyboard, results: Results, pose, execute: bool,
            steps: int, rtc: bool = False, start_frame: int = 1):
    state = 'resetting' if execute else 'ready'
    rtc_runner = RTC(robot, inference, steps, start_frame) if rtc and execute else None
    while True:
        keyboard.stopping = state in ('running', 'resetting')
        try:
            if state == 'resetting':
                reset_interruptible(robot, pose, keyboard.check)
                state = 'ready'
                print('复位完成，等待空格开始下一条。', flush=True)
            elif state == 'running':
                if rtc_runner is not None:
                    report = rtc_runner.step(keyboard.check)
                    if report['frames'] == 0:
                        continue
                    frames = report['frames']
                    timing = f'RTC {report["switch"]}'
                else:
                    inference.wait_idle(keyboard.check)
                    keyboard.check()
                    observation = robot.get_observation()
                    keyboard.check()
                    prediction = inference.infer(observation, keyboard.check)
                    keyboard.check()
                    actions = action_window(prediction['actions'], start_frame, steps)
                    if execute:
                        robot.execute(actions, observation['state'], check=keyboard.check)
                    else:
                        wait_checked(len(actions) / 20., keyboard.check)
                    frames = len(actions)
                    timing = f'推理 {prediction["inference_seconds"]:.3f}s'
                results.active['chunks_completed'] += 1
                print(f'本条第 {results.active["chunks_completed"]} 轮，{timing}，{frames} 帧' + ('' if execute else '（仅推理）'), flush=True)
            else:
                wait_checked(.02, keyboard.check)
        except KeyAction as action:
            if rtc_runner is not None:
                rtc_runner.clear()
            if action.key == 'q':
                return
            if action.key == ' ':
                if state == 'running':
                    state = 'paused'
                    print('已暂停发送动作；空格继续，S/F 标记本条。', flush=True)
                elif state == 'resetting':
                    state = 'reset_paused'
                    print('复位已暂停；空格继续复位。', flush=True)
                elif state == 'reset_paused':
                    state = 'resetting'
                else:
                    results.start()
                    state = 'running'
                    print('开始/继续本条：重新读取观测。', flush=True)
            elif action.key in ('s', 'f') and state in ('running', 'paused'):
                if results.finish('success' if action.key == 's' else 'failure'):
                    state = 'resetting' if execute else 'ready'
                    print('准备自动复位。' if execute else '等待空格开始下一条（仅推理）。', flush=True)
            # ready/resetting 中的 S/F 不生成重复结果，也不排队下一次复位。
        except (OSError, RuntimeError, ValueError) as exc:
            if rtc_runner is not None:
                rtc_runner.clear()
            print(f'已停止：{exc}', flush=True)
            state = 'reset_paused' if state == 'resetting' else 'paused'
            print('检查问题后按空格重试；Q 退出。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--task', type=canonical_task, default='guizi')
    parser.add_argument('--model', choices=FAMILIES, default='fastwam')
    parser.add_argument('--timeout', type=float, default=30.)
    parser.add_argument('--replan-steps', type=int, default=24)
    parser.add_argument('--start-frame', type=int, default=1,
                        help='每块 32 帧从第几帧开始执行（从 1 计数）；RTC 取该值与时间对齐帧中更晚者')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--rtc', action='store_true', help='动作执行期间预取下一块，按时间对齐并平滑接续；仅与 --execute 同用')
    parser.add_argument('--reset-episode', type=Path, default=DEFAULT_EPISODE)
    parser.add_argument('--log-dir', type=Path, default=Path('/data/agibot_inference/results'))
    args = parser.parse_args()
    parser.error('旧 19 维 checkpoint 含两只已拆除的夹爪；当前本体只支持离线 test，不能执行或读取 19 维实机观测')
    if not 1 <= args.replan_steps <= 32 or not 1 <= args.start_frame <= 32 or args.timeout <= 0:
        parser.error('replan-steps 和 start-frame 必须为 1..32，timeout 必须大于 0')
    if args.rtc and not args.execute:
        parser.error('--rtc 需要 --execute')
    if not sys.stdin.isatty():
        parser.error('需要交互终端；请在 real 的终端运行，或使用 ssh -t real')
    # 先完整验证复位轨迹和模型握手，再初始化机器人。
    frames, body = load_episode(args.reset_episode)
    pose = (body, frames[0][1], frames[0][2])
    runner = Inference(lambda: OursClient(args.ip, args.port, args.timeout, args.task, args.model))
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    log_path = args.log_dir / f'{args.task}_{args.model}_{session_id}.jsonl'
    results = None
    try:
        results = Results(log_path, {'session_id': session_id, 'task': args.task, 'model': args.model,
            'checkpoint': runner.info['checkpoint'], 'step': runner.info['step'], 'execute': args.execute,
            'replan_steps': args.replan_steps, 'start_frame': args.start_frame, 'rtc': args.rtc,
            'reset_episode': str(args.reset_episode.resolve())})
        robot = AgiBotRobot()
        print(f'结果文件：{log_path}\n空格 开始/暂停 | S 成功并复位 | F 失败并复位 | Q/Ctrl+C 退出', flush=True)
        print('实机模式：现在自动复位。' if args.execute else '仅推理模式：等待空格，不发送机器人动作。', flush=True)
        with Keyboard(sys.stdin.fileno()) as keyboard:
            session(robot, runner, keyboard, results, pose, args.execute,
                    args.replan_steps, args.rtc, args.start_frame)
    except (KeyboardInterrupt, SystemExit):
        print('\n已退出，停止发送动作。', flush=True)
    finally:
        try:
            if results is not None:
                try:
                    results.finish('aborted')
                finally:
                    results.close()
        finally:
            runner.close()


if __name__ == '__main__':
    # SSH 断开或进程收到终止信号时，同样走 finally 恢复终端、保存未完成记录。
    def stop_signal(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    main()
