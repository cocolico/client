#!/usr/bin/env python3
"""AgiBot 底盘定距遥控：点按 W/S/A/D，各移动 5 cm，方向相对机器人自身。

python scripts/teleop_chassis.py                         # 试按键，不连接机器人
python scripts/teleop_chassis.py --execute               # 每次目标 5 cm
python scripts/teleop_chassis.py --execute --step-cm 10   # 每次目标 10 cm

通过 Pnc.relative_move 提交位移目标，等待任务成功；不通过速度乘时间估算距离。
移动中按键不排队。空格取消本次运动，Q/Ctrl+C 取消并退出。
终端无松键事件：移动结束后须松键至少 0.7 秒再点按，过滤长按重复输入。
"""
import argparse
import math
import os
import select
import signal
import sys
import termios
import time
import tty

DIRECTIONS = {'w': (1., 0.), 's': (-1., 0.), 'a': (0., 1.), 'd': (0., -1.)}
# GDK PNCTaskState: 0 idle; 1..6 starting/running/pausing/paused/resuming/canceling;
# 7 canceled; 8 fail; 9 success。
ACTIVE = set(range(1, 7))
RELEASE_GAP = .7


def make_request(gdk, key: str, step: float):
    """机器人坐标系 +x 前、+y 左；单位米；单位四元数表示不旋转。"""
    x, y = DIRECTIONS[key]
    req = gdk.NaviReq()
    req.timestamp_ns = time.time_ns()
    req.target.position.x, req.target.position.y, req.target.position.z = x * step, y * step, 0.
    req.target.orientation.x = req.target.orientation.y = req.target.orientation.z = 0.
    req.target.orientation.w = 1.
    return req


class StepMover:
    """一次只跟踪一个相对移动任务，禁止按键积压成连续运动。"""
    def __init__(self, pnc, gdk, step: float, timeout: float):
        self.pnc, self.gdk, self.step, self.timeout = pnc, gdk, step, timeout
        self.started = None
        self.previous_id = self.task_id = None

    @property
    def busy(self) -> bool:
        return self.started is not None

    def start(self, key: str) -> None:
        if self.busy:
            return
        state = self.pnc.get_task_state()
        if state.state not in (0, 7, 8, 9):
            raise RuntimeError(f'底盘已有任务: id={state.id}, state={state.state}；请先结束该任务')
        self.previous_id, self.task_id = state.id, None
        # 先记录 pending，接口抛异常或 Ctrl+C 时也会尝试取消可能已接收的任务。
        self.started = time.monotonic()
        self.pnc.relative_move(make_request(self.gdk, key, self.step))
        print(f'已提交 {key.upper()}：目标 {self.step * 100:g} cm，等待任务完成', flush=True)

    def poll(self) -> bool:
        """返回 True 表示本步已结束（成功、取消或失败），不代表一定成功。"""
        if not self.busy:
            return False
        state = self.pnc.get_task_state()
        # 忽略提交前缓存的 success，必须看到新任务 ID 或新的执行状态。
        if self.task_id is None and (state.id != self.previous_id or state.state in ACTIVE):
            self.task_id = state.id
        if self.task_id is not None and state.id != self.task_id:
            raise RuntimeError('底盘任务被其他控制程序替换，停止遥控')
        if self.task_id is not None:
            if state.state == 9:
                self.started = None
                print('本步任务成功；松键后再点按下一步', flush=True)
                return True
            if state.state in (7, 8):
                # 规划器已结束该任务：报告失败并留在遥控界面，不自动重试。
                self.started = None
                print(f'本步未完成: id={state.id}, state={state.state}, message={state.message}', flush=True)
                if 'collision' in state.message.lower():
                    print('底盘检测到碰撞风险，等待超时。请检查周围空间后再点按；不会自动重试。', flush=True)
                else:
                    print('本步已结束/取消；确认原因后可点按下一步。', flush=True)
                return True
        if time.monotonic() - self.started > self.timeout:
            raise RuntimeError(f'本步超过 {self.timeout:g}s 仍未确认成功')
        return False

    def cancel(self) -> None:
        if not self.busy:
            return
        state = self.pnc.get_task_state()
        if self.task_id is None and (state.id != self.previous_id or state.state in ACTIVE):
            self.task_id = state.id
        if self.task_id is None:
            raise RuntimeError('未取得本次任务 ID，无法确认取消；请在机器人端确认底盘已停止')
        if state.id != self.task_id:
            # 不能擅自取消另一个程序后来提交的任务。
            raise RuntimeError('当前已是其他底盘任务；未取消该任务，请现场确认')
        if state.state not in (0, 7, 8, 9):
            self.pnc.cancel_task(self.task_id)
            deadline = time.monotonic() + 2.
            while True:
                state = self.pnc.get_task_state()
                if state.id != self.task_id:
                    raise RuntimeError('取消期间底盘任务被替换，请现场确认停止')
                if state.state in (0, 7, 8, 9):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('已发取消请求，但 2 秒内未确认停止')
                time.sleep(.05)
        self.started = None
        print('本次任务已结束/取消', flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true', help='真正执行位移；默认仅测试按键')
    parser.add_argument('--step-cm', type=float, default=5., help='每次位移目标，默认 5 cm，范围 (0, 30]')
    parser.add_argument('--timeout', type=float, default=15., help='每步等待秒数，默认 15')
    args = parser.parse_args()
    if not math.isfinite(args.step_cm) or not 0 < args.step_cm <= 30:
        parser.error('--step-cm 必须在 (0, 30] 内')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 60:
        parser.error('--timeout 必须在 (0, 60] 内')
    if not sys.stdin.isatty():
        parser.error('请在交互终端运行；SSH 直接启动须加 -t')
    fd = sys.stdin.fileno()
    old_terminal = termios.tcgetattr(fd)
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGHUP, signal.SIGTERM)}
    mover = None
    try:
        if args.execute:
            import agibot_gdk as gdk
            if gdk.gdk_init() != gdk.GDKRes.kSuccess:
                raise RuntimeError('GDK 初始化失败')
            pnc = gdk.Pnc()
            time.sleep(2)
            state = pnc.get_task_state()
            if state.state not in (0, 7, 8, 9):
                raise RuntimeError(f'底盘已有任务 id={state.id}, state={state.state}；请先结束它再运行')
            # 相对移动官方示例直接调用 relative_move，无须切换速度遥控模式。
            mover = StepMover(pnc, gdk, args.step_cm / 100., args.timeout)
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        print(f"{'实机' if args.execute else '试运行'}：每次 {args.step_cm:g} cm | "
              'W 前 / S 后 / A 左移 / D 右移 | 空格取消 | Q/Ctrl+C 退出', flush=True)
        last_input, armed = -math.inf, True
        while True:
            if mover is not None and mover.poll():
                # 完成后重新等待松键，防止缓冲区中的旧按键触发下一步。
                last_input, armed = time.monotonic(), False
            busy = mover is not None and mover.busy
            if not busy and time.monotonic() - last_input >= RELEASE_GAP:
                armed = True
            if select.select([fd], [], [], .05)[0]:
                data = os.read(fd, 4096)
                if not data:
                    break
                text = data.decode('ascii', errors='replace').lower()
                last_input = time.monotonic()
                if 'q' in text:
                    break
                if ' ' in text:
                    armed = False
                    if mover is not None:
                        mover.cancel()
                    continue
                # 一批重复字符只触发一步；运动中和松键间隔内的输入直接丢弃。
                if armed and not busy and text and all(c in DIRECTIONS for c in text):
                    armed = False
                    key = text[0]
                    if mover is None:
                        print(f'试运行 {key.upper()}：目标 {args.step_cm:g} cm（未发指令）', flush=True)
                    else:
                        mover.start(key)
    except KeyboardInterrupt:
        print('\n退出遥控', flush=True)
    finally:
        try:
            if mover is not None:
                mover.cancel()
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, old_terminal)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    main()
