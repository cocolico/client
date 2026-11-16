#!/usr/bin/env python3
"""离线：python scripts/run_ours_client.py --test --task yifu --model pi05
旧19维任务仍可使用；yifu/晾衣服为三相机61维50帧模型。
真机：source env.sh 后 python scripts/run_ours_client.py --task yifu --model pi05 --execute
只有 --execute 才会归位、连接/使能双手和执行动作；--test 不启动真机。
"""
import argparse
from pathlib import Path
import numpy as np

from _base_client import BaseClient
from _ours_protocol import send_message, recv_message
from agibot_tasks import canonical_task, default_episode, FAMILIES, interface_spec
from inference_connection import inference_connection


def startup_reset(pico_root: Path, robot_ssh: str) -> None:
    """The existing right-A G2 reset, not the legacy gripper/recording reset."""
    import json
    import sys
    import threading
    import time
    sys.path.insert(0, str(pico_root / 'src'))
    from sharpa_pose_studio.common.paths import INPUT_CONFIG_PATH
    from sharpa_pose_studio.teleop.session import RealOps
    config = json.loads(INPUT_CONFIG_PATH.read_text())
    config['robot'] = robot_ssh
    print('推理启动归位：调用右 A 同一归位方法，目标 G2零点；确认到位后等待1秒。', flush=True)
    RealOps(config, INPUT_CONFIG_PATH).reset_robot(threading.Event(), connected=False)
    time.sleep(1.)
    print('INFERENCE_RESET_READY：已确认归位，接下来获取实际状态。', flush=True)


class OursClient(BaseClient):
    def __init__(self, ip: str, port: int, timeout: float = 300.0,
                 task: str = 'clothes', model: str = 'ours') -> None:
        self.task, self.model = canonical_task(task), model
        self.spec = interface_spec(self.task, self.model)
        super().__init__(ip, port)
        self.socket.settimeout(timeout)  # 模型推理可能超过 BaseClient 的 5 秒连接超时。
        # 在读取实机和复位之前，确认 server 的任务、模型和关节顺序。
        try:
            send_message(self.socket, {'cmd': 'info'})
            self.server_info = recv_message(self.socket)
            expected = self.spec
            for key, value in expected.items():
                if self.server_info.get(key) != value:
                    raise ValueError(f'Server {key}={self.server_info.get(key)!r}, expected {value!r}; restart server with matching --task/--model')
        except Exception:
            self._disconnect()
            raise
        print(f'已连接 {self.model}/{self.task}，checkpoint真实步数={self.server_info["step"]}', flush=True)


    def policy_infer(self, observation: dict) -> dict:
        """按握手契约发送观测，返回完整动作块；61维模式不切片或补零。"""
        send_message(self.socket, {'cmd': 'infer', 'instruction': self.task, 'model': self.model, 'observation': observation})
        response = recv_message(self.socket)
        if 'error' in response:
            raise RuntimeError(response['error'])
        actions = np.asarray(response['actions'], dtype=np.float32)
        if actions.shape != (self.spec['horizon'], self.spec['action_dim']) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid model actions: {actions.shape}')
        return {'actions': actions, 'inference_seconds': response['inference_seconds']}


def test(client: OursClient, episode: Path | None = None, frame: int = 0) -> dict:
    """离线联调：已有轨迹 → 图像/状态 → server 推理；不创建机器人对象。"""
    episode = episode or default_episode(client.task)
    if client.task == 'yifu':
        from recorded_sharpa import recorded_observation61
        observation = recorded_observation61(episode, frame)
    else:
        from robot_wrapper import recorded_observation
        observation = recorded_observation(episode, frame)
    result = client.policy_infer(observation)
    image_shapes = {k: v.shape for k, v in observation.get('images', {}).items()} if client.task == 'yifu' else observation['active_rgb'].shape
    print(f'离线测试通过：RGB={image_shapes}, state={observation["state"].shape}, '
          f'actions={result["actions"].shape}, inference={result["inference_seconds"]:.3f}s')
    print('第一个预测动作：', result['actions'][0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--inference', choices=('local', 'remote'), default='local',
                        help='local: 原有 TCP 地址；remote: 持续 SSH 转发到远端 server')
    parser.add_argument('--remote-config', type=Path,
                        help='远端连接 JSON；remote 默认使用 configs/inference_b_dev.json')
    parser.add_argument('--task', type=canonical_task, default='clothes', help='yifu/晾衣服:61维；clothes/guizi/shuiping/dengpao:旧19维')
    parser.add_argument('--model', choices=FAMILIES, default='ours')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--execute', action='store_true', help='自动复位并执行模型动作')
    parser.add_argument('--replan-steps', type=int, help='61维执行固定50帧；旧19维默认24')
    parser.add_argument('--test', action='store_true', help='只做已有轨迹的离线推理')
    parser.add_argument('--episode', type=Path, help='覆盖当前任务的离线轨迹；不用于61维真机归位')
    parser.add_argument('--reset-episode', type=Path, help='旧版兼容参数；61维真机不接受，固定使用 G2零点')
    parser.add_argument('--frame', type=int, default=0, help='离线录制数据的帧序号')
    parser.add_argument('--iterations', type=int, default=0, help='真机0表示持续循环；离线0表示测试1帧')
    parser.add_argument('--output', type=Path, help='离线保存预测动作和服务信息；绝不执行动作')
    parser.add_argument('--hands-config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json')
    parser.add_argument('--pico-root', type=Path, default=Path('/home/descfly/code/egoav/pico'), help='右 A / G2零点归位代码所在仓库')
    parser.add_argument('--robot-ssh', default='agi', help='现有 G2 归位所用 SSH 主机别名')
    args = parser.parse_args()
    if not args.test and not (args.execute and args.task == 'yifu' and args.model == 'pi05'):
        parser.error('真机仅支持 --task yifu --model pi05 --execute；旧19维仅支持 --test')
    if args.replan_steps is None:
        args.replan_steps = 50 if args.task == 'yifu' else 24
    if not 1 <= args.replan_steps <= interface_spec(args.task, args.model)['horizon'] or args.iterations < 0 or args.timeout <= 0:
        parser.error('Invalid steps/iterations/timeout')
    if args.test and args.execute:
        parser.error('--test 不能与 --execute 同时使用')
    if not args.test and args.replan_steps != 50:
        parser.error('61维真机每次完整执行50帧，不能截断动作块')
    if not args.test and (args.episode or args.reset_episode or args.output):
        parser.error('--episode/--reset-episode/--output 为离线参数；61维真机启动使用 G2零点')
    with inference_connection(args) as (ip, port):
        _run_client(args, ip, port)


def _run_client(args, ip, port):
    """The existing client/robot loop, with only the connection address selected."""
    client = OursClient(ip, port, args.timeout, args.task, args.model)
    hands = None
    try:
        if args.test:
            import json
            results = [test(client, args.episode, args.frame + i) for i in range(max(1, args.iterations))]
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.output, actions=np.stack([r['actions'] for r in results]),
                                    inference_seconds=[r['inference_seconds'] for r in results],
                                    server_info=json.dumps(client.server_info, ensure_ascii=False),
                                    episode=str(args.episode or default_episode(client.task)), frame=args.frame)
            return
        from robot_wrapper import AgiBotRobot
        from sharpa_hands import SharpaHands
        from hand_startup import load_hand_reset, reset_hands_to_recording
        from replay_full_eef import assert_ready
        hand_reset = load_hand_reset(args.task)
        # Server handshake has already passed. Reset must succeed before enable.
        startup_reset(args.pico_root, args.robot_ssh)
        robot = AgiBotRobot(with_eef=True)
        hands = SharpaHands.from_config(args.hands_config)
        hands.connect()
        reset_hands_to_recording(hands, hand_reset, check=lambda:assert_ready(robot.robot))
        count = 0
        # Sequential loop: measured observation -> all 50 frames -> new observation.
        while True:
            observation = robot.get_observation61(hands)
            result = client.policy_infer(observation)
            actions = result['actions']
            print(f'第{count+1}轮：推理 {result["inference_seconds"]:.3f}s，执行完整 {len(actions)} 帧（20Hz → 100Hz插值）', flush=True)
            robot.execute_sharpa61(actions, observation['state'], observation['joints'], hands,
                                   startup_transition=(count == 0))
            count += 1
            if args.iterations and count >= args.iterations:
                break
    finally:
        try:
            if hands is not None:
                hands.close()
        finally:
            client._disconnect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已停止；不再发送新动作。')
