#!/usr/bin/env python3
"""Start the local policy server when the interactive client needs it."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/agibot'))
from _ours_protocol import recv_message, send_message


def info():
    try:
        with socket.create_connection(('127.0.0.1', 19000), timeout=5) as sock:
            sock.settimeout(5)
            send_message(sock, {'cmd': 'info'})
            return recv_message(sock)
    except ConnectionRefusedError:
        return None
    except (TimeoutError, socket.timeout, OSError) as exc:
        raise RuntimeError(
            '127.0.0.1:19000 已有推理服务但没有及时响应 info；'
            '如果要切换任务或模型，请先运行 bash server/server.sh restart '
            '--task 任务名 --model 模型名'
        ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    current = info()
    if current is None:
        print(f'启动本地推理服务：{args.model}/{args.task}', flush=True)
        subprocess.run(['bash', str(ROOT / 'server.sh'), 'start', '--task', args.task,
                        '--model', args.model], check=True)
        current = info()
    if current is None or 'error' in current:
        raise RuntimeError(f'推理服务未就绪：{current}')
    if (current.get('task'), current.get('model')) != (args.task, args.model):
        raise RuntimeError(
            f"本地服务是 {current.get('model')}/{current.get('task')}，所需为 "
            f'{args.model}/{args.task}。先运行 bash server/server.sh restart '
            f'--task {args.task} --model {args.model}'
        )
    print(f"本地推理服务就绪：{args.model}/{args.task}，step={current.get('step')}", flush=True)


if __name__ == '__main__':
    main()
