#!/usr/bin/env python3
"""AgiBot PI05 inference server using the same TCP protocol as ours/fastwam."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

OURS_ROOT = Path(__file__).resolve().parents[2]
AGIBOT_ROOT = OURS_ROOT.parent
PI05_ROOT = AGIBOT_ROOT / 'vendor/openpi'
sys.path.insert(0, str(AGIBOT_ROOT / 'scripts'))
sys.path.insert(0, str(PI05_ROOT))
sys.path.insert(0, str(OURS_ROOT / 'experiments/agibot'))

from agibot_tasks import (canonical_task, DEFAULT_PROMPT, interface_spec,
                          SHARPA_TRAIN_CONFIG, SHARPA_CHECKPOINT)
from _ours_protocol import recv_message, send_message
from pi_model import PI05ActiveOnly, PI05Sharpa61

DEFAULT_CKPT_ROOT = Path('/data/agibot_inference/models/pi05')
DEFAULT_TRAIN_CONFIG = 'pi05_base_aloha_robotwin_active_only_absolute_action_lora'


def _stats_dim(stats: dict, key: str) -> int:
    root = stats.get('norm_stats', stats)
    item = root.get(key)
    if not isinstance(item, dict):
        raise ValueError(f'norm_stats missing {key}')
    for stat_name in ('mean', 'std', 'q01', 'q99'):
        value = item.get(stat_name)
        if isinstance(value, list):
            return len(value)
    raise ValueError(f'norm_stats.{key} has no vector statistic')


class Predictor:
    def __init__(self, task: str, checkpoint_dir: Path, train_config: str, action_chunk: int) -> None:
        self.task = canonical_task(task)
        self.family = 'pi05'
        self.spec = interface_spec(self.task, self.family)
        self.action_dim = self.spec['action_dim']
        if self.task == 'yifu' and (action_chunk != 50 or train_config != SHARPA_TRAIN_CONFIG):
            raise ValueError('yifu requires the 61D full-finetune config and 50 frames')
        self.checkpoint_dir = checkpoint_dir.expanduser().resolve()
        self.checkpoint = str(self.checkpoint_dir)
        self.horizon = action_chunk
        self.step = self._read_step(self.checkpoint_dir)
        self.prompt = '晾衣服' if self.task == 'yifu' else DEFAULT_PROMPT.format(task=self.task)

        if not self.checkpoint_dir.is_dir():
            raise FileNotFoundError(f'checkpoint dir not found: {self.checkpoint_dir}')
        stats_file = self.checkpoint_dir / 'assets' / self.task / 'norm_stats.json'
        if not stats_file.exists():
            raise FileNotFoundError(f'norm stats not found: {stats_file}')
        stats = json.loads(stats_file.read_text())
        state_dim = _stats_dim(stats, 'state')
        action_dim = _stats_dim(stats, 'actions')
        if state_dim != self.action_dim or action_dim != self.action_dim:
            raise ValueError(f'Expected PI05 {self.action_dim}D stats, got state={state_dim}, actions={action_dim}')
        self.stats_sha256 = hashlib.sha256(stats_file.read_bytes()).hexdigest()

        print(f'Loading PI05 model from {self.checkpoint_dir}', flush=True)
        print(f'PI05 prompt: {self.prompt}', flush=True)
        model_class = PI05Sharpa61 if self.task == 'yifu' else PI05ActiveOnly
        self.model = model_class(train_config, self.checkpoint_dir, action_chunk=action_chunk)
        print(f'PI05 ready: task={self.task}, step={self.step}, action=[{self.horizon},{self.action_dim}]', flush=True)

    @staticmethod
    def _read_step(checkpoint_dir: Path) -> int:
        try:
            return int(checkpoint_dir.name)
        except ValueError:
            return -1

    def predict(self, observation: dict) -> np.ndarray:
        if self.task == 'yifu':
            state = np.asarray(observation['state'], dtype=np.float32)
            if state.shape != (61,) or not np.isfinite(state).all():
                raise ValueError('state must be finite [61]')
            images = observation['images']
            if set(images) != set(self.spec['camera_keys']):
                raise ValueError('All three trained camera keys are required')
            for key, rgb in images.items():
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
                    raise ValueError(f'{key} must be uint8 HWC RGB')
            actions = np.asarray(self.model.infer({'prompt': self.prompt, 'images': images, 'state': state}), dtype=np.float32)
            if actions.shape != (50, 61) or not np.isfinite(actions).all():
                raise ValueError(f'Expected finite (50,61) output, got {actions.shape}')
            return np.ascontiguousarray(actions)
        rgb = np.asarray(observation['active_rgb'])
        state = np.asarray(observation['state'], dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError(f'active_rgb must be uint8 [480,640,3], got {rgb.dtype} {rgb.shape}')
        if state.shape != (19,) or not np.isfinite(state).all():
            raise ValueError(f'state must be finite [19], got {state.shape}')

        payload = {'prompt': self.prompt, 'images': [rgb], 'state': state}
        actions = np.asarray(self.model.infer(payload), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[0] < self.horizon or actions.shape[1] < 19:
            raise ValueError(f'Invalid PI05 output shape: {actions.shape}')
        actions = actions[:self.horizon, :19].astype(np.float32, copy=False)
        if actions.shape != (self.horizon, 19) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid PI05 output after slicing: {actions.shape}')
        return np.ascontiguousarray(actions)


def serve(predictor: Predictor, host: str, port: int) -> None:
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(2)
        print(f'LISTENING {host}:{port}', flush=True)
        while True:
            client, address = server.accept()
            print(f'Connected: {address}', flush=True)
            with client:
                client.settimeout(600)
                while True:
                    try:
                        request = recv_message(client)
                        if request.get('cmd') == 'info':
                            response = {
                                'protocol_version': 2,
                                'step': predictor.step,
                                'horizon': predictor.horizon,
                                'action_dim': predictor.action_dim,
                                'fps': 20,
                                'task': predictor.task,
                                'model': predictor.family,
                                'state_order': predictor.spec['state_order'],
                                'checkpoint': predictor.checkpoint,
                                'stats_sha256': predictor.stats_sha256,
                            }
                            if 'camera_keys' in predictor.spec:
                                response['camera_keys'] = predictor.spec['camera_keys']
                                response['prompt'] = predictor.prompt
                        elif request.get('cmd') == 'infer':
                            if request.get('instruction') != predictor.task or request.get('model') != predictor.family:
                                raise ValueError('Client task/model differs from loaded PI05 checkpoint')
                            started = time.monotonic()
                            actions = predictor.predict(request['observation'])
                            elapsed = time.monotonic() - started
                            response = {'actions': actions, 'inference_seconds': elapsed}
                            print(f'PI05 inference {actions.shape}: {elapsed:.3f}s', flush=True)
                        else:
                            raise ValueError('Unknown command')
                        send_message(client, response)
                    except (ConnectionError, OSError):
                        break
                    except Exception as exc:
                        traceback.print_exc()
                        try:
                            send_message(client, {'error': f'{type(exc).__name__}: {exc}'})
                        except OSError:
                            pass
                        break
            print('Disconnected; waiting for next client', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=canonical_task, default='guizi')
    parser.add_argument('--model', choices=('pi05',), default='pi05', help='compatibility with unified server.sh arguments')
    parser.add_argument('--checkpoint-dir', type=Path, help='PI05 checkpoint step directory; default is the local model link 30000 dir')
    parser.add_argument('--train-config')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--action-chunk', type=int)
    args = parser.parse_args()
    if args.action_chunk is None:
        args.action_chunk = 50 if args.task == 'yifu' else 32
    if args.train_config is None:
        args.train_config = SHARPA_TRAIN_CONFIG if args.task == 'yifu' else DEFAULT_TRAIN_CONFIG
    if args.action_chunk <= 0:
        parser.error('--action-chunk must be positive')
    checkpoint_dir = args.checkpoint_dir or (SHARPA_CHECKPOINT if args.task == 'yifu' else DEFAULT_CKPT_ROOT / args.task / '30000')
    predictor = Predictor(args.task, checkpoint_dir, args.train_config, args.action_chunk)
    serve(predictor, args.host, args.port)


if __name__ == '__main__':
    main()
