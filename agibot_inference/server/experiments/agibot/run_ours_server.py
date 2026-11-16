#!/usr/bin/env python3
"""AgiBot r8 四任务 ours/FastWAM 推理服务：加载训练配置和最终权重，持续等待 TCP 客户端。"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 本文件放在 FastWAM/experiments/agibot/，复用模型源码，不修改仿真部署接口。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fastwam/src'))
sys.path.insert(0, str(ROOT.parent / 'scripts'))
from agibot_tasks import canonical_task, default_run_dir, FAMILIES, DEFAULT_PROMPT, MODEL_JOINT_NAMES
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from _ours_protocol import recv_message, send_message

TEXT_CACHE = Path('/data/agibot_inference/text')


def prepare_text_context(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """与 RobotVideoDataset 训练路径一致：padding 清零，但保留全部128个位置。"""
    context = cache['context'].clone()
    original_mask = cache['mask'].bool()
    context[~original_mask] = 0.0
    return context, torch.ones_like(original_mask)


class Predictor:
    def __init__(self, run_dir: Path, checkpoint: Path, device: str, steps: int,
                 task: str, family: str) -> None:
        cfg = OmegaConf.load(run_dir / 'config.yaml')
        self.task, self.family = canonical_task(task), family
        self.checkpoint = str(checkpoint.resolve())
        expected_target = {'ours': 'fastwam.runtime.create_fastwam_dual_action_idm',
                           'fastwam': 'fastwam.runtime.create_fastwam'}[family]
        if cfg.model._target_ != expected_target or cfg.model.proprio_dim != 19:
            raise ValueError('Model family/proprio dimension differs from requested AgiBot model')
        if family == 'ours':
            if (cfg.model.full_action_dim, cfg.model.arm_action_dim, cfg.model.cam_action_dim) != (19, 16, 3):
                raise ValueError('Expected ours dimensions 19=16+3')
        elif cfg.model.action_dit_config.action_dim != 19:
            raise ValueError('Expected FastWAM 19D action model')
        data = cfg.data.train
        if [Path(p).name for p in data.dataset_dirs] != [self.task]:
            raise ValueError('Training task differs from requested task')
        image_meta = data.shape_meta.images[0]
        if (len(data.shape_meta.images) != 1 or image_meta.key != 'cam_active'
                or list(image_meta.raw_shape) != [3, 480, 640]
                or list(image_meta.shape) != [3, 224, 224]
                or data.shape_meta.action[0].shape != 19 or data.shape_meta.state[0].shape != 19
                or data.num_frames != 33 or data.global_sample_stride != 1):
            raise ValueError('Training observation/action schema differs from the 20Hz AgiBot interface')
        # 与转换清单逐关节核对，不能只检查向量长度。
        metadata = json.loads((ROOT / 'fastwam/metadata' / f'{self.task}.json').read_text())
        names = [metadata['joint_name_mapping'][n] for n in metadata['state_names']]
        if names != list(MODEL_JOINT_NAMES) or metadata['fps'] != 20:
            raise ValueError('Training joint order/frequency differs from robot interface')
        prompt = DEFAULT_PROMPT.format(task=self.task)
        cache_file = TEXT_CACHE / (hashlib.sha256(prompt.encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt')
        if hashlib.sha256(cache_file.read_bytes()).hexdigest() != metadata['text_sha256']:
            raise ValueError('Text cache differs from verified training cache')
        self.stats_sha256 = hashlib.sha256((run_dir / 'dataset_stats.json').read_bytes()).hexdigest()
        self.text_sha256 = metadata['text_sha256']
        self.horizon = int(cfg.data.train.num_frames) - 1
        self.video_frames = self.horizon // int(cfg.data.train.action_video_freq_ratio) + 1
        self.steps = steps
        # 已有完整训练权重，跳过原始 DiT 权重；使用当前任务训练时的文本缓存。
        cfg.model.load_text_encoder = False
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.compile_training_denoise = False
        print(f'Loading model from {checkpoint}', flush=True)
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        # 严格加载，不能忽略缺失参数而使用随机初始化的权重。
        self.model.mot.load_state_dict(weights['mot'], strict=True)
        self.model.proprio_encoder.load_state_dict(weights['proprio_encoder'], strict=True)
        self.step = int(weights['step'])
        del weights
        self.model = self.model.to(device).eval()
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run_dir / 'dataset_stats.json')))
        # 直接使用训练的 ToTensor + Resize，保持图像预处理一致。
        self.image_transform = instantiate(cfg.data.train.processor.val_transforms[0])
        self.image_resize = instantiate(cfg.data.train.processor.val_transforms[1])
        text = torch.load(cache_file, map_location='cpu', weights_only=True)
        if text['context'].shape != (128, 4096) or text['mask'].shape != (128,):
            raise ValueError('Invalid training text cache shape')
        self.context, self.mask = prepare_text_context(text)
        self.fusion_sigma = float(cfg.model.get('fusion_sigma', 0.4))
        print(f'Model ready: model={family}, task={self.task}, step={self.step}, action=[{self.horizon},19], denoise_steps={steps}, text_mask={int(self.mask.sum())}/{self.mask.numel()}', flush=True)

    @torch.inference_mode()
    def predict(self, observation: dict) -> np.ndarray:
        rgb = np.asarray(observation['active_rgb'])
        state = np.asarray(observation['state'], dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('active_rgb must be RGB uint8 [H,W,3]')
        if state.shape != (19,) or not np.isfinite(state).all():
            raise ValueError('state must be finite [19]')
        # 训练视频为 640x480；输入统一到同一尺寸后再缩到 224x224。
        if rgb.shape != (480, 640, 3):
            raise ValueError(f'Expected RGB [480,640,3], got {rgb.shape}')
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        image = self.image_resize(self.image_transform(image)).mul(2).sub(1)
        batch = {'state': {'default': torch.from_numpy(state).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        proprio = self.processor.normalizer.forward(batch)['state']['default']
        common = dict(prompt=None, context=self.context, context_mask=self.mask,
                      input_image=image, proprio=proprio, num_inference_steps=self.steps,
                      seed=42, rand_device='cpu', tiled=False)
        if self.family == 'ours':
            out = self.model.infer_action(
                **common, cam_action_horizon=self.horizon, arm_action_horizon=self.horizon,
                num_video_frames=self.video_frames, fusion_sigma=self.fusion_sigma,
                parallel_coarse_video=False)
            normalized = torch.cat([out['arm_action'], out['cam_action']], dim=-1)
        else:
            out = self.model.infer_action(**common, action_horizon=self.horizon)
            normalized = out['action']
        norm = self.processor.normalizer.normalizers['action']['default']
        actions = norm.backward(normalized.float().cpu()).numpy().astype(np.float32)
        if actions.shape != (self.horizon, 19) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid output: {actions.shape}')
        return actions


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
                            response = {'protocol_version': 2, 'step': predictor.step, 'horizon': predictor.horizon,
                                        'action_dim': 19, 'fps': 20, 'task': predictor.task,
                                        'model': predictor.family, 'state_order': list(MODEL_JOINT_NAMES),
                                        'checkpoint': predictor.checkpoint, 'stats_sha256': predictor.stats_sha256,
                                        'text_sha256': predictor.text_sha256}
                        elif request.get('cmd') == 'infer':
                            if request.get('instruction') != predictor.task or request.get('model') != predictor.family:
                                raise ValueError('Client task/model differs from loaded checkpoint')
                            started = time.monotonic()
                            actions = predictor.predict(request['observation'])
                            elapsed = time.monotonic() - started
                            response = {'actions': actions, 'inference_seconds': elapsed}
                            print(f'Inference {actions.shape}: {elapsed:.3f}s', flush=True)
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
    parser.add_argument('--task', type=canonical_task, default='clothes')
    parser.add_argument('--model', choices=FAMILIES, default='ours')
    parser.add_argument('--run-dir', type=Path, help='覆盖任务的默认权重目录')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-inference-steps', type=int, default=10)
    args = parser.parse_args()
    if args.num_inference_steps <= 0:
        parser.error('--num-inference-steps must be positive')
    run_dir = args.run_dir or default_run_dir(args.task, args.model)
    filename = 'step_050000.pt' if args.model == 'ours' else 'step_030000.pt'
    checkpoint = args.checkpoint or run_dir / 'checkpoints/weights' / filename
    predictor = Predictor(run_dir, checkpoint, args.device, args.num_inference_steps, args.task, args.model)
    serve(predictor, args.host, args.port)


if __name__ == '__main__':
    main()
