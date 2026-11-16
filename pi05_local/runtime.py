"""Inference-only Pi0.5 humanhand runtime, matching the exported training transforms."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT/'checkpoints/pi05_posttrain_humanhand/8408'
MODEL_NAME = 'pi05_hh_8408'
TIMES = np.arange(1,33,dtype=np.float32)/30


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def state_right_left(keypoints):
    points=np.asarray(keypoints)
    if points.shape!=(2,21,3) or points.dtype!=np.float32 or not np.isfinite(points).all():
        raise ValueError('Expected float32 [left,right,21,xyz] in metres')
    return np.concatenate([points[1].reshape(63),points[0].reshape(63)])


def actions_left_right(actions):
    actions=np.asarray(actions,dtype=np.float32)
    if actions.shape!=(50,126) or not np.isfinite(actions).all():
        raise ValueError('Expected finite unnormalized [50,126] Pi0.5 actions')
    return actions.reshape(50,2,21,3)[:,[1,0]].copy()


class Runtime:
    def __init__(self,checkpoint=CHECKPOINT,device='cuda:0'):
        self.checkpoint=Path(checkpoint).resolve();self.device=device
        for path in (ROOT/'vendor/openpi/src',ROOT/'vendor/openpi/packages/openpi-client/src'):
            sys.path.insert(0,str(path))
        import torch
        import safetensors.torch
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.policy import Policy
        from openpi import transforms as t
        from openpi.shared.normalize import NormStats

        self.torch=torch
        config=json.loads((self.checkpoint/'model_config.json').read_text())
        if config.get('action_dim')!=126 or config.get('action_horizon')!=50 or not config.get('pi05'):
            raise ValueError('Checkpoint is not the expected Pi0.5 humanhand export')
        norm_path=self.checkpoint/'assets/posttrain_humanhand/norm_stats.json'
        stats=json.loads(norm_path.read_text())['norm_stats']
        norm={key:NormStats(**{k:np.asarray(v) for k,v in value.items()}) for key,value in stats.items()}
        for key in ('state','actions'):
            if norm[key].q01.shape!=(126,) or norm[key].q99.shape!=(126,):
                raise ValueError('Wrong humanhand quantile statistics')
        model=PI0Pytorch(Pi0Config(**config))
        safetensors.torch.load_model(model,str(self.checkpoint/'model.safetensors'))
        model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
        tokenizer=PaligemmaTokenizer(config['max_token_len'])
        self.policy=Policy(model,transforms=[
            t.Normalize(norm,use_quantiles=True),t.ResizeImages(224,224),
            t.TokenizePrompt(tokenizer,discrete_state_input=True),t.PadStatesAndActions(126)],
            output_transforms=[t.Unnormalize(norm,use_quantiles=True)],
            is_pytorch=True,pytorch_device=device)
        self.identity=dict(model_family='pi05',native_horizon=50,action_dim=126,
            native_hand_order=['right','left'],wire_hand_order=['left','right'],
            coordinate_frame='cam0_color_optical_frame',units='metres',actions='absolute keypoints',
            normalization='checkpoint q01/q99',history_required=False,
            image_processing='received RGB -> training PIL resize_with_pad 224x224; no XFlow undistortion',
            weights_sha256=sha256(self.checkpoint/'model.safetensors'),
            norm_sha256=sha256(norm_path),model_config_sha256=sha256(self.checkpoint/'model_config.json'),
            adapter_sha256=sha256(__file__),server_sha256=sha256(ROOT/'server.py'),
            openpi_model_sha256=sha256(ROOT/'vendor/openpi/src/openpi/models_pytorch/pi0_pytorch.py'),
            openpi_transforms_sha256=sha256(ROOT/'vendor/openpi/src/openpi/transforms.py'),
            tokenizer_sha256=sha256(ROOT/'cache/big_vision/paligemma_tokenizer.model'))
        manifest=json.loads((ROOT/'source_manifest.json').read_text())
        if self.identity['weights_sha256'] != manifest['model_safetensors_sha256']:
            raise ValueError('Downloaded checkpoint SHA256 differs from remote source')
        self.deployment_id=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()[:16]

    def describe(self):
        return dict(name=MODEL_NAME,weights=str(self.checkpoint/'model.safetensors'),
            deployment_id=self.deployment_id,contract='humanhand',
            history_offset_frames=0,keypoint_time_s=TIMES.tolist(),
            **self.identity)

    def infer(self,observation,prompt,steps=10,seed=0):
        image=np.asarray(observation['head_rgb'])
        if image.shape!=(400,640,3) or image.dtype!=np.uint8:
            raise ValueError('Expected RGB uint8 [400,640,3]')
        if not isinstance(prompt,str) or not prompt.strip(): raise ValueError('Task prompt required')
        if type(steps) is not int or not 1<=steps<=100: raise ValueError('Invalid inference steps')
        state=state_right_left(observation['keypoints'])
        zeros=np.zeros_like(image)
        data=dict(image={'base_0_rgb':image,'left_wrist_0_rgb':zeros,'right_wrist_0_rgb':zeros},
            image_mask={'base_0_rgb':np.True_,'left_wrist_0_rgb':np.False_,'right_wrist_0_rgb':np.False_},
            state=state,prompt=prompt)
        self.policy._sample_kwargs={'num_steps':steps}
        self.torch.manual_seed(int(seed))
        self.torch.cuda.synchronize(self.device)
        begin=time.monotonic()
        with self.torch.inference_mode(),self.torch.autocast('cuda',dtype=self.torch.bfloat16):
            result=self.policy.infer(data)
        self.torch.cuda.synchronize(self.device)
        points=actions_left_right(result['actions'])
        return points,dict(inference_seconds=time.monotonic()-begin,
            native_actions_right_left=np.asarray(result['actions'],np.float32),
            state_right_left=state,policy_timing=result.get('policy_timing',{}))
