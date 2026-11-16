"""Bounded per-rank VAE RAM cache for deterministic Q-pose windows."""
from collections import OrderedDict
import torch


class LatentRAM:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    @torch.no_grad()
    def prepare(self, model, sample: dict) -> dict:
        if 'video_latents' in sample:
            return sample
        keys = sample['qpos_cache_key']
        values = []
        for i, key in enumerate(keys):
            if key not in self.cache:
                self.misses += 1
                # Microbatch1 bounds VAE workspace independently of training batch.
                video = sample['video'][i:i+1].to(device=model.device, dtype=model.torch_dtype)
                value = model.vae.model.encode(video, model.vae.scale).detach().cpu()
                size = value.numel() * value.element_size()
                if size > self.max_bytes:
                    raise ValueError('VAE cache capacity smaller than one sample')
                while self.bytes + size > self.max_bytes:
                    _, old = self.cache.popitem(last=False)
                    self.bytes -= old.numel() * old.element_size()
                self.cache[key] = value
                self.bytes += size
            else:
                self.hits += 1
            self.cache.move_to_end(key)
            values.append(self.cache[key])
        result = dict(sample)
        result.pop('video')
        result['video_latents'] = torch.cat(values)
        return result
