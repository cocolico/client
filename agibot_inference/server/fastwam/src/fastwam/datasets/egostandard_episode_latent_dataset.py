"""Audited episode caches: bounded mmap handles, no MCAP or encoder at runtime."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from fastwam.vae_latent_cache import stable_json_hash


class EgoStandardEpisodeLatentDataset(Dataset):
    def __init__(self, cache_dir, split='train', processor=None, max_open_episodes=32,
                 node_cache_dir=None):
        del processor
        self.root = Path(cache_dir).expanduser().resolve()
        self.contract = json.loads((self.root/'contract.json').read_text())
        self.fingerprint = stable_json_hash(self.contract)
        complete = json.loads((self.root/'COMPLETE.json').read_text())
        manifest = json.loads((self.root/'manifest.json').read_text())
        if (manifest['schema'] != 'egostandard_episode_manifest_v1'
                or complete['fingerprint'] != self.fingerprint
                or manifest['fingerprint'] != self.fingerprint
                or complete['num_episodes'] != len(manifest['episodes'])
                or not complete.get('deep_audit') or not manifest.get('deep_audit')
                or complete['num_windows'] != manifest['num_windows']
                or sum(e['windows'] for e in manifest['episodes']) != manifest['num_windows']):
            raise ValueError('Incomplete or incompatible cache manifest')
        self.node_root = Path(node_cache_dir).resolve() if node_cache_dir else None
        self.split = split
        if self.node_root is not None:
            ready = json.loads((self.node_root/'READY.json').read_text())
            if ready['fingerprint'] != self.fingerprint:
                raise ValueError('RAM cache fingerprint mismatch')
            if ready['manifest_sha256'] != hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest():
                raise ValueError('RAM cache manifest mismatch')
            if 'NODE_RANK' in os.environ and ready['node'] != int(os.environ['NODE_RANK']):
                raise ValueError('RAM cache belongs to a different node')
        self.vae_latent_cache = SimpleNamespace(
            torch_dtype=torch.bfloat16,
            vae_model_fingerprint=stable_json_hash(self.contract['vae_model']))
        if split not in ('train','val','all') or max_open_episodes < 1:
            raise ValueError('Invalid split or mmap handle limit')
        self.episodes = [e for e in manifest['episodes'] if e['windows']>0
                         and (split=='all' or e['split']==split)]
        self.ends = np.cumsum([e['windows'] for e in self.episodes],dtype=np.int64)
        self.records = self.episodes
        self.meta = {'data_contract': self.contract}
        self.max_open_episodes = max_open_episodes
        self._pid = None
        self._episodes = OrderedDict()
        self._text = OrderedDict()

    def _path(self, relative):
        if self.node_root is not None:
            path = self.node_root / relative
            if path.is_file():
                return path
            if self.split == 'train':
                raise FileNotFoundError(f'Training sample was not staged on this node: {relative}')
        return self.root / relative

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None,_episodes=OrderedDict(),_text=OrderedDict())
        return state

    def _load(self, store, key, path):
        if key not in store:
            store[key] = torch.load(path,weights_only=True,mmap=True)
        store.move_to_end(key)
        while len(store)>self.max_open_episodes:
            store.popitem(last=False)
        return store[key]

    def __getitem__(self,index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self._episodes,self._text = OrderedDict(),OrderedDict()
            self._pid = os.getpid()
        episode = int(np.searchsorted(self.ends,index,side='right'))
        row = int(index-(self.ends[episode-1] if episode else 0))
        entry = self.episodes[episode]
        saved = self._load(self._episodes,episode,self._path(entry['cache']))
        if saved['fingerprint']!=self.fingerprint:
            raise ValueError('Episode fingerprint mismatch')
        text = self._load(self._text,saved['prompt_id'],self._path('text/'+saved['prompt_id']+'.pt'))
        result = {key:saved[key][row].clone() for key in
                  ('video_latents','action','action_valid_mask','proprio')}
        result.update(context=text['context'].clone(),context_mask=text['context_mask'].clone(),
                      action_is_pad=torch.zeros(32,dtype=torch.bool),
                      proprio_is_pad=torch.zeros(32,dtype=torch.bool),
                      image_is_pad=torch.zeros(9,dtype=torch.bool),
                      slide_start_s=torch.tensor(row*.32,dtype=torch.float32),
                      episode_id=entry['episode_id'],task=entry['task'],prompt=text['prompt'])
        return result
