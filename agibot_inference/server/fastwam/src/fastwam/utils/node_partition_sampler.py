"""One deterministic global stream; Accelerate assigns eight batches per node."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler


def node_bounds(size: int, nodes: int, node: int) -> tuple[int, int]:
    if nodes < 1 or size < nodes or not 0 <= node < nodes:
        raise ValueError('Invalid nonempty node partition')
    return size * node // nodes, size * (node + 1) // nodes


class NodePartitionSampler(Sampler[int]):
    def __init__(self, dataset, seed: int, batch_size: int, num_processes: int,
                 nodes: int, gpus_per_node: int):
        if num_processes != nodes * gpus_per_node or batch_size < 1:
            raise ValueError('Node sampler requires the configured complete world')
        self.size, self.seed = len(dataset), int(seed)
        self.nodes, self.node_batch = nodes, batch_size * gpus_per_node
        self.bounds = [node_bounds(self.size, nodes, n) for n in range(nodes)]
        self.steps = math.ceil(max(b-a for a,b in self.bounds) / self.node_batch)
        self.global_batch = self.node_batch * nodes
        self.epoch = self.epoch_offset = self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, offset: int):
        self.epoch_offset = int(offset)

    def set_resume_batch_offset(self, offset: int):
        if offset < 0 or offset > self.steps:
            raise ValueError('Invalid resume batch offset')
        self.resume_batch_offset = int(offset)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __len__(self):
        return self.steps * self.global_batch

    def __iter__(self):
        permutations = []
        for node, (start, end) in enumerate(self.bounds):
            gen = torch.Generator().manual_seed(
                self.seed + (self.epoch + self.epoch_offset) * 1000003 + node * 9176)
            permutations.append(torch.randperm(end-start, generator=gen))
        offset = self.resume_batch_offset if self.epoch == 0 else 0
        for step in range(offset, self.steps):
            for (start, end), permutation in zip(self.bounds, permutations):
                indices = torch.arange(step*self.node_batch, (step+1)*self.node_batch) % (end-start)
                yield from (permutation[indices] + start).tolist()
