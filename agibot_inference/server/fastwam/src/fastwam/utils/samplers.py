from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = self._indices(g)
        if self.epoch == 0 and self.resume_batch_offset > 0:
            global_batch_size = self.batch_size * self.num_processes
            # Accelerate's even_batches=True pads an incomplete global batch
            # using the beginning of the full epoch. Preserve those indices
            # before skipping consumed batches; otherwise its sharder pads
            # using the beginning of the remaining suffix after a restart.
            if indices:
                padding = (-len(indices)) % global_batch_size
                indices += [indices[i % len(indices)] for i in range(padding)]
            sample_offset = self.resume_batch_offset * global_batch_size
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def _indices(self, generator: torch.Generator) -> list[int]:
        return torch.randperm(len(self.dataset), generator=generator).tolist()


class EqualTaskSampler(ResumableEpochSampler):
    """Uniform task probability, uniform frames within each task, resumable."""

    def _indices(self, generator: torch.Generator) -> list[int]:
        datasets = self.dataset.lerobot_dataset.multi_dataset._datasets
        lengths = [len(dataset) for dataset in datasets]
        if len(lengths) != 6 or min(lengths) <= 0 or sum(lengths) != len(self.dataset):
            raise ValueError(f"Invalid six-task frame partitions: {lengths}")
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        tasks = torch.randint(6, (len(self.dataset),), generator=generator)
        indices = torch.empty(len(self.dataset), dtype=torch.long)
        for task, length in enumerate(lengths):
            mask = tasks == task
            indices[mask] = offsets[task] + torch.randint(length, (int(mask.sum()),), generator=generator)
        return indices.tolist()
