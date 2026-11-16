"""Bounded held-out diffusion loss, independent of rollout/video evaluation."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def episode_ids(dataset) -> set[str]:
    records = getattr(dataset, "records", None)
    if records is None:
        records = getattr(dataset, "meta", {}).get("samples")
    if not records or any("episode_id" not in r for r in records):
        raise ValueError("Validation requires auditable episode_id metadata in records or meta.samples")
    return {str(r["episode_id"]) for r in records}


def validate_loss_setup(cfg, train_dataset, val_dataset) -> None:
    options = cfg.get("validation", {})
    if not options.get("enabled", False):
        return
    for field in ("every_steps", "max_samples", "batch_size"):
        if int(options.get(field, 0)) <= 0:
            raise ValueError(f"validation.{field} must be positive")
    if int(options.get("loss_optimizer_step", 1)) <= 0:
        raise ValueError("validation.loss_optimizer_step must be positive and fixed")
    if val_dataset is None or len(val_dataset) == 0:
        raise ValueError("validation.enabled requires a nonempty independent data.val; no training-data fallback")
    if val_dataset is train_dataset:
        raise ValueError("Training dataset cannot be used as validation dataset")
    if options.get("require_episode_disjoint", False):
        overlap = episode_ids(train_dataset) & episode_ids(val_dataset)
        if overlap:
            raise ValueError(f"Train/validation episode overlap: {len(overlap)} episodes")
    train_cache = getattr(train_dataset, "vae_latent_cache", None)
    val_cache = getattr(val_dataset, "vae_latent_cache", None)
    if options.get("require_cached", False) and val_cache is None:
        raise ValueError("validation.require_cached requires a VAE latent validation cache")
    if train_cache is not None and val_cache is not None:
        for field in ("torch_dtype", "vae_model_fingerprint"):
            if getattr(train_cache, field) != getattr(val_cache, field):
                raise ValueError(f"Train/validation latent cache mismatch: {field}")
        train_contract = getattr(train_dataset, "meta", {}).get("data_contract")
        val_contract = getattr(val_dataset, "meta", {}).get("data_contract")
        provenance = {"indices", "manifest_sha256", "split"}
        if train_contract is not None:
            sampling = {k: v for k, v in train_contract.items() if k not in provenance}
            other = {k: v for k, v in (val_contract or {}).items() if k not in provenance}
            if sampling != other:
                raise ValueError("Train/validation latent sampling contracts differ")


def selected_indices(size: int, limit: int, seed: int) -> list[int]:
    # random.sample(range(...)) does not allocate a full million-window randperm.
    return random.Random(seed).sample(range(size), min(size, limit))


def aggregate(records: list[dict]) -> dict:
    errors = [r["error"] for r in records if r["error"]]
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors))
    populated = [r for r in records if r["count"]]
    if not populated:
        raise RuntimeError("No validation samples evaluated")
    keys = set(populated[0]["sums"])
    if any(set(r["sums"]) != keys for r in populated):
        raise RuntimeError("Validation metric keys differ across ranks")
    count = sum(r["count"] for r in populated)
    result = {f"val/{k}": sum(r["sums"][k] for r in populated) / count for k in sorted(keys)}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Nonfinite validation aggregate")
    result["val/num_samples"] = count
    return result


def _seed_local(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.default_generators[device.index].manual_seed(seed)


@contextmanager
def isolated_eval(model, device: torch.device):
    modes = [(module, module.training) for module in model.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)


def evaluate_validation_loss(trainer) -> dict:
    options = trainer.cfg.validation
    accelerator = trainer.accelerator
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is not None and plugin.deepspeed_config.get("zero_optimization", {}).get("stage") == 3:
        raise ValueError("Loss-only validation currently supports replicated parameters / ZeRO-2, not ZeRO-3")
    model = accelerator.unwrap_model(trainer.model)
    device = accelerator.device
    seed = int(options.seed)
    batch_size = int(options.batch_size)
    indices = selected_indices(len(trainer.val_dataset), int(options.max_samples), seed)
    local_indices = indices[accelerator.process_index::accelerator.num_processes]
    record = {"sums": {}, "count": 0, "error": None}
    started = time.perf_counter()
    try:
        with isolated_eval(model, device):
            _seed_local(seed, device)
            loader = DataLoader(Subset(trainer.val_dataset, local_indices), batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(seed))
            for batch_number, sample in enumerate(loader):
                offset = batch_number * batch_size
                count = min(batch_size, len(local_indices) - offset)
                # Default batch1 gives the same window/noise pairing across world sizes.
                _seed_local(seed + local_indices[offset], device)
                with accelerator.autocast():
                    loss, details = trainer._compute_training_loss(
                        model, sample, optimizer_step=int(options.loss_optimizer_step))
                values = {"loss": float(loss.detach().float().item())}
                values.update({k: float(v) for k, v in details.items()})
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("Nonfinite validation loss/metrics")
                if record["count"] and set(values) != set(record["sums"]):
                    raise ValueError("Validation metric keys changed between batches")
                for key, value in values.items():
                    record["sums"][key] = record["sums"].get(key, 0.0) + value * count
                record["count"] += count
    except Exception as error:
        record["error"] = f"rank{accelerator.process_index}: {type(error).__name__}: {error}"
    records = [record]
    if torch.distributed.is_initialized():
        records = [None] * torch.distributed.get_world_size()
        # Small scalar summaries only. No padding/duplication of validation examples.
        torch.distributed.all_gather_object(records, record)
    result = aggregate(records)
    result.update({"val/loss_optimizer_step": int(options.loss_optimizer_step),
                   "val/seconds": time.perf_counter() - started})
    return result


def maybe_log_validation_loss(trainer) -> dict | None:
    options = trainer.cfg.get("validation", {})
    if not options.get("enabled", False):
        return None
    if trainer.global_step % int(options.every_steps):
        return None
    result = evaluate_validation_loss(trainer)
    if trainer.accelerator.is_main_process:
        with (Path(trainer.output_dir) / "validation_loss.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": trainer.global_step, **result}, allow_nan=False) + "\n")
        trainer._wandb_log(result)
    return result
