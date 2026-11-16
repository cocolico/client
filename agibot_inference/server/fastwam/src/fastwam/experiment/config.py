"""Strict schema and loader for reproducible FastWAM experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class ResourceConfig:
    nodes: int = 2
    gpus_per_node: int = 8


@dataclass
class PathConfig:
    source_data_root: str = ""
    matched_data_root: str = ""
    rgb_source_root: str = ""
    pretrain_checkpoint: str = ""
    output_root: str = ""
    wandb_key_file: str = ""


@dataclass
class DataConfig:
    mode: str = "source"
    datasets: list[str] = field(default_factory=list)
    expected_frames: dict[str, int] = field(default_factory=dict)
    expected_episodes: int = 0
    expected_total_frames: int = 0
    expected_transition_contract: str = ""
    prepared_root: str = ""
    cache_root: str = ""
    hf_cache_root: str = "/dev/shm/fastwam_hf_datasets"
    cache_workers: int = 64
    min_shm_gb: int = 450
    target_fps: int = 20
    source_fps: int = 50


@dataclass
class TrainingConfig:
    task: str = ""
    run_name: str = ""
    resume: str = ""
    log_every: int = 1
    save_every: int = 1000
    eval_every: int = 100000


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "fastwam-robotwin-av"
    group: str = "cam-active-6task"


@dataclass
class ExperimentConfig:
    schema_version: int = 1
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    runtime_env: dict[str, str] = field(default_factory=dict)
    expected_hydra: dict[str, Any] = field(default_factory=dict)
    expected_artifacts: dict[str, str] = field(default_factory=dict)


def _load_with_base(path: Path, seen: set[Path]) -> DictConfig:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic experiment base chain: {path}")
    seen.add(path)
    raw = OmegaConf.load(path)
    base = raw.pop("base", None)
    if base is None:
        return raw
    base_path = (path.parent / str(base)).resolve()
    return OmegaConf.merge(_load_with_base(base_path, seen), raw)


def _validate(cfg: DictConfig) -> None:
    if cfg.schema_version != 1:
        raise ValueError(f"Unsupported schema_version={cfg.schema_version}")
    if cfg.resources.nodes <= 0 or cfg.resources.gpus_per_node <= 0:
        raise ValueError("resources.nodes and gpus_per_node must be positive")
    if cfg.data.mode not in {"source", "downsample_20hz", "matched_eef"}:
        raise ValueError(f"Unsupported data.mode={cfg.data.mode!r}")
    if not cfg.training.task or not cfg.training.run_name:
        raise ValueError("training.task and training.run_name are required")
    if cfg.training.save_every <= 0 or cfg.training.save_every % 1000 != 0:
        raise ValueError("training.save_every must be a positive multiple of 1000")
    if cfg.data.cache_workers <= 0:
        raise ValueError("data.cache_workers must be positive")
    if not cfg.data.hf_cache_root:
        raise ValueError("data.hf_cache_root is required")
    if not cfg.data.datasets or len(cfg.data.datasets) != len(set(cfg.data.datasets)):
        raise ValueError("data.datasets must be a non-empty unique list")
    if not cfg.paths.source_data_root:
        raise ValueError("paths.source_data_root is required")
    if cfg.data.mode == "downsample_20hz" and not cfg.data.prepared_root:
        raise ValueError("downsample_20hz requires data.prepared_root")
    if cfg.data.mode == "matched_eef":
        if not cfg.paths.matched_data_root or not cfg.paths.rgb_source_root:
            raise ValueError("matched_eef requires matched_data_root and rgb_source_root")
        if set(cfg.data.expected_frames) != set(cfg.data.datasets):
            raise ValueError("matched_eef expected_frames must cover every dataset")
        if cfg.data.expected_episodes <= 0 or cfg.data.expected_total_frames <= 0:
            raise ValueError("matched_eef requires positive expected episode/frame totals")
    if cfg.training.resume == "pretrain_checkpoint" and not cfg.paths.pretrain_checkpoint:
        raise ValueError("pretrain_checkpoint resume requires paths.pretrain_checkpoint")
    if cfg.wandb.enabled and not cfg.paths.wandb_key_file:
        raise ValueError("wandb.enabled requires paths.wandb_key_file")


def load_experiment_config(path: str | Path) -> DictConfig:
    """Load one experiment, recursively merge its base, resolve env, and validate."""

    merged = _load_with_base(Path(path), set())
    structured = OmegaConf.structured(ExperimentConfig)
    cfg = OmegaConf.merge(structured, merged)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    _validate(cfg)
    return cfg


def resolved_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)


def config_fingerprint(cfg: DictConfig) -> str:
    return hashlib.sha256(resolved_yaml(cfg).encode("utf-8")).hexdigest()
