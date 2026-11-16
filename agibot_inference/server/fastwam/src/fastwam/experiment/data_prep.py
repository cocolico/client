"""Idempotent node-local data preparation for FastWAM HTrain jobs."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig


def _run(command: list[str], cwd: Path) -> None:
    print("[exec]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _info(root: Path, name: str) -> dict:
    return json.loads((root / name / "meta" / "info.json").read_text())


def _validate_downsampled(
    root: Path,
    source: Path,
    datasets: list[str],
    source_fps: int,
    target_fps: int,
) -> None:
    for name in datasets:
        info = _info(root, name)
        manifest = json.loads((root / name / "downsample_manifest.json").read_text())
        if info["fps"] != target_fps:
            raise ValueError(f"{name}: expected fps={target_fps}, got {info['fps']}")
        if manifest["source"] != str((source / name).resolve()):
            raise ValueError(f"{name}: downsample source mismatch")
        if manifest["source_fps"] != source_fps:
            raise ValueError(f"{name}: downsample source fps mismatch")
        if manifest["target_frames"] != info["total_frames"]:
            raise ValueError(f"{name}: downsample frame count mismatch")


def _ensure_downsampled(cfg: DictConfig, repo: Path, python: str) -> Path:
    source = Path(cfg.paths.source_data_root)
    target = Path(cfg.data.prepared_root)
    try:
        _validate_downsampled(
            target,
            source,
            list(cfg.data.datasets),
            cfg.data.source_fps,
            cfg.data.target_fps,
        )
        return target
    except (FileNotFoundError, ValueError):
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite invalid prepared dataset: {target}")

    building = Path(f"{target}.building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    for name in cfg.data.datasets:
        _run(
            [
                python,
                "scripts/build_lerobot_downsampled_dataset.py",
                str(source / name),
                str(building / name),
                "--target-fps",
                str(cfg.data.target_fps),
                "--camera",
                "observation.images.cam_active",
            ],
            repo,
        )
    building.rename(target)
    _validate_downsampled(
        target,
        source,
        list(cfg.data.datasets),
        cfg.data.source_fps,
        cfg.data.target_fps,
    )
    return target


def _validate_matched(cfg: DictConfig, root: Path) -> None:
    ready = json.loads((root / "MATCHED_READY.json").read_text())
    if (
        ready.get("episodes") != cfg.data.expected_episodes
        or ready.get("total_frames") != cfg.data.expected_total_frames
    ):
        raise ValueError(f"Matched EEF READY contract mismatch: {ready}")
    if ready.get("transition_contract") != cfg.data.expected_transition_contract:
        raise ValueError(f"Matched EEF transition contract mismatch: {ready}")
    for name, frames in cfg.data.expected_frames.items():
        info = _info(root, name)
        if info["fps"] != 50 or info["total_frames"] != frames:
            raise ValueError(f"{name}: matched EEF timeline mismatch")
        if info["features"]["action"]["shape"] != [23]:
            raise ValueError(f"{name}: expected raw EEF action shape [23]")


def _validate_cache(
    cache: Path, data: Path, image_source: Path, datasets: list[str]
) -> None:
    for name in datasets:
        manifest = json.loads((cache / name / "manifest.json").read_text())
        info = _info(data, name)
        if manifest["source"] != str((data / name).resolve()):
            raise ValueError(f"{name}: cache manifest source mismatch")
        if manifest["image_source"] != str((image_source / name).resolve()):
            raise ValueError(f"{name}: cache image source mismatch")
        if manifest["total_frames"] != info["total_frames"]:
            raise ValueError(f"{name}: cache frame count mismatch")
        expected_camera = {"observation.images.cam_active": "cam_active.npy"}
        if manifest["cameras"] != expected_camera:
            raise ValueError(f"{name}: cache camera contract mismatch")


def _ensure_cache(
    cfg: DictConfig,
    repo: Path,
    python: str,
    data: Path,
    image_source: Path,
) -> Path:
    cache = Path(cfg.data.cache_root)
    try:
        _validate_cache(cache, data, image_source, list(cfg.data.datasets))
        return cache
    except (FileNotFoundError, ValueError):
        if cache.exists():
            raise RuntimeError(f"Refusing to overwrite invalid frame cache: {cache}")

    available_gb = shutil.disk_usage("/dev/shm").free / (1024**3)
    if available_gb < cfg.data.min_shm_gb:
        raise RuntimeError(
            f"Insufficient /dev/shm: {available_gb:.1f} GiB available, "
            f"{cfg.data.min_shm_gb} GiB required"
        )
    building = Path(f"{cache}.building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    for name in cfg.data.datasets:
        _run(
            [
                python,
                "scripts/build_lerobot_raw_frame_cache.py",
                str(image_source / name),
                str(building / name),
                "--manifest-source",
                str(data / name),
                "--camera",
                "observation.images.cam_active",
                "--workers",
                str(cfg.data.cache_workers),
            ],
            repo,
        )
    building.rename(cache)
    _validate_cache(cache, data, image_source, list(cfg.data.datasets))
    return cache


def prepare_data(cfg: DictConfig, repo: Path, python: str) -> tuple[Path, Path]:
    """Return the model data root and validated node-local RGB cache root."""

    source = Path(cfg.paths.source_data_root)
    if cfg.data.mode == "downsample_20hz":
        data = _ensure_downsampled(cfg, repo, python)
        image_source = data
    elif cfg.data.mode == "matched_eef":
        data = Path(cfg.paths.matched_data_root)
        _validate_matched(cfg, data)
        image_source = Path(cfg.paths.rgb_source_root)
    else:
        data = source
        image_source = source
    for name in cfg.data.datasets:
        _info(data, name)
        _info(image_source, name)
    cache = _ensure_cache(cfg, repo, python, data, image_source)
    return data, cache
