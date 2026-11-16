"""Fixed-shape, read-only-at-training-time cache for Wan VAE latents."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

CACHE_VERSION = 2
MANIFEST_NAME = "manifest.json"
SUPPORTED_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_vae_model_contract(weights_path: str | os.PathLike[str]) -> dict[str, Any]:
    wan22_root = Path(__file__).parent / "models" / "wan22"
    implementation_paths = (
        wan22_root / "wan_video_vae.py",
        wan22_root / "helpers" / "io.py",
        wan22_root / "helpers" / "loader.py",
        wan22_root / "helpers" / "state_dict_converters.py",
    )
    return {
        "contract_version": 1,
        "weights_sha256": sha256_file(weights_path),
        "implementation_files_sha256": {
            str(path.relative_to(Path(__file__).parent)): sha256_file(path)
            for path in implementation_paths
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _shard_size(num_samples: int, rank: int, world_size: int) -> int:
    if rank >= num_samples:
        return 0
    return (num_samples - 1 - rank) // world_size + 1


def _latent_file(rank: int) -> str:
    return f"latents.rank{rank:05d}.bin"


def _padding_file(rank: int) -> str:
    return f"image_is_pad.rank{rank:05d}.bin"


def _ready_file(rank: int) -> str:
    return f"ready.rank{rank:05d}.bin"


def _state_file(rank: int) -> str:
    return f"state.rank{rank:05d}.json"


def _shard_paths_and_sizes(
    root: Path,
    *,
    num_samples: int,
    rank: int,
    world_size: int,
    latent_shape: tuple[int, ...],
    num_video_frames: int,
) -> tuple[tuple[Path, int], ...]:
    rows = _shard_size(num_samples, rank, world_size)
    latent_bytes = rows * int(np.prod(latent_shape)) * np.dtype(np.uint16).itemsize
    padding_bytes = rows * num_video_frames * np.dtype(np.uint8).itemsize
    return (
        (root / _latent_file(rank), latent_bytes),
        (root / _padding_file(rank), padding_bytes),
        (root / _ready_file(rank), rows),
    )


def _validate_shard_file_sizes(
    root: Path,
    *,
    num_samples: int,
    rank: int,
    world_size: int,
    latent_shape: tuple[int, ...],
    num_video_frames: int,
    error_type: type[Exception] = ValueError,
) -> None:
    for path, expected_size in _shard_paths_and_sizes(
        root,
        num_samples=num_samples,
        rank=rank,
        world_size=world_size,
        latent_shape=latent_shape,
        num_video_frames=num_video_frames,
    ):
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_size != expected_size:
            raise error_type(
                f"Invalid VAE cache shard file {path}: "
                f"expected {expected_size} bytes, got {actual_size}."
            )


def cache_payload_bytes(
    root: str | os.PathLike[str],
    *,
    num_samples: int,
    world_size: int,
    latent_shape: Iterable[int],
    num_video_frames: int,
    existing_only: bool = False,
) -> int:
    """Return expected or already allocated bytes for the exact shard layout.

    Existing files are capped at their expected size and unrelated files are
    ignored, so stale artifacts cannot make a capacity preflight optimistic.
    """
    root_path = Path(root)
    shape = tuple(int(x) for x in latent_shape)
    total = 0
    for rank in range(int(world_size)):
        for path, expected_size in _shard_paths_and_sizes(
            root_path,
            num_samples=int(num_samples),
            rank=rank,
            world_size=int(world_size),
            latent_shape=shape,
            num_video_frames=int(num_video_frames),
        ):
            if existing_only:
                if not path.is_file():
                    continue
                stat = path.stat()
                # Memmaps are created at their final logical size up front, but
                # sparse files do not consume those bytes until rows are written.
                # POSIX st_blocks is reported in 512-byte units and therefore
                # reflects the space that a resumed job has actually allocated.
                allocated_size = (
                    stat.st_blocks * 512
                    if hasattr(stat, "st_blocks")
                    else stat.st_size
                )
                total += min(allocated_size, expected_size)
            else:
                total += expected_size
    return total


def load_shard_state(root: str | os.PathLike[str], rank: int) -> dict[str, Any] | None:
    state_path = Path(root) / _state_file(rank)
    if not state_path.is_file():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def shard_missing_indices(
    root: str | os.PathLike[str], *, num_samples: int, rank: int, world_size: int
) -> list[int]:
    rows = _shard_size(num_samples, rank, world_size)
    ready_path = Path(root) / _ready_file(rank)
    state_path = Path(root) / _state_file(rank)
    if (
        not state_path.is_file()
        or not ready_path.is_file()
        or ready_path.stat().st_size != rows
    ):
        return [rank + row * world_size for row in range(rows)]
    ready = np.memmap(ready_path, dtype=np.uint8, mode="r", shape=(rows,))
    return [rank + row * world_size for row in np.flatnonzero(ready == 0).tolist()]


class VaeLatentCache:
    """Lazy per-process reader for an immutable completed cache."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        expected_num_samples: int | None = None,
        expected_data_fingerprint: str | None = None,
        expected_dtype: str | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Completed VAE cache manifest not found: {manifest_path}. "
                "Run scripts/precompute_vae_latents.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            expected_num_samples,
            expected_data_fingerprint,
            expected_dtype,
        )

        self.num_samples = int(self.manifest["num_samples"])
        self.world_size = int(self.manifest["world_size"])
        self.latent_shape = tuple(int(x) for x in self.manifest["latent_shape"])
        self.num_video_frames = int(self.manifest["num_video_frames"])
        self.torch_dtype = SUPPORTED_DTYPES[self.manifest["dtype"]]
        self.vae_model_contract = self.manifest["vae_model_contract"]
        self.vae_model_fingerprint = str(self.manifest["vae_model_fingerprint"])
        self._pid: int | None = None
        self._latents: dict[int, np.memmap] = {}
        self._padding: dict[int, np.memmap] = {}

    def _validate_manifest(
        self,
        expected_num_samples: int | None,
        expected_data_fingerprint: str | None,
        expected_dtype: str | None,
    ) -> None:
        required = {
            "cache_version",
            "complete",
            "num_samples",
            "world_size",
            "latent_shape",
            "num_video_frames",
            "dtype",
            "data_contract",
            "data_fingerprint",
            "vae_model_contract",
            "vae_model_fingerprint",
        }
        missing = sorted(required - set(self.manifest))
        if missing:
            raise ValueError(f"VAE cache manifest is missing fields: {missing}")
        if int(self.manifest["cache_version"]) != CACHE_VERSION:
            raise ValueError(
                f"Unsupported VAE cache version {self.manifest['cache_version']}; "
                f"expected {CACHE_VERSION}."
            )
        if self.manifest["complete"] is not True:
            raise ValueError("VAE cache manifest is not marked complete.")
        if self.manifest["dtype"] not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported cached dtype: {self.manifest['dtype']}")
        if expected_dtype is not None and self.manifest["dtype"] != expected_dtype:
            raise ValueError(
                "VAE cache dtype mismatch: "
                f"cache={self.manifest['dtype']} current={expected_dtype}."
            )
        recorded_data_fingerprint = stable_json_hash(self.manifest["data_contract"])
        if self.manifest["data_fingerprint"] != recorded_data_fingerprint:
            raise ValueError(
                "VAE cache manifest data contract hash mismatch: "
                f"recorded={self.manifest['data_fingerprint']} "
                f"computed={recorded_data_fingerprint}."
            )
        recorded_vae_fingerprint = stable_json_hash(self.manifest["vae_model_contract"])
        if self.manifest["vae_model_fingerprint"] != recorded_vae_fingerprint:
            raise ValueError(
                "VAE cache manifest model contract hash mismatch: "
                f"recorded={self.manifest['vae_model_fingerprint']} "
                f"computed={recorded_vae_fingerprint}."
            )
        if expected_num_samples is not None and int(self.manifest["num_samples"]) != int(expected_num_samples):
            raise ValueError(
                "VAE cache dataset length mismatch: "
                f"cache={self.manifest['num_samples']} current={expected_num_samples}."
            )
        if (
            expected_data_fingerprint is not None
            and self.manifest["data_fingerprint"] != expected_data_fingerprint
        ):
            raise ValueError(
                "VAE cache preprocessing/data fingerprint mismatch: "
                f"cache={self.manifest['data_fingerprint']} current={expected_data_fingerprint}."
            )

        num_samples = int(self.manifest["num_samples"])
        world_size = int(self.manifest["world_size"])
        latent_shape = tuple(int(x) for x in self.manifest["latent_shape"])
        num_video_frames = int(self.manifest["num_video_frames"])
        if (
            num_samples <= 0
            or world_size <= 0
            or num_samples < world_size
            or not latent_shape
            or num_video_frames <= 1
        ):
            raise ValueError("VAE cache manifest contains invalid dimensions.")
        for rank in range(world_size):
            rows = _shard_size(num_samples, rank, world_size)
            _validate_shard_file_sizes(
                self.root,
                num_samples=num_samples,
                rank=rank,
                world_size=world_size,
                latent_shape=latent_shape,
                num_video_frames=num_video_frames,
            )
            ready = np.memmap(self.root / _ready_file(rank), dtype=np.uint8, mode="r", shape=(rows,))
            if rows and not bool(np.all(ready == 1)):
                raise ValueError(f"VAE cache shard {rank} contains unfinished rows.")

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return
        self._pid = pid
        self._latents = {}
        self._padding = {}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"VAE cache index {index} out of bounds for {self.num_samples} samples.")
        self._ensure_open()
        rank = index % self.world_size
        row = index // self.world_size
        rows = _shard_size(self.num_samples, rank, self.world_size)
        if rank not in self._latents:
            self._latents[rank] = np.memmap(
                self.root / _latent_file(rank),
                dtype=np.uint16,
                mode="r",
                shape=(rows, *self.latent_shape),
            )
            self._padding[rank] = np.memmap(
                self.root / _padding_file(rank),
                dtype=np.uint8,
                mode="r",
                shape=(rows, self.num_video_frames),
            )
        latent_bits = np.array(self._latents[rank][row], copy=True)
        latent = torch.from_numpy(latent_bits).view(self.torch_dtype)
        image_is_pad = torch.from_numpy(np.array(self._padding[rank][row], copy=True)).bool()
        return latent, image_is_pad


class VaeLatentShardWriter:
    """Writes one disjoint ``index % world_size == rank`` cache shard."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        num_samples: int,
        rank: int,
        world_size: int,
        latent_shape: Iterable[int],
        num_video_frames: int,
        dtype: str,
        data_fingerprint: str,
        vae_model_fingerprint: str,
        resume: bool,
    ) -> None:
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported cached dtype: {dtype}")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.num_samples = int(num_samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.latent_shape = tuple(int(x) for x in latent_shape)
        self.num_video_frames = int(num_video_frames)
        self.dtype = dtype
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"Invalid shard rank/world_size: rank={self.rank}, world_size={self.world_size}."
            )
        if self.num_samples < self.world_size:
            raise ValueError("num_samples must be at least world_size.")
        if not self.latent_shape or any(dimension <= 0 for dimension in self.latent_shape):
            raise ValueError(f"latent_shape must contain positive dimensions: {self.latent_shape}.")
        if self.num_video_frames <= 1:
            raise ValueError("num_video_frames must be greater than one.")
        self.rows = _shard_size(self.num_samples, self.rank, self.world_size)
        if self.rows <= 0:
            raise ValueError(
                f"Writer rank {self.rank} owns no rows; require num_samples >= world_size."
            )
        state_path = self.root / _state_file(rank)
        expected_state = {
            "cache_version": CACHE_VERSION,
            "num_samples": self.num_samples,
            "rank": self.rank,
            "world_size": self.world_size,
            "latent_shape": list(self.latent_shape),
            "num_video_frames": self.num_video_frames,
            "dtype": self.dtype,
            "data_fingerprint": str(data_fingerprint),
            "vae_model_fingerprint": str(vae_model_fingerprint),
        }
        mode = "w+"
        if resume and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state != expected_state:
                raise ValueError(
                    f"VAE cache shard state mismatch for rank {rank}: "
                    f"expected={expected_state} actual={state}."
                )
            required_paths = (
                self.root / _latent_file(rank),
                self.root / _padding_file(rank),
                self.root / _ready_file(rank),
            )
            if not all(path.is_file() for path in required_paths):
                raise ValueError(f"VAE cache shard {rank} is missing data files.")
            mode = "r+"
        self.latents = np.memmap(
            self.root / _latent_file(rank),
            dtype=np.uint16,
            mode=mode,
            shape=(self.rows, *self.latent_shape),
        )
        self.padding = np.memmap(
            self.root / _padding_file(rank),
            dtype=np.uint8,
            mode=mode,
            shape=(self.rows, self.num_video_frames),
        )
        self.ready = np.memmap(
            self.root / _ready_file(rank), dtype=np.uint8, mode=mode, shape=(self.rows,)
        )
        self._pending_rows: set[int] = set()
        if mode == "w+":
            self.ready[:] = 0
            self.flush()
            _atomic_json_dump(expected_state, state_path)

    def missing_indices(self) -> list[int]:
        return [
            self.rank + row * self.world_size
            for row in np.flatnonzero(self.ready == 0).tolist()
        ]

    def write(self, indices: torch.Tensor, latents: torch.Tensor, image_is_pad: torch.Tensor) -> None:
        if indices.ndim != 1:
            raise ValueError(f"Expected 1D sample indices, got shape {tuple(indices.shape)}.")
        if indices.dtype != torch.long:
            raise ValueError(f"Expected int64 sample indices, got {indices.dtype}.")
        if indices.shape[0] != latents.shape[0]:
            raise ValueError(
                "Sample index batch size mismatch: "
                f"indices={indices.shape[0]} latents={latents.shape[0]}."
            )
        if latents.dtype != SUPPORTED_DTYPES[self.dtype]:
            raise ValueError(f"Expected latent dtype {self.dtype}, got {latents.dtype}.")
        if tuple(latents.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Latent shape mismatch: expected [B,{self.latent_shape}], got {tuple(latents.shape)}."
            )
        if tuple(image_is_pad.shape) != (latents.shape[0], self.num_video_frames):
            raise ValueError(
                "image_is_pad shape mismatch: "
                f"expected {(latents.shape[0], self.num_video_frames)}, got {tuple(image_is_pad.shape)}."
            )
        if image_is_pad.dtype != torch.bool:
            raise ValueError(f"Expected boolean image_is_pad, got {image_is_pad.dtype}.")
        if not bool(torch.isfinite(latents.float()).all().item()):
            raise ValueError("Refusing to cache NaN/Inf VAE latents.")

        indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if np.any(indices_np < 0) or np.any(indices_np >= self.num_samples):
            raise ValueError(
                f"Sample indices must be in [0, {self.num_samples}); got {indices_np.tolist()}."
            )
        if len(np.unique(indices_np)) != len(indices_np):
            raise ValueError(f"Duplicate sample indices in one cache write: {indices_np.tolist()}.")
        rows = indices_np // self.world_size
        if np.any(indices_np % self.world_size != self.rank):
            raise ValueError(f"Writer rank {self.rank} received indices owned by another shard.")
        latent_bits = latents.detach().contiguous().cpu().view(torch.uint16).numpy()
        self.latents[rows] = latent_bits
        self.padding[rows] = image_is_pad.detach().cpu().numpy().astype(np.uint8, copy=False)
        self._pending_rows.update(int(row) for row in rows.tolist())

    def flush(self) -> None:
        # Commit readiness only after both payload files are durable. A crash
        # before this point merely recomputes the pending rows on resume.
        self.latents.flush()
        self.padding.flush()
        if self._pending_rows:
            self.ready[list(self._pending_rows)] = 1
        self.ready.flush()
        self._pending_rows.clear()


def finalize_cache(
    root: str | os.PathLike[str],
    *,
    num_samples: int,
    world_size: int,
    latent_shape: Iterable[int],
    num_video_frames: int,
    dtype: str,
    data_contract: dict[str, Any],
    vae_model_contract: dict[str, Any],
) -> dict[str, Any]:
    root_path = Path(root)
    latent_shape = tuple(int(x) for x in latent_shape)
    data_fingerprint = stable_json_hash(data_contract)
    vae_model_fingerprint = stable_json_hash(vae_model_contract)
    for rank in range(world_size):
        expected_state = {
            "cache_version": CACHE_VERSION,
            "num_samples": int(num_samples),
            "rank": rank,
            "world_size": int(world_size),
            "latent_shape": list(latent_shape),
            "num_video_frames": int(num_video_frames),
            "dtype": dtype,
            "data_fingerprint": data_fingerprint,
            "vae_model_fingerprint": str(vae_model_fingerprint),
        }
        state = load_shard_state(root_path, rank)
        if state != expected_state:
            raise RuntimeError(
                f"Cannot finalize VAE cache: shard {rank} state mismatch. "
                f"expected={expected_state} actual={state}."
            )
        rows = _shard_size(num_samples, rank, world_size)
        _validate_shard_file_sizes(
            root_path,
            num_samples=num_samples,
            rank=rank,
            world_size=world_size,
            latent_shape=latent_shape,
            num_video_frames=num_video_frames,
            error_type=RuntimeError,
        )
        ready = np.memmap(root_path / _ready_file(rank), dtype=np.uint8, mode="r", shape=(rows,))
        if rows and not bool(np.all(ready == 1)):
            raise RuntimeError(f"Cannot finalize VAE cache: shard {rank} is incomplete.")
    manifest = {
        "cache_version": CACHE_VERSION,
        "complete": True,
        "num_samples": int(num_samples),
        "world_size": int(world_size),
        "latent_shape": list(latent_shape),
        "num_video_frames": int(num_video_frames),
        "dtype": dtype,
        "data_contract": data_contract,
        "data_fingerprint": data_fingerprint,
        "vae_model_contract": vae_model_contract,
        "vae_model_fingerprint": str(vae_model_fingerprint),
    }
    _atomic_json_dump(manifest, root_path / MANIFEST_NAME)
    return manifest
