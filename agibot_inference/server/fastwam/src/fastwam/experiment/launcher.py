"""Strict, metadata-recording HTrain launcher for versioned experiments."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .config import config_fingerprint, resolved_yaml
from .data_prep import prepare_data


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def require_clean_commit(repo_root: Path) -> str:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise RuntimeError(
            "Refusing to launch from a dirty worktree. Commit the exact experiment "
            f"snapshot first. Dirty entries:\n{status}"
        )
    return _git(repo_root, "rev-parse", "HEAD")


def resolve_hydra(repo: Path, python: str, task: str) -> DictConfig:
    result = subprocess.run(
        [python, "scripts/train.py", f"task={task}", "--cfg", "job", "--resolve"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return OmegaConf.create(result.stdout)


def validate_hydra_contract(resolved: DictConfig, expected: DictConfig) -> None:
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = OmegaConf.select(resolved, key)
        if actual != expected_value:
            failures.append(f"{key}: expected={expected_value!r} actual={actual!r}")
    if failures:
        raise ValueError("Resolved Hydra contract mismatch:\n" + "\n".join(failures))


def validate_artifact_contract(
    hydra_cfg: DictConfig,
    expected: DictConfig,
    base_dir: Path | None = None,
) -> None:
    """Verify configured file hashes before any training-side data preparation."""

    failures: list[str] = []
    for key, expected_sha256 in expected.items():
        configured_path = OmegaConf.select(hydra_cfg, key)
        if configured_path is None:
            failures.append(f"{key}: missing from resolved Hydra config")
            continue
        path = Path(str(configured_path))
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        try:
            actual_sha256 = _sha256(path)
        except FileNotFoundError:
            failures.append(f"{key}: file not found at {path}")
            continue
        if actual_sha256 != expected_sha256:
            failures.append(
                f"{key}: expected_sha256={expected_sha256} "
                f"actual_sha256={actual_sha256}"
            )
    if failures:
        raise ValueError("Resolved artifact contract mismatch:\n" + "\n".join(failures))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reproducibility artifact not found: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    """Record a file or a complete training-state directory."""

    if path.is_file():
        return _file_record(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Required reproducibility artifact not found: {path}")
    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entry = _file_record(child)
        entry["relative_path"] = str(child.relative_to(path))
        files.append(entry)
    if not files:
        raise FileNotFoundError(f"Training-state directory is empty: {path}")
    return {
        "path": str(path.resolve()),
        "type": "directory",
        "files": files,
    }


def _artifact_manifest(
    cfg: DictConfig,
    hydra_cfg: DictConfig,
    data_root: Path,
    cache_root: Path,
    resume_path: str,
    repo: Path | None = None,
) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    records_by_path: dict[Path, dict[str, object]] = {}

    def record(path: Path) -> dict[str, object]:
        resolved = path.resolve()
        if resolved not in records_by_path:
            records_by_path[resolved] = _artifact_record(resolved)
        return records_by_path[resolved]

    for key in (
        "data.train.pretrained_norm_stats",
        "model.action_dit_pretrained_path",
        "model.cam_action_dit_pretrained_path",
        "model.arm_action_dit_pretrained_path",
    ):
        configured_path = OmegaConf.select(hydra_cfg, key)
        if configured_path is None:
            continue
        path = Path(str(configured_path))
        if repo is not None and not path.is_absolute():
            path = repo / path
        files[key] = record(path)
    if resume_path != "null":
        files["training.resume"] = record(Path(resume_path))

    datasets: dict[str, object] = {}
    for dataset_dir in OmegaConf.select(hydra_cfg, "data.train.dataset_dirs"):
        dataset = Path(str(dataset_dir))
        info_path = dataset / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        cache_manifest = cache_root / dataset.name / "manifest.json"
        datasets[dataset.name] = {
            "root": str(dataset.resolve()),
            "fps": info["fps"],
            "total_frames": info["total_frames"],
            "info": record(info_path),
            "cache_manifest": record(cache_manifest),
        }
        downsample_manifest = dataset / "downsample_manifest.json"
        if downsample_manifest.is_file():
            datasets[dataset.name]["downsample_manifest"] = record(downsample_manifest)
    matched_ready = data_root / "MATCHED_READY.json"
    if matched_ready.is_file():
        files["matched_eef_ready"] = record(matched_ready)
    full_action_dim = OmegaConf.select(hydra_cfg, "model.full_action_dim")
    if full_action_dim is None:
        full_action_dim = OmegaConf.select(
            hydra_cfg, "model.action_dit_config.action_dim"
        )
    model_action_dimensions = {"full": int(full_action_dim)}
    for name, key in (
        ("arm", "model.arm_action_dim"),
        ("camera", "model.cam_action_dim"),
    ):
        value = OmegaConf.select(hydra_cfg, key)
        if value is not None:
            model_action_dimensions[name] = int(value)

    contracts = {
        "model_action_dimensions": model_action_dimensions,
        "data_dimensions": {
            "action": int(
                OmegaConf.select(hydra_cfg, "data.train.processor.action_output_dim")
            ),
            "proprio": int(
                OmegaConf.select(hydra_cfg, "data.train.processor.proprio_output_dim")
            ),
        },
    }
    return {"files": files, "datasets": datasets, "contracts": contracts}


def _record_metadata(
    cfg: DictConfig,
    hydra_cfg: DictConfig,
    output_dir: Path,
    commit: str,
    artifact_manifest: dict[str, object],
    hf_cache_dir: Path,
) -> None:
    metadata = output_dir / "reproducibility"
    _atomic_write(metadata / "experiment.resolved.yaml", resolved_yaml(cfg))
    _atomic_write(
        metadata / "hydra.resolved.yaml",
        OmegaConf.to_yaml(hydra_cfg, resolve=True, sort_keys=True),
    )
    _atomic_write(metadata / "git_commit.txt", f"{commit}\n")
    public_env = {
        key: os.environ.get(key, "")
        for key in ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT")
    }
    public_env["config_sha256"] = config_fingerprint(cfg)
    public_env["hf_datasets_cache"] = str(hf_cache_dir)
    _atomic_write(metadata / "launch.json", json.dumps(public_env, indent=2) + "\n")
    _atomic_write(
        metadata / "artifacts.json",
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
    )


def _hf_cache_dir(cfg: DictConfig, node_rank: int) -> Path:
    """Return a fresh node-local HF cache path for this HTrain allocation."""

    allocation = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.environ["MASTER_ADDR"])
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cfg.training.run_name))
    return Path(cfg.data.hf_cache_root) / f"{run_name}_{allocation}_node{node_rank}"


def _prewarm_hf_cache(
    repo: Path,
    python: str,
    hydra_cfg: DictConfig,
    cache_dir: Path,
    environment: dict[str, str],
) -> None:
    """Build every Parquet-to-Arrow cache once before distributed ranks start."""

    dataset_dirs = [
        str(path) for path in OmegaConf.select(hydra_cfg, "data.train.dataset_dirs")
    ]
    command = [
        python,
        "scripts/prewarm_lerobot_hf_cache.py",
        "--cache-dir",
        str(cache_dir),
    ]
    for dataset_dir in dataset_dirs:
        command.extend(["--dataset-dir", dataset_dir])
    print("[hf-cache]", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, env=environment, check=True)


def _resume_path(cfg: DictConfig) -> str:
    if not cfg.training.resume:
        return "null"
    if cfg.training.resume == "pretrain_checkpoint":
        checkpoint = Path(cfg.paths.pretrain_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint}")
        return str(checkpoint)
    checkpoint = Path(cfg.training.resume)
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    return str(checkpoint)


def launch(
    cfg: DictConfig,
    repo: Path,
    repo_root: Path,
    *,
    validate_only: bool = False,
) -> None:
    python = sys.executable
    hydra_cfg = resolve_hydra(repo, python, cfg.training.task)
    validate_hydra_contract(hydra_cfg, cfg.expected_hydra)
    validate_artifact_contract(hydra_cfg, cfg.expected_artifacts, base_dir=repo)
    if validate_only:
        print(resolved_yaml(cfg))
        print(f"config_sha256={config_fingerprint(cfg)}")
        print("Hydra and artifact contracts: OK")
        return

    commit = require_clean_commit(repo_root)
    required_env = ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [key for key in required_env if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing HTrain environment variables: {missing}")
    nodes = int(os.environ["NNODES"])
    node_rank = int(os.environ["NODE_RANK"])
    if nodes != cfg.resources.nodes:
        raise RuntimeError(f"Expected {cfg.resources.nodes} nodes, got {nodes}")

    import torch

    local_gpus = torch.cuda.device_count()
    if local_gpus != cfg.resources.gpus_per_node:
        raise RuntimeError(
            f"Expected {cfg.resources.gpus_per_node} local GPUs, got {local_gpus}"
        )
    total_processes = nodes * local_gpus
    data_root, cache = prepare_data(cfg, repo, python)
    resume_path = _resume_path(cfg)
    output_dir = Path(cfg.paths.output_root) / cfg.training.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in cfg.runtime_env.items()})
    hf_cache_dir = _hf_cache_dir(cfg, node_rank)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    environment["HF_DATASETS_CACHE"] = str(hf_cache_dir)
    _prewarm_hf_cache(repo, python, hydra_cfg, hf_cache_dir, environment)

    if node_rank == 0:
        artifacts = _artifact_manifest(
            cfg, hydra_cfg, data_root, cache, resume_path, repo=repo
        )
        _record_metadata(
            cfg, hydra_cfg, output_dir, commit, artifacts, hf_cache_dir
        )

    environment["FASTWAM_RAW_FRAME_CACHE_DIR"] = str(cache)
    environment["PYTHONPATH"] = f"{repo / 'src'}:{environment.get('PYTHONPATH', '')}"
    environment["TORCHINDUCTOR_CACHE_DIR"] = (
        f"/dev/shm/fastwam_inductor_{cfg.training.run_name}_node{node_rank}"
    )
    environment["TRITON_CACHE_DIR"] = (
        f"/dev/shm/fastwam_triton_{cfg.training.run_name}_node{node_rank}"
    )
    if cfg.wandb.enabled:
        key_file = Path(cfg.paths.wandb_key_file)
        if not key_file.is_file():
            raise FileNotFoundError(f"W&B key file not found: {key_file}")
        environment["WANDB_API_KEY"] = key_file.read_text().strip()
        environment["WANDB_MODE"] = "online"

    command = [
        python,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--config_file",
        "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
        "--num_processes",
        str(total_processes),
        "--num_machines",
        str(nodes),
        "--machine_rank",
        str(node_rank),
        "--main_process_ip",
        os.environ["MASTER_ADDR"],
        "--main_process_port",
        os.environ["MASTER_PORT"],
        "--same_network",
        "--deepspeed_multinode_launcher",
        "standard",
        "scripts/train.py",
        f"task={cfg.training.task}",
        f"resume={resume_path}",
        f"log_every={cfg.training.log_every}",
        f"save_every={cfg.training.save_every}",
        f"eval_every={cfg.training.eval_every}",
        f"wandb.enabled={str(cfg.wandb.enabled).lower()}",
        f"wandb.project={cfg.wandb.project}",
        f"wandb.name={cfg.training.run_name}",
        f"wandb.group={cfg.wandb.group}",
        "wandb.mode=online",
        f"output_dir={output_dir}",
    ]
    print(f"[repro] commit={commit} config_sha256={config_fingerprint(cfg)}")
    print("[launch]", " ".join(command), flush=True)
    os.chdir(repo)
    os.execvpe(command[0], command, environment)
