"""Configuration and explicit access to the existing replay implementation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
SIDES = ('left', 'right')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def encode(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=encode, ensure_ascii=False,
                                    allow_nan=False, indent=2) + '\n')


def load_config(path=ROOT/'config.json'):
    path = Path(path).resolve()
    cfg = read_json(path)
    for key in ('replay_root', 'hand_model_root', 'replay_config', 'runtime_config', 'output_root'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg['_path'] = str(path)
    if cfg['execution']['mode'] != 'retimed_replay':
        raise ValueError('Only explicitly retimed, stop/observe/plan/execute control is implemented')
    if cfg['execution'].get('collision_check_policy') != 'disabled_by_operator':
        raise ValueError('collision_check_policy must explicitly be disabled_by_operator')
    for key in ('maximum_observation_age_s', 'max_tip_fit_m', 'max_point_fit_m',
                'max_root_translation_m', 'max_root_rotation_rad'):
        value = cfg['execution'][key]
        if key == 'max_root_translation_m' and value is None: continue
        if not np.isfinite(value) or value <= 0: raise ValueError('Invalid '+key)
    if not 0 < cfg['execution']['speed'] <= 1.: raise ValueError('speed must be (0, 1]')
    lateness = cfg['execution'].get('maximum_tick_lateness_s')
    if lateness is not None and (not np.isfinite(lateness) or lateness <= 0):
        raise ValueError('maximum_tick_lateness_s must be finite and positive')
    horizon = cfg['execution'].get('prediction_horizon_s')
    if horizon is not None and (not np.isfinite(horizon) or horizon <= 0):
        raise ValueError('prediction_horizon_s must be finite and positive')
    return cfg


def replay_imports(cfg):
    root = Path(cfg['replay_root'])
    if not (root/'aubo_arms_replay.py').is_file(): raise ValueError('Missing replay_root')
    for path in (root, root/'vendor/replay_ik/src', root/'vendor/pyaubo_sdk_0_26',
                 root/'vendor/wuji_sdk_2026_11_9'):
        sys.path.insert(0, str(path))
    return root


def transform(value):
    value = np.asarray(value, float)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError('Expected finite transform (...,4,4)')
    r = value[..., :3, :3]
    if not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-8) or not np.allclose(
        np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('Invalid rigid transform')
    return value


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
