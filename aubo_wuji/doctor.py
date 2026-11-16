"""Local-only diagnostics; does not connect cameras, controllers or remote hosts."""
import importlib.util
from pathlib import Path
import numpy as np
from common import ROOT, SIDES, read_json, replay_imports
from hand_kinematics import HandModel


def doctor(cfg):
    root = replay_imports(cfg)
    from replay_pipeline import config
    c = config(cfg['replay_config'])
    from camera_geometry import Cam0Geometry
    dependencies = {name:bool(importlib.util.find_spec(name)) for name in
        ('numpy','scipy','cv2','pinocchio','coal','rclpy','sensor_msgs','pyaubo_sdk','wuji_sdk')}
    print('DEPENDENCIES',dependencies)
    if not all(dependencies.values()): raise ValueError('Missing runtime dependencies')
    if cfg['camera'].get('backend') == 'orbbec_net':
        from observation import load_orbbec
        sdk = load_orbbec()
        print('NETWORK CAMERA SDK',sdk.__version__,'IP',cfg['camera']['ip'])
    expected = read_json(ROOT/'vendor/cam0_color_camera_info.json')
    for side in SIDES:
        h = HandModel(cfg['hand_model_root'],side)
        collision = Path(c['source_episode_root'])/'assets/wuji/urdf'/f'{side}.urdf'
        if collision.read_bytes() != h.path.read_bytes(): raise ValueError('Hand model mismatch')
        meta = read_json(c['sides'][side]['handeye'])['metadata']
        source = dict(width=meta['image_size'][1],height=meta['image_size'][0],K=meta['K'],D=meta['D'])
        geometry = Cam0Geometry(source,expected)
        print(side,'20 joints / 21 points; canonical cam0 ray coverage',round(geometry.coverage,6))
        print(side,'Aubo',c['sides'][side]['ip'],'handeye',c['sides'][side]['handeye'])
    print('MODEL',cfg['server']['model'],'RUN',cfg['server']['expected_run_name'],'TASK',cfg['server']['task'])
    print('PASS local files/imports only. Live camera, network, robot state and task performance are not verified.')
