"""Explicit hardware read/IK/collision smoke test using a synthetic TCP policy.

No camera or GPU model is used. Never enables motors or calls execute_plan.
Run only intentionally: ./python tests/live_readonly.py --live-readonly
"""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import SIDES, load_config, write_json, replay_imports
from hand_kinematics import HandModel, camera_points
from observation import Hardware
from planning import plan_chunk
from run import infer_and_retarget
from network import Client
from test_pipeline import served


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live-readonly',action='store_true',required=True)
    p.parse_args()
    cfg = load_config(); replay_imports(cfg)
    from replay_control import acquire_replay_lock
    lock = acquire_replay_lock(); hardware = None
    session = Path(cfg['output_root'])/('live_readonly_'+time.strftime('%Y%m%d_%H%M%S'))
    out = session/'cycle_0000'; out.mkdir(parents=True)
    report = dict(execution_attempted=False,synthetic_policy=True,synthetic_images=True,passed=False)
    try:
        models = {s:HandModel(cfg['hand_model_root'],s) for s in SIDES}
        hardware = Hardware(cfg,models); snapshot = hardware.measured()
        initial = {}; points = []
        for side in SIDES:
            q = np.asarray(snapshot['hands'][side]['positions_rad'])
            arm_q = np.asarray(snapshot['controllers'][side]['joint_positions_rad'])
            root = np.linalg.inv(hardware.base_T_camera[side]) @ hardware.arms[side].fk(arm_q) @ np.asarray(hardware.replay['sides'][side]['flange_T_wuji_base'])
            initial[side] = dict(q=q,arm_q=arm_q,camera_T_root=root)
            points.append(camera_points(models[side],q,root))
        frame = dict(head_rgb=np.zeros((400,640,3),np.uint8),history_rgb=np.zeros((400,640,3),np.uint8),
                     keypoints=np.asarray(points,np.float32),initial=initial,sample_ns=time.monotonic_ns())
        with served() as server:
            client = Client(server.host,server.port,cfg['server'])
            try: chunk = infer_and_retarget(cfg,client,models,frame,out,32)
            finally: client.close()
        _,_,trajectory,plan = plan_chunk(cfg,hardware,models,chunk,frame,out)
        report.update(passed=True,ticks=len(trajectory),duration_s=plan['duration_s'],
            ready_for_execution=plan['ready_for_execution'],collision_passed=plan['collision_passed'])
        # A synthetic test must not leave a reusable execution authorization.
        plan.update(ready_for_execution=False,test_only=True,synthetic_policy=True)
        write_json(out/'runtime.json',plan)
        print('READ-ONLY SDK PREFLIGHT PASS',out,flush=True)
    except BaseException as error:
        report['error'] = str(error); raise
    finally:
        write_json(session/'validation.json',report)
        if hardware is not None: hardware.close()
        lock.close()


if __name__ == '__main__': main()
