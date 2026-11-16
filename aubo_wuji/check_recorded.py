#!/usr/bin/env python3
"""Real model + recorded flower observations; never connects robot hardware.

Reports model-vs-recorded error, then fits current recorded points to Wuji and
retargets the predicted chunk. The inferred initial hand pose is for offline
diagnostics only; it is not a measured robot start or an execution plan.
"""
import argparse
from pathlib import Path
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT, SIDES, load_config, write_json, fingerprint
from hand_kinematics import HandModel, normalized_goals, retarget, retarget_chunk
from network import Client, connection
from run import infer_and_retarget


def align(a,b):
    u,_,vh = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    fix = np.eye(3); fix[2,2] = np.linalg.det(vh.T @ u.T)
    r = vh.T @ fix @ u.T
    out = np.eye(4); out[:3,:3] = r; out[:3,3] = b.mean(0)-r @ a.mean(0)
    return out


def initial_from_recording(models,keypoints,limits):
    initial = {}
    for si,side in enumerate(SIDES):
        hand = models[side]; neutral = hand.fk(hand.neutral)
        goals = normalized_goals(keypoints[si],neutral)
        palm = [1,5,9,13,17]
        seed = align(neutral[palm],goals[palm])
        # Only the initialization uses a wider envelope: it recovers a recorded
        # pose from neutral, not a motion target. Actual predictions keep config limits.
        initialization = dict(limits,max_root_translation_m=.5,max_root_rotation_rad=np.pi)
        fitted = retarget(hand,keypoints[si][None],hand.neutral,seed,initialization)
        initial[side] = dict(q=fitted['joints'][0],camera_T_root=fitted['poses'][0],arm_q=np.zeros(6))
    return initial


def compare_recording(prediction,observation,raw):
    future = np.concatenate([observation['keypoints'][None],raw['future_keypoints_30fps']],axis=0)
    ticks = np.asarray(prediction['keypoint_time_s'],float)*30
    if ticks.max() > len(future)-1: raise ValueError('Not enough recorded future data')
    lo = np.floor(ticks).astype(int); hi = np.minimum(lo+1,len(future)-1)
    weight = (ticks-lo)[:,None,None,None]
    truth = future[lo]*(1-weight)+future[hi]*weight
    error = np.linalg.norm(prediction['keypoints']-truth,axis=-1)*1000
    baseline = np.linalg.norm(observation['keypoints'][None]-truth,axis=-1)*1000
    return dict(recorded_mpjpe_mm=float(error.mean()),hold_pose_mpjpe_mm=float(baseline.mean()),
                recorded_max_error_mm=float(error.max()),interpretation='Single recorded window; not robot task-success evidence')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--n',type=int,default=3)
    args = p.parse_args()
    if args.n < 1: p.error('--n must be positive')
    cfg = load_config(args.config)
    with np.load(args.observation,allow_pickle=False) as z: raw = {k:z[k].copy() for k in z.files}
    if str(raw['task']) != cfg['server']['task']: raise ValueError('Recorded task mismatch')
    observation = {k:raw[k] for k in ('head_rgb','history_rgb','keypoints')}
    directory = Path(cfg['output_root'])/('real_model_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    report = dict(execution_attempted=False,recorded_input=True,source=fingerprint(args.observation),
                  task=cfg['server']['task'],model=cfg['server']['model'],round_trips=[],retarget_passed=False)
    print('OUTPUT',directory,flush=True)
    try:
        with connection(cfg['server']) as (host,port):
            client = Client(host,port,cfg['server'])
            try:
                write_json(directory/'server_info.json',client.info)
                if int(raw['history_offset_frames']) != client.spec['history_offset_frames']:
                    raise ValueError('Recorded history offset differs from model')
                for i in range(args.n):
                    started = time.monotonic()
                    prediction = client.infer(observation,time.monotonic_ns())
                    seconds = time.monotonic()-started
                    quality = compare_recording(prediction,observation,raw)
                    report['round_trips'].append(dict(seconds=seconds,**quality))
                    print('MODEL PASS',i,round(seconds,3),'s',quality,flush=True)
                    np.savez_compressed(directory/f'prediction_{i:02d}.npz',
                        **{k:v for k,v in prediction.items() if isinstance(v,np.ndarray)})
                    write_json(directory/f'prediction_{i:02d}.json',
                        {k:v for k,v in prediction.items() if not isinstance(v,np.ndarray)})
                report.update(deployment_id=client.spec['deployment_id'],weights=client.spec['weights'])
                models = {s:HandModel(cfg['hand_model_root'],s) for s in SIDES}
                initial = initial_from_recording(models,raw['keypoints'],cfg['execution'])
                chunk = retarget_chunk(models,prediction,initial,cfg['execution'])
                arrays = dict(time_s=chunk['time_s'],test_only=np.array(True))
                for side in SIDES:
                    arrays.update({side+'_'+k:v for k,v in chunk[side].items() if k!='diagnostics'})
                np.savez_compressed(directory/'wuji_retarget.npz',**arrays)
                report['retarget_passed'] = True
                report['maximum_tip_fit_mm'] = {s:float(chunk[s]['fit_error_m'][:,[4,8,12,16,20]].max()*1000) for s in SIDES}
                print('RECORDED → REAL MODEL → WUJI RETARGET PASS',flush=True)
            finally: client.close()
    except BaseException as error:
        report['error'] = str(error); raise
    finally: write_json(directory/'validation.json',report)


if __name__ == '__main__': main()
