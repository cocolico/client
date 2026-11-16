#!/usr/bin/env python3
"""XFlow → Wuji21 IK → checked Aubo/Wuji replay; motion only with run --execute."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import concurrent.futures
import os
import signal
import time
import uuid
import numpy as np
from common import ROOT, SIDES, load_config, read_json, write_json, replay_imports
from hand_kinematics import HandModel, retarget_chunk
from network import Client, connection


def save_observation(path, frame):
    arrays = {k:frame[k] for k in ('head_rgb','history_rgb','keypoints') if k in frame}
    arrays['sample_ns'] = np.array(frame['sample_ns'],np.int64)
    for key in ('history_delta_s','history_error_s'):
        if key in frame: arrays[key] = np.array(frame[key])
    for side in SIDES:
        for key in ('q','camera_T_root','arm_q'):
            arrays[side+'_'+key] = frame['initial'][side][key]
    np.savez_compressed(path,**arrays)


def load_observation(path):
    with np.load(path,allow_pickle=False) as z:
        frame = {k:z[k].copy() for k in ('head_rgb','history_rgb','keypoints') if k in z}
        # Recorded timestamps belong to another process/session: fresh request identity.
        frame['recorded_sample_ns'] = int(z['sample_ns']) if 'sample_ns' in z else None
        frame['sample_ns'] = time.monotonic_ns()
        frame['initial'] = {s:{k:z[s+'_'+k].copy() for k in ('q','camera_T_root','arm_q')} for s in SIDES}
        return frame


def control_prediction_timing(prediction,execution):
    """Override only forecast timestamps after validating the raw wire contract."""
    duration = execution.get('prediction_horizon_s')
    if duration is None: return prediction
    raw = np.asarray(prediction['keypoint_time_s'],float)
    if raw.shape != (32,) or not np.isfinite(raw).all() or np.any(np.diff(raw)<=0) or raw[0]<=0:
        raise ValueError('Invalid raw prediction timestamps')
    if not np.isfinite(duration) or duration <= 0: raise ValueError('Invalid control forecast horizon')
    return dict(prediction,keypoint_time_s=raw*(float(duration)/raw[-1]))


def infer_and_retarget(cfg,client,models,frame,out,steps):
    observation = {k:frame[k] for k in ('head_rgb','history_rgb','keypoints') if k in frame}
    save_observation(out/'observation.npz',frame)
    prediction = client.infer(observation,frame['sample_ns'])
    arrays = {k:v for k,v in prediction.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(out/'prediction.npz',**arrays)
    write_json(out/'prediction.json',{k:v for k,v in prediction.items() if k not in arrays})
    control_prediction = control_prediction_timing(prediction,cfg['execution'])
    write_json(out/'prediction_timing.json',dict(
        source='client_horizon_override' if control_prediction is not prediction else 'server',
        raw_keypoint_time_s=prediction['keypoint_time_s'],
        control_keypoint_time_s=control_prediction['keypoint_time_s'],
        time_scale=float(control_prediction['keypoint_time_s'][-1]/prediction['keypoint_time_s'][-1]),
        requested_replay_speed=cfg['execution']['speed']))
    chunk = retarget_chunk(models,control_prediction,frame['initial'],cfg['execution'],steps)
    saved = dict(time_s=chunk['time_s'])
    for side in SIDES:
        saved.update({side+'_'+k:v for k,v in chunk[side].items() if k != 'diagnostics'})
    np.savez_compressed(out/'retarget.npz',**saved)
    write_json(out/'retarget.json',{s:chunk[s]['diagnostics'] for s in SIDES})
    print('RETARGET PASS',out,flush=True)
    return chunk


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config.json')
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('info',help='SSH + info/reset only; no robot connections')
    sub.add_parser('doctor',help='Local dependency/config check; no hardware or network')
    sub.add_parser('camera',help='Read configured network/ROS camera and calibration; no robot or model connections')
    snap = sub.add_parser('snapshot',help='Read current arm/hand state and FK only')
    infer = sub.add_parser('infer',help='Recorded observation → real server → offline retarget')
    infer.add_argument('--observation',type=Path,required=True)
    infer.add_argument('--steps',type=int,default=32)
    run = sub.add_parser('run',help='Live observation + complete replay preflight; execute only by flag')
    run.add_argument('--execute',action='store_true')
    run.add_argument('--iterations',type=int,default=1,help='positive number of full closed-loop cycles')
    run.add_argument('--steps',type=int,default=32)
    args = p.parse_args(argv); cfg = load_config(args.config)
    if hasattr(args,'steps') and not 1 <= args.steps <= 32: p.error('--steps must be 1..32')
    if hasattr(args,'iterations') and args.iterations < 1: p.error('--iterations must be positive')
    if args.command == 'run' and not args.execute and args.iterations != 1:
        p.error('Read-only preflight supports one cycle; multiple cycles require --execute')
    if args.command == 'doctor':
        from doctor import doctor
        doctor(cfg); return
    models = {s:HandModel(cfg['hand_model_root'],s) for s in SIDES}
    session = Path(cfg['output_root'])/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    session.mkdir(parents=True,exist_ok=True)
    write_json(session/'configuration.json',cfg)
    print('OUTPUT',session,flush=True)
    state = dict(pid=os.getpid(),command=args.command,execution_requested=bool(getattr(args,'execute',False)),status='starting')
    def status(value,**extra):
        state.update(status=value,**extra); write_json(session/'session.json',state)
    status('starting')
    try:
        with ExitStack() as stack:
            if args.command == 'camera':
                replay_imports(cfg)
                from observation import create_camera, check_camera_info
                from replay_pipeline import config
                camera = create_camera(cfg['camera']); stack.callback(camera.close)
                frame = camera.capture(4,time.monotonic_ns())
                write_json(session/'camera_info.json',frame['camera_info'])
                check_camera_info(cfg['camera'],frame['camera_info'],config(cfg['replay_config']))
                if cfg['camera']['geometry_mode'] == 'canonical_cam0':
                    from camera_geometry import Cam0Geometry
                    geometry = Cam0Geometry(frame['camera_info'],read_json(ROOT/'vendor/cam0_color_camera_info.json'))
                    np.savez_compressed(session/'camera_frames.npz',raw_rgb=frame['head_rgb'],head_rgb=geometry.apply(frame['head_rgb']))
                print('CAMERA PASS',session/'camera_info.json',flush=True)
                status('completed',read_only=True); return
            if args.command in ('snapshot','run'):
                replay_imports(cfg)
                from replay_control import acquire_replay_lock
                from observation import Hardware
                lock = acquire_replay_lock(); stack.callback(lock.close)
                if args.command == 'snapshot':
                    hardware = Hardware(cfg,models); stack.callback(hardware.close)
                    snapshot = hardware.measured(); write_json(session/'snapshot.json',snapshot)
                    status('completed',read_only=True); print('SNAPSHOT SAVED',session/'snapshot.json'); return
            host,port = stack.enter_context(connection(cfg['server']))
            client = Client(host,port,cfg['server']); stack.callback(client.close)
            write_json(session/'server_info.json',client.info)
            if args.command == 'info':
                print(client.info); status('completed',read_only=True); return
            if args.command == 'infer':
                frame = load_observation(args.observation)
                infer_and_retarget(cfg,client,models,frame,session,args.steps)
                status('completed',read_only=True); return
            from observation import create_camera
            from planning import plan_chunk, execute_plan, check_age
            camera = create_camera(cfg['camera']); stack.callback(camera.close)
            # Native camera initialization can hold the GIL for seconds. Finish it
            # before subscribing to high-rate Wuji state/diagnostics.
            hardware = Hardware(cfg,models); stack.callback(hardware.close)
            if cfg.get('monitor',{}).get('enabled',False):
                from live_compare import LiveCompare
                try:
                    hardware.live_compare = LiveCompare(cfg,hardware,camera,models,session)
                    stack.callback(hardware.live_compare.close)
                except Exception as error:
                    print('WARNING live comparison unavailable:',error,flush=True)
            if args.execute and cfg['execution'].get('persistent_servo',False):
                def prepare_cycle(iteration):
                    out = session/f'cycle_{iteration:04d}'; out.mkdir()
                    begin = time.monotonic()
                    status('observing',iteration=iteration)
                    frame = hardware.capture(camera,client.spec['history_offset_frames'])
                    observed = time.monotonic()
                    write_json(out/'camera_info.json',frame['camera_info'])
                    if frame.get('warnings'): write_json(out/'warnings.json',frame['warnings'])
                    status('inference_and_retarget',iteration=iteration)
                    chunk = infer_and_retarget(cfg,client,models,frame,out,args.steps)
                    inferred = time.monotonic()
                    status('planning',iteration=iteration)
                    prepared = plan_chunk(cfg,hardware,models,chunk,frame,out)
                    write_json(out/'timing.json',dict(observation_s=observed-begin,
                        inference_and_retarget_s=inferred-observed,planning_s=time.monotonic()-inferred,
                        preparation_total_s=time.monotonic()-begin,persistent_servo=True))
                    print('PREFLIGHT PASS',out/'runtime.json',flush=True)
                    return frame,prepared
                first_frame,first_plan = prepare_cycle(0)
                worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                pending = None; next_iteration = 1
                def next_chunk():
                    nonlocal pending,next_iteration
                    if next_iteration >= args.iterations: return False
                    if pending is None:
                        pending = worker.submit(prepare_cycle,next_iteration)
                    if not pending.done(): return None
                    frame,prepared = pending.result()
                    check_age(frame['sample_ns'],cfg['execution']['maximum_observation_age_s'])
                    pending = None; next_iteration += 1
                    status('executing',iteration=next_iteration-1)
                    return prepared
                try:
                    status('executing',iteration=0)
                    execute_plan(cfg,hardware,first_frame,first_plan,loop_hook=next_chunk)
                finally:
                    if hasattr(hardware,'cancel_wait'): hardware.cancel_wait.set()
                    worker.shutdown(wait=True,cancel_futures=True)
                status('completed',read_only=False); return
            for iteration in range(args.iterations):
                out = session/f'cycle_{iteration:04d}'; out.mkdir()
                cycle_started = time.monotonic(); timings = {}
                status('observing',iteration=iteration)
                frame = hardware.capture(camera,client.spec['history_offset_frames'])
                timings['observation_s'] = time.monotonic()-cycle_started
                write_json(out/'camera_info.json',frame['camera_info'])
                if frame.get('warnings'): write_json(out/'warnings.json',frame['warnings'])
                status('inference_and_retarget',iteration=iteration)
                stage_started = time.monotonic()
                chunk = infer_and_retarget(cfg,client,models,frame,out,args.steps)
                timings['inference_and_retarget_s'] = time.monotonic()-stage_started
                check_age(frame['sample_ns'],cfg['execution']['maximum_observation_age_s'])
                status('planning',iteration=iteration)
                stage_started = time.monotonic()
                prepared = plan_chunk(cfg,hardware,models,chunk,frame,out)
                timings['planning_s'] = time.monotonic()-stage_started
                print('PREFLIGHT PASS',out/'runtime.json',flush=True)
                if args.execute:
                    status('executing',iteration=iteration)
                    stage_started = time.monotonic()
                    execute_plan(cfg,hardware,frame,prepared)
                    timings['execution_s'] = time.monotonic()-stage_started
                else: print('Read-only preflight complete; no motion commands sent.',flush=True)
                timings['cycle_total_s'] = time.monotonic()-cycle_started
                write_json(out/'timing.json',timings)
                print('CYCLE TIMING',timings,flush=True)
            status('completed',read_only=not args.execute)
    except BaseException as error:
        status('failed',error=str(error),error_type=type(error).__name__); raise


if __name__ == '__main__':
    def interrupt(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    try: main()
    except (ValueError,RuntimeError,OSError,TimeoutError,KeyError) as error:
        print(f'BLOCKED: {error}',flush=True); raise SystemExit(2)
    except (KeyboardInterrupt,InterruptedError):
        print('Stopped; inspect session and replay execution logs for cleanup status.',flush=True)
        raise SystemExit(130)
