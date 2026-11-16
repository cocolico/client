"""Reuse replay IK/dynamics and synchronized execution; collision checking is operator-disabled."""
import concurrent.futures
import copy
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT, SIDES, read_json, write_json, fingerprint, replay_imports, transform


def check_age(sample_ns, limit):
    age = (time.monotonic_ns()-sample_ns)/1e9
    if age < 0 or age > limit: raise TimeoutError(f'Observation age {age:.3f}s exceeds {limit}s')
    return age


def arrays_from_chunk(chunk, base_T_camera, mounts, models):
    arrays = dict(time_s=np.asarray(chunk['time_s']))
    for side in SIDES:
        poses = transform(chunk[side]['poses'])
        arrays[side+'_source_camera_T_wuji_base'] = poses
        arrays[side+'_aubo_base_T_flange'] = transform(base_T_camera[side]) @ poses @ np.linalg.inv(transform(mounts[side]))
        arrays[side+'_wuji_joint_positions_rad'] = chunk[side]['joints']
        arrays[side+'_wuji_wire_joint_names'] = np.array([f'{side}_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)])
    return arrays


def unchanged_start(initial, controllers, hands, settings):
    warnings = []
    for side in SIDES:
        arm_delta = abs(np.asarray(controllers[side]['joint_positions_rad'])-initial[side]['arm_q'])
        arm_joint = int(np.argmax(arm_delta)); arm_max = float(arm_delta[arm_joint])
        if arm_max > settings['stationary_arm_tolerance_rad']:
            warning = dict(type='arm_moved_since_observation',side=side,
                joint_index=arm_joint,max_delta_rad=arm_max,
                threshold_rad=float(settings['stationary_arm_tolerance_rad']),action='warn_only')
            warnings.append(warning); print('WARNING',warning,flush=True)
        delta = abs(np.asarray(hands[side]['positions_rad'])-initial[side]['q'])
        joint = int(np.argmax(delta)); maximum = float(delta[joint])
        if maximum > settings['stationary_hand_tolerance_rad']:
            warning = dict(type='hand_moved_since_observation',side=side,
                joint_index=joint,max_delta_rad=maximum,
                threshold_rad=float(settings['stationary_hand_tolerance_rad']),
                action='warn_only')
            warnings.append(warning)
            print('WARNING',warning,flush=True)
    return warnings


def plan_chunk(cfg, hardware, models, chunk, frame, directory):
    root = replay_imports(cfg)
    from replay_pipeline import config, provenance, hand_contract
    from replay_control import read_arms, check_hand_controls
    from runtime_plan import check_controller
    import aubo_arms_replay as replay_planner
    SETTINGS, solve_side = replay_planner.SETTINGS, replay_planner.solve_side
    from diagnose_ik import diagnose
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    settings = dict(read_json(cfg['runtime_config']), **SETTINGS)
    settings.update(speed=cfg['execution']['speed'], arms_only=False, ik_method='episode594_sdk',
                    collision_check_policy=cfg['execution']['collision_check_policy'],
                    minimum_transition_s=0.,entry_hold_s=0.,warmup_s=0.,
                    trajectory_sampling=cfg['execution'].get('trajectory_sampling','interpolated'),
                    scheduler_lateness_policy=cfg['execution'].get('scheduler_lateness_policy','stop'),
                    maximum_tick_lateness_s=cfg['execution'].get('maximum_tick_lateness_s',SETTINGS['maximum_tick_lateness_s']),
                    arm_velocity_fraction=1.,arm_acceleration_fraction=1.)
    if 'first_step_arm_deg' in cfg['execution']:
        settings['first_step_arm_deg'] = cfg['execution']['first_step_arm_deg']
        settings['first_step_right_tolerance_deg'] = cfg['execution']['first_step_right_tolerance_deg']
    if 'servo_subdivisions' in cfg['execution']:
        settings['servo_subdivisions'] = cfg['execution']['servo_subdivisions']
    check_hand_controls(hardware.hands,settings)
    controllers = read_arms(hardware.replay); hands = hardware.hands.snapshot()
    hand_motion_warnings = unchanged_start(frame['initial'],controllers,hands,cfg['execution'])
    from read_hands import check_current_limit
    check_current_limit(hands,settings['maximum_hand_current_a'])
    c = copy.deepcopy(hardware.replay); c.pop('_config_path')
    for side in SIDES:
        checked = dict(controllers[side])
        if getattr(hardware,'owns_servo',False): checked['servo_mode_enabled'] = False
        check_controller(c,side,checked)
        c['sides'][side]['initial_q_rad'] = controllers[side]['joint_positions_rad']
        c['sides'][side]['model_confirmed'] = True
        model = Path(c['source_episode_root'])/'assets/wuji/urdf'/f'{side}.urdf'
        if model.read_bytes() != models[side].path.read_bytes(): raise ValueError('Retarget and execution hand models differ')
    write_json(directory/'config.json',c); c = config(directory/'config.json')
    arrays = arrays_from_chunk(chunk,hardware.base_T_camera,
        {s:c['sides'][s]['flange_T_wuji_base'] for s in SIDES},models)
    prepared = directory/'prepared.npz'; np.savez_compressed(prepared,**arrays)
    # Fingerprint actual code, endpoint contract and calibration. Collision
    # geometry is intentionally outside this operator-disabled execution policy.
    paths = list(ROOT.glob('*.py')) + list((ROOT/'vendor').glob('*.py'))
    paths += [Path(cfg['_path']),Path(cfg['replay_config']),Path(cfg['runtime_config']),
              ROOT/'vendor/cam0_color_camera_info.json',directory/'prediction.npz',directory/'prediction.json',
              directory/'observation.npz',directory/'retarget.npz']
    if (directory/'prediction_timing.json').exists(): paths.append(directory/'prediction_timing.json')
    if (directory/'camera_info.json').exists(): paths.append(directory/'camera_info.json')
    paths += list(root.glob('*.py'))
    for side in SIDES:
        urdf = Path(c['sides'][side]['urdf'])
        paths += [Path(c['sides'][side]['handeye']),urdf,models[side].path]
    for folder in (Path(c['source_episode_root'])/'assets/wuji',root/'vendor/pyaubo_sdk_0_26',root/'vendor/wuji_sdk_2026_11_9'):
        paths += [p for p in folder.rglob('*') if p.suffix.lower() in ('.stl','.urdf','.so')]
    sources = [provenance(p) for p in sorted(set(paths))]
    report = dict(artifact=provenance(prepared),configuration=provenance(c['_config_path']),
                  sources=sources,blockers=[],input_type='live_cam0_kp21')
    report_path = directory/'prepared.json'; write_json(report_path,report)
    candidate = directory/'nominal_ik_candidate.npz'
    diagnostic = diagnose(c,prepared,report_path,1,candidate)
    write_json(directory/'nominal_ik_diagnostic.json',diagnostic)
    with np.load(candidate,allow_pickle=False) as z: sdk_arrays = {k:z[k].copy() for k in z.files}
    # These are read-only SDK calls; no power, enable or Servo is requested here.
    solutions = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {s:pool.submit(solve_side,c,s,sdk_arrays,controllers,hands,directory) for s in SIDES}
        for side in SIDES: solutions[side] = futures[side].result()[0]
    print('WARNING collision preflight disabled by operator; relying on human supervision and emergency stop.',flush=True)
    trajectory, plan = replay_planner.runtime_pair(
        c,controllers,hands,solutions['left'],solutions['right'],settings,arrays)
    write_json(directory/'pair_checks.json',plan)
    if not plan['ready_for_execution']: raise ValueError('Replay trajectory checks blocked this prediction')
    # Independent final SDK FK pass, as in aubo_arms_replay.main.
    for side in SIDES:
        arm = hardware.arms[side]
        for q,target in zip(solutions[side][1],arrays[side+'_aubo_base_T_flange'],strict=True):
            actual = arm.fk(q)
            if np.linalg.norm(actual[:3,3]-target[:3,3]) > 1e-5 or Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude() > 1e-4:
                raise ValueError('Independent SDK FK check failed')
    check_age(frame['sample_ns'],cfg['execution']['maximum_observation_age_s'])
    sources += [provenance(prepared),provenance(candidate)]
    dest = directory/'runtime.npz'
    np.savez_compressed(dest,joint_positions_left_right_rad=trajectory,execution_enabled=False)
    plan.update(name='xflow_'+directory.parent.name+'_'+directory.name,arms_only=False,
        execution_enabled=False,configuration=provenance(c['_config_path']),sources=sources,
        trajectory=provenance(dest),controllers=controllers,hand_start=hands,settings=settings,
        control_mode='retimed_replay',sample_ns=frame['sample_ns'],
        model_prediction_time_s=chunk['time_s'].tolist(),
        hand_motion_warnings=list(frame.get('warnings',[]))+hand_motion_warnings,
        timing_note='Forecast relative timing is uniformly slowed by replay; NOT wall-clock model-time tracking')
    write_json(directory/'runtime.json',plan)
    return c, settings, trajectory, plan


def execute_plan(cfg, hardware, frame, prepared, loop_hook=None):
    from replay_pipeline import provenance
    from replay_control import read_arms
    from read_hands import check_current_limit
    from replay_control import execute
    c, settings, trajectory, plan = prepared
    check_age(frame['sample_ns'],cfg['execution']['maximum_observation_age_s'])
    live = read_arms(c); fresh_hands = hardware.hands.snapshot()
    warnings = unchanged_start(frame['initial'],live,fresh_hands,cfg['execution'])
    # Equivalent integrity/arm checks to aubo_arms_replay.require_plan, except the
    # 0.005-rad hand-stationarity condition is deliberately warning-only.  The
    # executor still enforces its larger command-continuity/tracking bounds.
    if plan.get('ready_for_execution') is not True or plan.get('collision_check_policy') != 'disabled_by_operator':
        raise ValueError('Preflight has not passed')
    if provenance(c['_config_path']) != plan['configuration']: raise ValueError('Configuration changed')
    for item in plan['sources']:
        if provenance(item['path']) != item: raise ValueError('Source or implementation changed: '+item['path'])
    if provenance(plan['trajectory']['path']) != plan['trajectory']: raise ValueError('Trajectory changed')
    with np.load(plan['trajectory']['path'],allow_pickle=False) as saved:
        if not np.array_equal(saved['joint_positions_left_right_rad'],trajectory):
            raise ValueError('Trajectory in memory differs')
    check_current_limit(fresh_hands,plan['settings']['maximum_hand_current_a'])
    for side in SIDES:
        delta = abs(np.asarray(live[side]['joint_positions_rad'])-np.asarray(plan['controllers'][side]['joint_positions_rad']))
        joint = int(np.argmax(delta)); maximum = float(delta[joint])
        if maximum > .002:
            warning = dict(type='arm_start_drifted_after_planning',side=side,
                joint_index=joint,max_delta_rad=maximum,threshold_rad=.002,action='warn_only')
            warnings.append(warning); print('WARNING',warning,flush=True)
    if warnings:
        plan.setdefault('hand_motion_warnings',[]).extend(warnings)
        write_json(Path(plan['trajectory']['path']).with_name('start_motion_warnings.json'),warnings)
    hardware.owns_servo = True
    try:
        monitor = getattr(hardware,'live_compare',None)
        execute(c,settings,trajectory,plan,hardware.hands,arms_only=False,loop_hook=loop_hook,
                diagnostic_hook=monitor.command if monitor is not None else None)
    finally:
        hardware.owns_servo = False
