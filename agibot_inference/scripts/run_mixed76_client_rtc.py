#!/usr/bin/env python3
"""Opt-in RTC entry copied from the original client; original remains unchanged.

Default is recorded NPZ testing, with no robot imports or commands.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from client_mixed76 import Mixed76Client
from mixed76_contract import encode_state, decode_actions
from inference_connection import inference_connection
from controller_diagnostics import record_controller_fault

CONTROLLER_FAULT_DIRECTORY = Path(__file__).resolve().parents[1]/'reports/controller_faults'


def prepare_g2_observation(obs, *, sample_ns, adapter=None, wire78=False):
    """Coordinate-only hook. Default remains the existing absolute76 contract.

    A relative adapter yields physical71, NOT old v4 wire76. Selecting it for
    live inference requires a matching server contract; no automatic guessing.
    """
    if wire78 and adapter is None:
        raise ValueError('78D requires the existing camera coordinate adapter')
    if adapter is None:
        return encode_state(obs['state'], 'g2'), None
    context = adapter.begin(obs['state'], sample_ns=sample_ns)
    if wire78:
        from egoav78_contract import g2_to78
        return g2_to78(context.state71), context
    return context.state71, context


def restore_g2_actions(actions, *, context=None, adapter=None, wire78=False):
    """Return native61 to the SAME executor, without changing execution policy."""
    if wire78:
        if adapter is None:
            raise ValueError('78D requires the existing camera coordinate adapter')
        from egoav78_contract import g2_from78
        actions=g2_from78(actions)
    if adapter is None:
        if context is not None:
            raise ValueError('Coordinate context requires its adapter')
        return decode_actions(actions, 'g2'), None
    return adapter.decode(context, actions)


def read_reset_body_target(pico_root, robot_ssh):
    """Read the SAME tablet G2 pose as startup_reset, without --execute."""
    import subprocess
    import sys
    sys.path.insert(0,str(pico_root/'src'))
    from sharpa_pose_studio.common.paths import INPUT_CONFIG_PATH
    from sharpa_pose_studio.teleop.session import RealOps
    config=json.loads(INPUT_CONFIG_PATH.read_text());config['robot']=robot_ssh
    command=RealOps(config,INPUT_CONFIG_PATH).reset_command()
    if not command[-1].endswith(' --execute'):
        raise ValueError('Unexpected reset helper command; cannot make read-only request')
    command[-1]=command[-1][:-len(' --execute')]
    result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
    rows=[line for line in result.stdout.splitlines() if line.startswith('SHARPA_RESET_V1 ')]
    if len(rows)!=1:
        raise ValueError('Missing read-only reset pose report')
    report=json.loads(rows[0].split(' ',1)[1]);pose=report.get('pose',{})
    names=pose.get('joint_names',[]);positions=pose.get('joint_positions',[])
    if (report.get('validated') is not True or report.get('motion_sent') is not False
            or pose.get('name')!='G2零点' or pose.get('unit')!='rad'
            or len(names)!=len(positions) or len(names)!=len(set(names))):
        raise ValueError('Invalid read-only G2 reset pose report')
    values=dict(zip(names,positions))
    body=[values[f'idx0{i}_body_joint{i}'] for i in range(1,6)]
    if not np.isfinite(body).all():
        raise ValueError('Nonfinite fixed body target')
    return body


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',choices=['local','remote'],default='remote')
    p.add_argument('--remote-config',type=Path)
    p.add_argument('--representation',choices=['mixed76','g2-camera71','g2-camera78'],default='mixed76')
    p.add_argument('--model',required=True,choices=['fastwam','alpha','joint','joint_relative','joint_relative_c13','joint_relative_history','tri','tri_relative','coarse_fine','pi05','groot','model_a','model_b','model_c','model_d','c0_full','c0_lora','c0_lr'])
    p.add_argument('--embodiment',required=True,choices=['3b','g2'])
    p.add_argument('--checkpoint',help='optional exact expected checkpoint identity')
    p.add_argument('--ip',default='127.0.0.1');p.add_argument('--port',type=int,default=8857)
    p.add_argument('--task',choices=['basket','book','diguan','yifu','dengpao','drink'],required=True)
    p.add_argument('--observation',type=Path,help='NPZ containing native or76D state and head_rgb')
    p.add_argument('--coordinate-check',type=Path,
                   help='G2 recorded offline_replay.npz coordinate validation only; no network or robot')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--iterations',type=int,default=1)
    p.add_argument('--steps',type=int,default=32,help='execution prefix; reobserve after this prefix')
    p.add_argument('--max-response-age',type=float,default=6.)
    p.add_argument('--hands-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json')
    p.add_argument('--pico-root',type=Path,default=Path.home()/'code/egoav/pico')
    p.add_argument('--robot-ssh',default='agi')
    p.add_argument('--rtc-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/g2_rtc.json')
    args=p.parse_args()
    from g2_rtc import Settings
    rtc_settings=Settings.load(args.rtc_config)
    if args.representation!='g2-camera78' or args.embodiment!='g2' or args.model not in ('c0_full','c0_lora','c0_lr'):
        p.error('This isolated RTC entry currently supports G2 aligned-C0 models, including mid_a1_final_local')
    if args.steps*.05 <= rtc_settings.initial_lead_s+rtc_settings.blend_s+rtc_settings.planning_reserve_s:
        p.error('RTC execution prefix too short for configured lead/blend/planning budget; use --steps32')
    if args.coordinate_check:
        if args.execute or args.observation or args.embodiment!='g2':
            p.error('coordinate-check is G2 offline only; cannot combine with execute/observation')
        from g2_chunk_camera import validate_recorded
        report=validate_recorded(args.coordinate_check,
            Path(__file__).resolve().parents[1]/'assets/robot/agibot.urdf',chunk_size=args.steps)
        payload=json.dumps(report,ensure_ascii=False,indent=2)
        if args.output:
            args.output.write_text(payload+'\n')
        print(payload)
        return
    if args.inference=='local':
        args.remote_config=None
    if args.iterations < 1 or (args.execute and args.observation):
        p.error('iterations must be positive; execute and recorded observation are exclusive')
    if not 1<=args.steps<=32 or not np.isfinite(args.max_response_age) or args.max_response_age<=0:
        p.error('invalid steps or max-response-age')
    kind=args.embodiment
    wire78=args.representation=='g2-camera78'
    paired_history=args.model in ('model_a','model_b','model_c','model_d','joint_relative_history','c0_full','c0_lora','c0_lr')
    if paired_history and (not wire78 or kind!='g2'):
        p.error('Paired-history models require G2 camera78 on this host')
    drink=args.task=='drink'
    if drink and (not wire78 or kind!='g2' or args.model not in ('fastwam','joint_relative','joint_relative_c13','joint_relative_history','joint','tri','tri_relative','alpha','pi05','groot','model_c','model_d','c0_full','c0_lora','c0_lr')):
        p.error('drink requires G2 camera78 with a reviewed Joint/Tri/Alpha/Pi0.5/GR00T/FastWAM deployment')
    camera71=args.representation in ('g2-camera71','g2-camera78')
    if camera71 and kind!='g2':
        p.error('g2-camera71 supports G2 only')
    if args.inference=='remote' and args.remote_config is None:
        name='g2_camera78_remote.json' if wire78 else ('g2_camera71_remote.json' if camera71 else f'{args.model}_mixed76_v4.json')
        if wire78 and args.model in ('fastwam','joint_relative','joint_relative_c13','joint_relative_history','joint','tri','tri_relative','alpha','pi05','groot','model_a','model_b','model_c','model_d','c0_full','c0_lora','c0_lr'):
            name=f'deployed_{args.model}.json'
        args.remote_config=Path(__file__).resolve().parents[1]/'configs'/name
    if args.execute and kind!='g2':
        p.error('This real machine has G2 hardware; Piper inference is offline only')
    if wire78:
        from egoav78_contract import load_tasks
        from client_deployed78 import open_client78
        tasks78=load_tasks(Path(__file__).resolve().parents[1]/'configs/g2_tasks78.json')
        if args.task not in tasks78 or tasks78[args.task]['embodiment']!='g2':
            p.error('This host requires a registered G2 task')
    with inference_connection(args) as (host,port):
        if wire78:
            client=open_client78(host,port,remote_config=args.remote_config,model=args.model,task=args.task,tasks=tasks78,
                embodiment='g2',checkpoint=args.checkpoint,allow_simulated=not args.execute)
        else:
            client=Mixed76Client(host,port,model=args.model,checkpoint=args.checkpoint,
                             representation=args.representation)
        hands=None
        c13_capture=None
        paired_capture=None
        print('RUN_CONFIGURATION '+json.dumps(dict(model=args.model, task=args.task, steps=args.steps, remote_config=str(args.remote_config)),ensure_ascii=False),flush=True)
        run_context=dict(model=args.model, task=args.task, embodiment=kind,
                         representation=args.representation, steps=args.steps,
                         max_response_age_s=args.max_response_age,
                         server_info=client.info)
        fault_context=dict(run_context, phase='initialization')
        try:
            if wire78 and args.steps>client.info['horizon']:
                raise ValueError('Requested execution prefix exceeds policy horizon')
            print(json.dumps(client.info,ensure_ascii=False))
            adapter=None
            if camera71:
                from g2_chunk_camera import G2ChunkCamera
                urdf=Path(__file__).resolve().parents[1]/'assets/robot/agibot.urdf'
            if drink:
                from g2_drink import DrinkAdapter, execute_plan
                from replay import BODY_NAMES
                adapter=DrinkAdapter(urdf)
            if not args.execute:
                if args.observation:
                    with np.load(args.observation,allow_pickle=False) as data:
                        state=data['state'];rgb=data['head_rgb']
                        if drink:
                            # Explicit ACTUAL body feedback; never substitute reset targets.
                            context=adapter.begin(state,data['body_measured'],sample_ns=time.monotonic_ns())
                            state=context.state78
                        elif camera71:
                            adapter=G2ChunkCamera(urdf,data['body_target'])
                            state,context=prepare_g2_observation({'state':state},
                                sample_ns=time.monotonic_ns(),adapter=adapter,wire78=wire78)
                        elif state.shape!=(76,):
                            state=encode_state(state,kind)
                    actions=client.infer(state,rgb,task=args.task,embodiment=kind,
                                        sample_ns=context.sample_ns if camera71 else None)
                    if drink:
                        plan=adapter.plan(context,actions[:args.steps],startup_transition=True)
                        diagnostic=plan.report
                        print('G2_DRINK_DIAGNOSTIC '+json.dumps(diagnostic,ensure_ascii=False))
                    elif camera71:
                        native,diagnostic=restore_g2_actions(actions,context=context,adapter=adapter,wire78=wire78)
                        print('G2_COORDINATE_DIAGNOSTIC '+json.dumps(diagnostic,ensure_ascii=False))
                    if args.output:
                        if drink:
                            np.savez(args.output,actions78=actions,waist100=plan.body,head100=plan.head,
                                hands100=plan.hands,base_from_camera=context.base_from_camera,
                                diagnostic_json=json.dumps(diagnostic))
                        elif camera71:
                            np.savez(args.output,**{('actions78' if wire78 else 'actions71'):actions},actions61=native,
                                     base_from_camera=context.base_from_camera,
                                     diagnostic_json=json.dumps(diagnostic))
                        else:
                            np.save(args.output,actions)
                    print(json.dumps({'shape':list(actions.shape),'finite':bool(np.isfinite(actions).all()),
                        'inference_seconds':client.last_response['inference_seconds'],'robot_io':False}))
                return
            # Preserve the previously installed G2 reset, feedback, hands and
            # execution lifecycle. No hardware imports in the default path.
            from run_ours_client import startup_reset
            from robot_wrapper import AgiBotRobot
            from sharpa_hands import SharpaHands
            from hand_startup import load_hand_reset, reset_hands_to_recording
            from replay_full_eef import assert_ready
            hand_reset=load_hand_reset(args.task)
            if camera71 and not drink:
                body_target=read_reset_body_target(args.pico_root,args.robot_ssh)
                adapter=G2ChunkCamera(urdf,body_target)
                print('G2_FIXED_BODY_TARGET '+json.dumps(body_target),flush=True)
            startup_reset(args.pico_root,args.robot_ssh)
            robot=AgiBotRobot(with_eef=True)
            hands=SharpaHands.from_config(args.hands_config);hands.connect()
            fault_context['phase']='hand_startup_reset'
            reset_hands_to_recording(hands,hand_reset,check=lambda:assert_ready(robot.robot))
            if paired_history:
                import threading
                from abcd_g2_capture import SerializedSDK, g2_history_reader
                from paired_history import PairedHistory
                sdk_lock=threading.RLock()
                hands=SerializedSDK(hands,sdk_lock)
                robot.robot=SerializedSDK(robot.robot,sdk_lock)
                robot.tf=SerializedSDK(robot.tf,sdk_lock)
                paired_capture=PairedHistory(g2_history_reader(robot,hands,adapter,
                    drink=drink,body_names=BODY_NAMES if drink else ()))
                client.attach_history(paired_capture)
                paired_capture.__enter__()
            if args.model=='joint_relative_c13':
                from c13_rgb_transport import g2_head_reader
                c13_capture=client.capture_history(g2_head_reader(robot))
                c13_capture.__enter__()
            from g2_rtc import run_live_rtc
            import os
            rtc_log=Path(__file__).resolve().parents[1]/'logs'/'rtc'/(time.strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())+'.jsonl')
            fault_context.update(phase='rtc_execution', rtc_log=str(rtc_log))
            run_live_rtc(robot,hands,client,paired_capture,adapter,task=args.task,
                steps=args.steps,iterations=args.iterations,settings=rtc_settings,
                max_response_age=args.max_response_age,log_path=rtc_log)
        except Exception as exc:
            record_controller_fault(exc, context=fault_context,
                                    directory=CONTROLLER_FAULT_DIRECTORY)
            raise
        finally:
            try:
                try:
                    if paired_capture is not None:paired_capture.close()
                finally:
                    if c13_capture is not None:c13_capture.close()
            finally:
                try:
                    if hands is not None:
                        hands.close()
                finally:
                    client.close()


if __name__=='__main__':
    main()
