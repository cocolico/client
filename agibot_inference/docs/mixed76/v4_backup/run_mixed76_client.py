#!/usr/bin/env python3
"""Mixed76 client: receives76D; converts to native only at explicit execution.

Default is recorded NPZ testing, with no robot imports or commands.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from client_mixed76 import Mixed76Client
from mixed76_contract import encode_state, decode_actions
from inference_connection import inference_connection


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference',choices=['local','remote'],default='remote')
    p.add_argument('--remote-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/pi05_mixed76_b_dev.json')
    p.add_argument('--ip',default='127.0.0.1');p.add_argument('--port',type=int,default=8857)
    p.add_argument('--task',choices=['basket','book','diguan','yifu','dengpao'],required=True)
    p.add_argument('--observation',type=Path,help='NPZ containing native or76D state and head_rgb')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--iterations',type=int,default=1)
    p.add_argument('--hands-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json')
    p.add_argument('--pico-root',type=Path,default=Path('/home/descfly/code/egoav/pico'))
    p.add_argument('--robot-ssh',default='agi')
    args=p.parse_args()
    if args.inference=='local':
        args.remote_config=None
    if args.iterations < 1 or (args.execute and args.observation):
        p.error('iterations must be positive; execute and recorded observation are exclusive')
    kind='3b' if args.task in ('basket','book') else 'g2'
    if args.execute and kind!='g2':
        p.error('This real machine has G2 hardware; Piper inference is offline only')
    with inference_connection(args) as (host,port):
        client=Mixed76Client(host,port)
        hands=None
        try:
            print(json.dumps(client.info,ensure_ascii=False))
            if not args.execute:
                if args.observation:
                    with np.load(args.observation,allow_pickle=False) as data:
                        state=data['state'];rgb=data['head_rgb']
                        if state.shape!=(76,):
                            state=encode_state(state,kind)
                    actions=client.infer(state,rgb,task=args.task,embodiment=kind)
                    if args.output:
                        np.save(args.output,actions)
                    print(json.dumps({'shape':list(actions.shape),'finite':bool(np.isfinite(actions).all()),
                        'inference_seconds':client.last_response['inference_seconds'],'robot_io':False}))
                return
            # Preserve the previously installed G2 reset, feedback, hands and
            # execution lifecycle. No hardware imports in the default path.
            from run_ours_client import startup_reset
            from robot_wrapper import AgiBotRobot
            from sharpa_hands import SharpaHands
            startup_reset(args.pico_root,args.robot_ssh)
            robot=AgiBotRobot(with_eef=True)
            hands=SharpaHands.from_config(args.hands_config);hands.connect()
            for iteration in range(args.iterations):
                obs=robot.get_observation61(hands)
                actions76=client.infer(encode_state(obs['state'],'g2'),obs['images']['head_color'],
                                      task=args.task,embodiment='g2')
                actions61=decode_actions(actions76,'g2')
                robot.execute_sharpa61(actions61,obs['state'],obs['joints'],hands,
                                      startup_transition=(iteration==0))
        finally:
            try:
                if hands is not None:
                    hands.close()
            finally:
                client.close()


if __name__=='__main__':
    main()
