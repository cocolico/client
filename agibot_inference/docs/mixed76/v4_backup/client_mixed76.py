"""76D TCP client. infer() always returns physical76, never native61/23."""
from __future__ import annotations
import argparse
import json
import socket
from pathlib import Path
import numpy as np
from _ours_protocol import send_message, recv_message
from mixed76_contract import REPRESENTATION, encode_state, validate_actions, embodiment_mask


class Mixed76Client:
    def __init__(self, host: str, port: int):
        self.socket = socket.create_connection((host,port), timeout=300)
        try:
            send_message(self.socket, {'cmd':'info'})
            self.info = recv_message(self.socket)
            for key,want in {'action_dim':76,'state_dim':76,'representation':REPRESENTATION,
                             'action_units':'physical','protocol_version':3}.items():
                if self.info.get(key) != want:
                    raise ValueError(f'Server contract mismatch: {key}')
        except BaseException:
            self.close(); raise

    def close(self) -> None:
        self.socket.close()

    def infer(self, state76: np.ndarray, head_rgb: np.ndarray, *, task: str, embodiment: str) -> np.ndarray:
        state = np.asarray(state76,np.float32)
        if state.shape != (76,) or not np.isfinite(state).all():
            raise ValueError('Expected physical76 state')
        if np.any(state[~embodiment_mask(embodiment)] != 0):
            raise ValueError('Inactive state slots must be zero')
        send_message(self.socket, {'cmd':'infer','task':task,'embodiment':embodiment,
                                  'observation':{'state':state,'head_rgb':head_rgb}})
        result = recv_message(self.socket)
        if 'error' in result:
            raise RuntimeError(result['error'])
        if result.get('representation') != REPRESENTATION or result.get('embodiment') != embodiment:
            raise ValueError('Response contract mismatch')
        if result.get('action_mask') != embodiment_mask(embodiment).tolist():
            raise ValueError('Response mask mismatch')
        self.last_response = result
        return validate_actions(result['actions'],embodiment,self.info['horizon'])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8857)
    p.add_argument('--observation',type=Path,help='npz: state (native23/61 or76), head_rgb uint8 HWC')
    p.add_argument('--embodiment',choices=['3b','g2'],default='g2')
    p.add_argument('--task',default='yifu'); p.add_argument('--output',type=Path)
    args = p.parse_args()
    client = Mixed76Client(args.host,args.port)
    try:
        print(json.dumps(client.info,ensure_ascii=False))
        if args.observation:
            with np.load(args.observation,allow_pickle=False) as data:
                state = data['state']
                if state.shape != (76,):
                    state = encode_state(state,args.embodiment)
                actions = client.infer(state,data['head_rgb'],task=args.task,embodiment=args.embodiment)
            if args.output:
                np.save(args.output,actions)
            print(json.dumps({'shape':list(actions.shape),'finite':bool(np.isfinite(actions).all()),
                              'inference_seconds':client.last_response['inference_seconds']}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
