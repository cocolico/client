"""Actual deployed client/tunnel, recorded input only / 原客户端与隧道，绝不初始化SDK。"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
G2=(ROOT/'scripts/client_deployed78.py').exists()
COMMON=ROOT/'scripts' if G2 else ROOT/'src/infer/mixed78_common'
CONFIG=ROOT/'configs' if G2 else ROOT/'src/infer/mixed78_configs'
sys.path[:0]=[str(COMMON),str(ROOT/'src/infer/mixed76_common'),str(ROOT/'tests')]
from inference_connection import SSHTunnel
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
from probe_four_model_endpoints import recorded_capture


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    data=ROOT/'tests/fixtures'/('four_models_yifu_recorded.npz' if G2 else 'four_models_book_recorded.npz')
    results=[]
    with np.load(data,allow_pickle=False) as recording:
        for model in ('c0_full','c0_lora','c0_lr'):
            path=CONFIG/f'deployed_{model}.json';cfg=json.loads(path.read_text())
            tasks=load_tasks(CONFIG/cfg['tasks_file'])
            with SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
                client=open_client78(*address,remote_config=path,model=model,task=task,tasks=tasks,embodiment=kind)
                try:
                    for repeat in range(2):
                        h,packet=recorded_capture(recording);client.attach_history(h)
                        started=time.monotonic()
                        actions=client.infer(packet['state'],packet['head_rgb'],task=task,embodiment=kind,sample_ns=packet['sample_ns'])
                        if actions.shape!=(32,78) or not np.isfinite(actions).all():raise ValueError('Invalid actions')
                        row=dict(model=model,embodiment=kind,task=task,repeat=repeat,
                            inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-started,
                            shape=list(actions.shape),robot_io=False,deployment_id=client.info['deployment_id'],
                            server_host=cfg['server_host'],server_port=cfg['server_port'])
                        print(json.dumps(row),flush=True);results.append(row)
                finally:client.close()
    with a.output.open('x') as f:json.dump(results,f,indent=2)
