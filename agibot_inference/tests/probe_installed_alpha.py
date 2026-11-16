"""Installed legacy Alpha client + reused SSH transport; recorded input only.

使用原有 client 及持久 SSH 隧道，只喂录制观测，不导入机器人 SDK。
"""
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
sys.path[:0]=[str(COMMON),str(ROOT/'src/infer/mixed76_common')]
from inference_connection import SSHTunnel
from client_deployed78 import open_client78
from egoav78_contract import load_tasks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    data=ROOT/'tests/fixtures'/('four_models_yifu_recorded.npz' if G2 else 'four_models_book_recorded.npz')
    path=CONFIG/'deployed_alpha.json';cfg=json.loads(path.read_text())
    tasks=load_tasks(CONFIG/cfg['tasks_file']);receipts=[]
    with np.load(data,allow_pickle=False) as d:
        state,rgb=d['state'].copy(),d['head_rgb'].copy()
    with SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
        client=open_client78(*address,remote_config=path,model='alpha',task=task,tasks=tasks,embodiment=kind)
        try:
            assert client.info['checkpoint_sha256']=='5aa4808c2b73f141f185da878661ab4dff0f2ee13f611b6f9402e5b8f402073d'
            assert not client.info['history_images'] and not client.info['history_proprio']
            for repeat in range(3):
                started=time.monotonic()
                actions=client.infer(state,rgb,task=task,embodiment=kind,sample_ns=time.monotonic_ns())
                assert actions.shape==(32,78) and np.isfinite(actions).all()
                receipt=dict(model='alpha',embodiment=kind,task=task,repeat=repeat,
                    inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-started,
                    shape=list(actions.shape),robot_io=False,deployment_id=client.info['deployment_id'],
                    server_host=cfg['server_host'],server_port=cfg['server_port'])
                print(json.dumps(receipt),flush=True);receipts.append(receipt)
        finally:client.close()
    with args.output.open('x') as f:json.dump(receipts,f,indent=2)
