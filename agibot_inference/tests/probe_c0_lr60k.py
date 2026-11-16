"""Installed client, recorded history, persistent SSH; no hardware imports.

复用安装的客户端、历史与持久SSH隧道，不初始化机器人或手部SDK。
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
sys.path[:0]=[str(COMMON),str(ROOT/'src/infer/mixed76_common'),str(ROOT/'tests')]
from inference_connection import SSHTunnel
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
from probe_four_model_endpoints import recorded_capture


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    data=ROOT/'tests/fixtures'/('four_models_yifu_recorded.npz' if G2 else 'four_models_book_recorded.npz')
    path=CONFIG/'deployed_c0_lr.json';cfg=json.loads(path.read_text());tasks=load_tasks(CONFIG/cfg['tasks_file'])
    assert cfg['checkpoint_step']==60000;results=[]
    with np.load(data,allow_pickle=False) as recording,SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
        client=open_client78(*address,remote_config=path,model='c0_lr',task=task,tasks=tasks,embodiment=kind)
        try:
            for repeat in range(3):
                h,packet=recorded_capture(recording);client.attach_history(h);started=time.monotonic()
                actions=client.infer(packet['state'],packet['head_rgb'],task=task,embodiment=kind,sample_ns=packet['sample_ns'])
                assert actions.shape==(32,78) and np.isfinite(actions).all()
                result=dict(model='c0_lr',step=60000,embodiment=kind,task=task,repeat=repeat,shape=list(actions.shape),
                    inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-started,
                    deployment_id=client.info['deployment_id'],checkpoint_sha256=client.info['checkpoint_sha256'],robot_io=False)
                results.append(result);print(json.dumps(result),flush=True)
        finally:client.close()
    with a.output.open('x') as f:json.dump(results,f,indent=2)
