"""Installed native client and persistent SSH, recorded inputs, no hardware.

四个profile均复用已安装client与SSH；只喂录制数据，绝不导入机器人SDK。
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];G2=(ROOT/'scripts/client_deployed78.py').exists()
COMMON=ROOT/'scripts' if G2 else ROOT/'src/infer/mixed78_common'
CONFIG=ROOT/'configs' if G2 else ROOT/'src/infer/mixed78_configs'
sys.path[:0]=[str(COMMON),str(ROOT/'src/infer/mixed76_common'),str(ROOT/'tests')]
from inference_connection import SSHTunnel
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
from probe_four_model_endpoints import recorded_capture


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    data=ROOT/'tests/fixtures'/('four_models_yifu_recorded.npz' if G2 else 'four_models_book_recorded.npz')
    results=[]
    for profile,step in [('mid_a0',30000),('mid_a1',30000),('mid_b1',30000)]:
        model='alpha' if profile=='alpha' else 'c0_lr';path=CONFIG/f'deployed_{profile}.json';cfg=json.loads(path.read_text())
        tasks=load_tasks(CONFIG/cfg['tasks_file'])
        with np.load(data,allow_pickle=False) as recording,SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
            client=open_client78(*address,remote_config=path,model=model,task=task,tasks=tasks,embodiment=kind)
            try:
                assert client.info['checkpoint_step']==step
                for repeat in range(2):
                    if model=='alpha':state,rgb,stamp=recording['state'],recording['head_rgb'],time.monotonic_ns()
                    else:
                        h,packet=recorded_capture(recording);client.attach_history(h)
                        state,rgb,stamp=packet['state'],packet['head_rgb'],packet['sample_ns']
                    started=time.monotonic();actions=client.infer(state,rgb,task=task,embodiment=kind,sample_ns=stamp)
                    assert actions.shape==(32,78) and np.isfinite(actions).all()
                    result=dict(profile=profile,model=model,step=step,embodiment=kind,task=task,repeat=repeat,
                        shape=list(actions.shape),inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-started,
                        deployment_id=client.info['deployment_id'],checkpoint_sha256=client.info['checkpoint_sha256'],robot_io=False)
                    results.append(result);print(json.dumps(result),flush=True)
            finally:client.close()
    with args.output.open('x') as f:json.dump(results,f,indent=2)
