"""Use installed client and SSH tunnel with recorded input only.

直接复用已安装client、隧道、配对历史和坐标处理；不导入机器人或SDK。
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
from paired_history import PairedHistory


def recorded_capture(data):
    """Re-express each recorded pose in its own camera before normal ingestion.

    文件中的历史已统一到当前相机；恢复各自相机系后喂给原采集缓冲，验证其变换。
    """
    capture=PairedHistory(lambda:None)
    base=time.monotonic_ns()-810_000_000
    for i,(state,rgb) in enumerate(zip(data['state_history'],data['head_rgb_history'])):
        state=state.copy();r6=state[21:27].reshape(2,3).T
        rotation=np.column_stack((r6,np.cross(r6[:,0],r6[:,1])))
        anchor=np.eye(4);anchor[:3,:3]=rotation;anchor[:3,3]=state[18:21]
        for p in (0,9,18):
            state[p:p+3]=rotation.T@(state[p:p+3]-anchor[:3,3])
            columns=rotation.T@state[p+3:p+9].reshape(2,3).T
            state[p+3:p+9]=columns.T.reshape(6)
        capture.push(dict(sample_ns=base+i*200_000_000,source_id=i,state=state,
            head_rgb=rgb,T_source_camera=anchor))
    packet=capture.snapshot()
    prepared=capture.prepare(packet['sample_ns'])
    np.testing.assert_allclose(prepared['state_history'],data['state_history'],atol=2e-6,rtol=0)
    return capture,packet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    registry=load_tasks(CONFIG/('g2_tasks78.json' if G2 else 'tasks.json'))
    results=[]
    with np.load(args.input,allow_pickle=False) as data:
        for model in ('alpha','joint_relative_history','joint','tri_relative'):
            path=CONFIG/('deployed_'+model+'.json');config=json.loads(path.read_text())
            with SSHTunnel(config['ssh_host'],config['server_host'],config['server_port']) as address:
                client=open_client78(*address,remote_config=path,model=model,task=task,
                                     tasks=registry,embodiment=kind,timeout=60)
                try:
                    if model=='joint_relative_history':
                        capture,packet=recorded_capture(data);client.attach_history(capture)
                        state,rgb,stamp=packet['state'],packet['head_rgb'],packet['sample_ns']
                    else:
                        state,rgb,stamp=data['state'],data['head_rgb'],time.monotonic_ns()
                    start=time.monotonic();actions=client.infer(state,rgb,task=task,embodiment=kind,sample_ns=stamp)
                    elapsed=time.monotonic()-start
                    assert actions.shape==(32,78) and np.isfinite(actions).all()
                    result=dict(model=model,task=task,embodiment=kind,server_host=config['server_host'],
                        server_port=config['server_port'],deployment_id=client.info['deployment_id'],
                        shape=list(actions.shape),inference_s=client.last_response['inference_seconds'],
                        roundtrip_s=elapsed,history=client.last_response.get('history'),robot_io=False)
                    results.append(result);print(json.dumps(result),flush=True)
                finally:client.close()
    args.output.write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
