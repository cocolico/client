"""Recorded-data calls through installed clients/tunnels; never import robot SDK.

录制数据联调：复用现有客户端和SSH隧道，不启动机器人。
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from probe_four_model_endpoints import (
    G2, CONFIG, SSHTunnel, open_client78, load_tasks, recorded_capture,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    task,kind=('yifu','g2') if G2 else ('book','3b')
    registry=load_tasks(CONFIG/('g2_tasks78.json' if G2 else 'tasks.json'))
    results=[]
    with np.load(args.input,allow_pickle=False) as data:
        for model in ('model_a','model_b','model_c','model_d'):
            path=CONFIG/('deployed_'+model+'.json')
            config=json.loads(path.read_text())
            with SSHTunnel(config['ssh_host'],config['server_host'],config['server_port']) as address:
                client=open_client78(*address,remote_config=path,model=model,task=task,
                    tasks=registry,embodiment=kind,timeout=300)
                try:
                    assert client.info['proprio_history'] is False
                    assert client.info['conditioning']=='image_history_current_proprio'
                    for repeat in range(2):
                        capture,packet=recorded_capture(data);client.attach_history(capture)
                        start=time.monotonic()
                        actions=client.infer(packet['state'],packet['head_rgb'],task=task,
                            embodiment=kind,sample_ns=packet['sample_ns'])
                        assert actions.shape==(32,78) and np.isfinite(actions).all()
                        mask=np.asarray(client.spec['validity'],bool)
                        assert not actions[:,~mask].any()
                        result=dict(model=model,task=task,repeat=repeat,embodiment=kind,
                            server_host=config['server_host'],port=config['server_port'],
                            deployment_id=client.info['deployment_id'],
                            checkpoint_sha256=client.info['checkpoint_sha256'],
                            shape=list(actions.shape),inference_s=client.last_response['inference_seconds'],
                            roundtrip_s=time.monotonic()-start,proprio_history=False,
                            conditioning=client.info['conditioning'],robot_io=False)
                        results.append(result);print(json.dumps(result),flush=True)
                finally:client.close()
    with args.output.open('x') as stream:json.dump(results,stream,indent=2)


if __name__=='__main__':main()
