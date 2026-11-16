"""Recorded-input probes using installed clients and SSH. No robot imports."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
G2=(ROOT/'scripts/client_deployed78.py').exists()
CONFIG=ROOT/'configs' if G2 else ROOT/'src/infer/mixed78_configs'
COMMON=ROOT/'scripts' if G2 else ROOT/'src/infer/mixed78_common'
kind='g2' if G2 else '3b'
sys.path[:0]=[str(COMMON),str(ROOT/'src/infer/mixed76_common'),str(ROOT/'tests')]
from inference_connection import SSHTunnel
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
from probe_four_model_endpoints import recorded_capture
p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task');p.add_argument('--fixture',type=Path);args=p.parse_args()
task=args.task or ('yifu' if G2 else 'book')
fixture=args.fixture or ROOT/'tests/fixtures'/('four_models_yifu_recorded.npz' if G2 else 'four_models_book_recorded.npz')
results=[]
for profile in args.profiles:
 path=CONFIG/f'deployed_{profile}.json';cfg=json.loads(path.read_text());model=cfg['model'];tasks=load_tasks(CONFIG/cfg['tasks_file'])
 with np.load(fixture,allow_pickle=False) as rec,SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
  client=open_client78(*address,remote_config=path,model=model,task=task,tasks=tasks,embodiment=kind)
  try:
   assert client.info['checkpoint_step']==30000
   assert client.info['deployment_id']==cfg['deployment_id']
   for repeat in range(2):
    if profile in ('c0_lr','mid_a0','mid_a1','mid_b1'):
     h,packet=recorded_capture(rec);client.attach_history(h);state,rgb,stamp=packet['state'],packet['head_rgb'],packet['sample_ns']
    else:state,rgb,stamp=rec['state'],rec['head_rgb'],time.monotonic_ns()
    start=time.monotonic();actions=client.infer(state,rgb,task=task,embodiment=kind,sample_ns=stamp)
    assert actions.shape==(32,78) and np.isfinite(actions).all()
    row=dict(profile=profile,model=model,task=task,repeat=repeat,shape=list(actions.shape),checkpoint_step=client.info['checkpoint_step'],checkpoint_sha256=client.info['checkpoint_sha256'],deployment_id=client.info['deployment_id'],server_host=cfg['server_host'],server_port=cfg['server_port'],inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-start,robot_io=False)
    print(json.dumps(row),flush=True);results.append(row)
  finally:client.close()
with args.output.open('x') as f:json.dump(results,f,indent=2)
