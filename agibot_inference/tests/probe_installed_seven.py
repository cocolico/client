"""Installed-client acceptance of seven models and every supported task; no SDK."""
import argparse,json,sys,time
from pathlib import Path
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
from paired_history import PairedHistory

def capture(rec):
    if 'source_from_chunk_camera' not in rec.files:return recorded_capture(rec)
    h=PairedHistory(lambda:None);stamp=time.monotonic_ns()-10_000_000
    h.push(dict(sample_ns=stamp,source_id=0,state=rec['state'],head_rgb=rec['head_rgb'],T_source_camera=rec['source_from_chunk_camera']))
    return h,h.snapshot()

def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',nargs='+',default=['pi05','alpha','groot','c0_lr','mid_a1','mid_b1','mid_a1_new']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);results=[]
    for profile in a.profiles:
        path=CONFIG/f'deployed_{profile}.json';cfg=json.loads(path.read_text());tasks=load_tasks(CONFIG/cfg['tasks_file'])
        with SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
            for task in (['diguan','yifu','dengpao','drink'] if G2 else ['basket','book']):
                key=('g2_' if G2 else '3B_')+('drink_head_only' if task=='drink' else task)
                filename='final_g2_diguan.npz' if profile=='mid_a1_new' and task=='diguan' else key+'.npz'
                fixture=ROOT/'tests/fixtures/newdata_a1_1106'/filename
                with np.load(fixture,allow_pickle=False) as rec:
                    client=open_client78(*address,remote_config=path,model=cfg['model'],task=task,tasks=tasks,embodiment='g2' if G2 else '3b',timeout=900)
                    try:
                        for repeat in range(2):
                            if cfg['wire_protocol']=='egoav78_aligned_c0_v4':
                                h,packet=capture(rec);client.attach_history(h);state,rgb,stamp=packet['state'],packet['head_rgb'],packet['sample_ns']
                            else:state,rgb,stamp=rec['state'],rec['head_rgb'],time.monotonic_ns()
                            start=time.monotonic();actions=client.infer(state,rgb,task=task,embodiment='g2' if G2 else '3b',sample_ns=stamp)
                            assert actions.shape==(32,78) and np.isfinite(actions).all()
                            mask=np.asarray(tasks[task]['validity'],bool);assert not np.any(actions[:,~mask])
                            assert client.last_response['deployment_id']==cfg['deployment_id']
                            np.savez_compressed(a.output/(profile+'_'+key+'_'+str(repeat)+'.npz'),actions=actions,state=state)
                            row=dict(profile=profile,task=task,repeat=repeat,shape=list(actions.shape),inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-start,deployment_id=cfg['deployment_id'],server_host=cfg['server_host'],port=cfg['server_port'],robot_io=False)
                            results.append(row);print(json.dumps(row),flush=True)
                            (a.output/'results.json').write_text(json.dumps(results,indent=2))
                    finally:client.close()
    (a.output/'ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,requests=len(results),robot_io=False,results=results),indent=2))

if __name__=='__main__':main()
