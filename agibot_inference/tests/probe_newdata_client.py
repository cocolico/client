"""Recorded-input acceptance through the installed client and SSH; no SDK imports."""
import json,sys,time
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
from task_image_preprocess import CONTRACT

def main():
    output=Path(sys.argv[1]);output.mkdir(exist_ok=False)
    path=CONFIG/'deployed_mid_a1_new.json';cfg=json.loads(path.read_text())
    tasks=load_tasks(CONFIG/cfg['tasks_file']);results=[];kind='g2' if G2 else '3b'
    with SSHTunnel(cfg['ssh_host'],cfg['server_host'],cfg['server_port']) as address:
        for task in (['diguan','yifu','dengpao','drink'] if G2 else ['basket','book']):
            key=('g2_' if G2 else '3B_')+('drink_head_only' if task=='drink' else task)
            fixture=ROOT/'tests/fixtures/newdata_a1_1106'/(key+'.npz')
            with np.load(fixture,allow_pickle=False) as rec:
                client=open_client78(*address,remote_config=path,model='c0_lr',task=task,tasks=tasks,embodiment=kind,timeout=900)
                try:
                    assert client.info['client_image_preprocess']==CONTRACT
                    assert client.info['checkpoint_sha256']=='3002094efffb43895c6668ac3fdc12e6feb883f6c39e11c54f351c62085d9d8f'
                    for repeat in range(2):
                        h,packet=recorded_capture(rec);client.attach_history(h)
                        start=time.monotonic()
                        actions=client.infer(packet['state'],packet['head_rgb'],task=task,embodiment=kind,sample_ns=packet['sample_ns'])
                        assert actions.shape==(32,78) and np.isfinite(actions).all()
                        mask=np.asarray(tasks[task]['validity'],bool)
                        assert np.all(actions[:,~mask]==0)
                        assert client.last_response['client_image_preprocess']==CONTRACT
                        np.savez_compressed(output/(key+'_'+str(repeat)+'.npz'),actions=actions,state=packet['state'])
                        row=dict(task=task,repeat=repeat,shape=list(actions.shape),inference_s=client.last_response['inference_seconds'],
                            roundtrip_s=time.monotonic()-start,cuda_memory=client.last_response['cuda_memory'],mask_verified=True,
                            checkpoint_sha256=client.info['checkpoint_sha256'],deployment_id=client.info['deployment_id'],
                            original_rgb_shape=list(packet['head_rgb'].shape),client_crop=task=='diguan',robot_io=False)
                        results.append(row);print(json.dumps(row),flush=True)
                        (output/'results.json').write_text(json.dumps(results,indent=2))
                finally:client.close()
    (output/'ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,robot_io=False,results=results),indent=2))

if __name__=='__main__':main()
