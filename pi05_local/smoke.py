"""Real local GPU protocol test using saved observations; never connects robots."""
from pathlib import Path
import argparse
import sys
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'aubo_wuji'))
from common import load_config,write_json
from network import Client,connection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT.parent/'aubo_wuji/config.pi05_local.json')
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--n',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.n<=20:p.error('--n must be 1..20')
    cfg=load_config(args.config)
    out=ROOT/'smoke_runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir(parents=True)
    with np.load(args.observation,allow_pickle=False) as recorded:
        observation={k:recorded[k].copy() for k in ('head_rgb','keypoints')}
    reports=[]
    with connection(cfg['server']) as (host,port):
        client=Client(host,port,cfg['server'])
        try:
            write_json(out/'server_info.json',client.info)
            for i in range(args.n):
                begin=time.monotonic();result=client.infer(observation,time.monotonic_ns())
                elapsed=time.monotonic()-begin
                if result['native_keypoints'].shape!=(50,2,21,3) or not np.isfinite(result['native_keypoints']).all():
                    raise ValueError('Native 50-step prediction invalid')
                np.testing.assert_array_equal(result['keypoints'],result['native_keypoints'][:32])
                np.testing.assert_allclose(result['keypoint_time_s'],np.arange(1,33)/30,rtol=1e-6)
                np.savez_compressed(out/f'prediction_{i:02d}.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
                reports.append(dict(round_trip_s=elapsed,inference_s=result['inference_seconds']))
                print('GPU INFERENCE PASS',reports[-1],flush=True)
            write_json(out/'validation.json',dict(passed=True,hardware_connected=False,model=client.model,
                deployment_id=client.spec['deployment_id'],task=cfg['server']['task'],
                observation=str(args.observation.resolve()),reports=reports))
        finally:client.close()
    print('RESULT',out,flush=True)


if __name__=='__main__':main()
