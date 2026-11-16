"""Recorded packets only. No robot SDK or motion calls."""
from pathlib import Path
import json,sys,time,hashlib,numpy as np
b=Path(sys.argv[1]).resolve();sys.path.insert(0,str(Path('/home/descfly/code/agibot_inference/scripts')))
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
from paired_history import PairedHistory
receipt=json.loads((b/'LATEST_SERVER.json').read_text());service=Path(receipt['output']);info=json.loads((service/'info.json').read_text());rows=json.loads((b/'verification/manifest.json').read_text());results=[]
out=Path('/home/descfly/code/agibot_inference/tests/local_server_warmup_20261111/recorded_client');out.mkdir(exist_ok=True)
for row in rows:
 kind=row['embodiment'].lower();task=row['task'].split('/')[-1];task='drink' if task=='drink_head_only' else task
 tasks={}
 for full,spec in info['tasks'].items():
  if spec['embodiment'].lower()!=kind:continue
  name=full.split('/')[-1];name='drink' if name=='drink_head_only' else name
  tasks[name]=dict(embodiment=kind,instruction=spec['prompt'],validity=[int(x) for x in spec['mask']])
 tf=out/('tasks_'+kind+'.json');tf.write_text(json.dumps(dict(schema='egoav_78d_20261021_v1',tasks=tasks)))
 cfg=dict(deployment_id=info['deployment_id'],checkpoint_sha256=info['checkpoint_sha256'],checkpoint_step=30000,normalization_sha256=info['normalization_sha256'],wire_protocol='egoav78_aligned_c0_v4',model='c0_lr',transport='tcp',lossless_transport=False,server_host='127.0.0.1',server_port=receipt['port'],tasks_file=tf.name,expected_experiment='aligned_camera_c0',model_native_dim=78,conditioning=info['conditioning'],proprio_history=False,denoise_steps=10,compile=True,client_image_preprocess=info['client_image_preprocess'])
 cf=out/('deployed_'+kind+'.json');cf.write_text(json.dumps(cfg));tasks=load_tasks(tf)
 with np.load(b/row['fixture'],allow_pickle=False) as data:state=data['state'].copy();frames=data['head_rgb_history'].copy()
 with np.load(b/row['reference'],allow_pickle=False) as data:reference=data['actions'].copy()
 client=open_client78('127.0.0.1',receipt['port'],remote_config=cf,model='c0_lr',task=task,tasks=tasks,embodiment=kind,timeout=900)
 try:
  for repeat in range(2):
   stamp=time.monotonic_ns()-5_000_000;capture=PairedHistory(lambda:None)
   for i,rgb in enumerate(frames):capture.push(dict(sample_ns=stamp+(i-4)*200_000_000,source_id=i,state=state,head_rgb=rgb,T_source_camera=np.eye(4)))
   packet=capture.snapshot();client.attach_history(capture);start=time.monotonic();action=client.infer(state,packet['head_rgb'],task=task,embodiment=kind,sample_ns=stamp)
   mask=np.asarray(tasks[task]['validity'],bool);assert action.shape==(32,78) and np.isfinite(action).all() and not action[:,~mask].any()
   for offset in (0,9,18):
    rot=action[:,offset+3:offset+9].reshape(-1,2,3).transpose(0,2,1);assert np.allclose(rot.transpose(0,2,1)@rot,np.eye(2),atol=1e-3)
   age_s=(time.monotonic_ns()-stamp)/1e9
   assert age_s<6., 'warm observation exceeded unchanged 6s budget'
   result=dict(response_age_s=age_s,task=row['task'],repeat=repeat,shape=list(action.shape),inference_s=client.last_response['inference_seconds'],roundtrip_s=time.monotonic()-start,reference_max_abs=float(np.max(np.abs(action-reference))),reference_mean_abs=float(np.mean(np.abs(action-reference))),robot_io=False)
   results.append(result);(out/'RESULTS.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
 finally:client.close()
assert len(results)==12
assert all(item['reference_max_abs']==0 for item in results), 'reference parity failed'
(out/'ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,requests=12,checkpoint_sha256=info['checkpoint_sha256'],deployment_id=info['deployment_id'],results=results,robot_io=False),indent=2));print('LOCAL_SERVER_VERIFIED 12_REQUESTS robot_io=false',flush=True)
