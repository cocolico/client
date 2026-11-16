"""Recorded images/state -> real local model -> RTC planner. No robot IO."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from g2_rtc import Settings,Trajectory,to_base78,compile_plan,PredictionWorker
from g2_chunk_camera import G2ChunkCamera
from g2_drink import DrinkAdapter
from egoav78_contract import load_tasks
from client_deployed78 import open_client78
from paired_history import PairedHistory

bundle=root/'model_servers/mid_a1_final';cfgpath=root/'configs/deployed_mid_a1_final_local.json'
cfg=json.loads(cfgpath.read_text());tasks=load_tasks(cfgpath.parent/cfg['tasks_file'])
out=root/'tests/g2_rtc_recorded_20261111';out.mkdir(exist_ok=True)
rows=json.loads((bundle/'verification/manifest.json').read_text());results=[]
for row in rows:
    if row['embodiment'].lower()!='g2':continue
    task=row['task'].split('/')[-1];task='drink' if task=='drink_head_only' else task
    drink=task=='drink';urdf=root/'assets/robot/agibot.urdf'
    adapter=DrinkAdapter(urdf) if drink else G2ChunkCamera(urdf,np.zeros(5))
    with np.load(bundle/row['fixture'],allow_pickle=False) as z:state=z['state'].copy();frames=z['head_rgb_history'].copy()
    body=np.zeros(5)
    if drink:body[4]=state[27];adapter.locked=body[:4].copy()
    T=adapter.fk.head_pose(body,np.zeros(3))
    context=SimpleNamespace(base_from_camera=T)
    base=to_base78(state[None],context,drink=drink)[0]
    client=open_client78('127.0.0.1',cfg['server_port'],remote_config=cfgpath,model='c0_lr',task=task,tasks=tasks,embodiment='g2',timeout=30)
    class Capture:
        def snapshot(self):
            stamp=time.monotonic_ns()-5_000_000;history=PairedHistory(lambda:None)
            for i,rgb in enumerate(frames):history.push(dict(sample_ns=stamp+(i-4)*200_000_000,source_id=i,state=state,head_rgb=rgb,T_source_camera=T))
            client.attach_history(history)
            return dict(sample_ns=stamp,state=state,head_rgb=frames[-1],context=context)
    worker=PredictionWorker(client,Capture(),adapter,Settings(),task=task,steps=32,drink=drink,max_response_age=6,emit=lambda *a,**k:None)
    try:
        for repeat in range(2):
            now=time.monotonic_ns();old=Trajectory(np.array([now-100_000_000,now+3_000_000_000]),np.repeat(base[None],2,0))
            start=time.monotonic();worker.submit(SimpleNamespace(trajectory=old),now)
            assert worker.job['done'].wait(30),'worker timeout'
            job=worker.poll();p=job['plan'];v=p.trajectory.values
            assert p.native61.shape[1]==61 and np.isfinite(p.native61).all()
            np.testing.assert_allclose(v[0],old.sample(p.trajectory.times[:1])[0],atol=1e-6)
            assert np.all(np.diff(p.trajectory.times)==10_000_000)
            r=dict(task=task,repeat=repeat,infer_s=job['infer_s'],planning_s=job['planning_s'],worker_roundtrip_s=time.monotonic()-start,
                   skipped_frames=job['skipped_prediction_frames'],subframes=len(v),blend_s=.1,robot_io=False)
            results.append(r);print(json.dumps(r),flush=True)
    finally:worker.close();client.close()
assert len(results)==8
(out/'ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,requests=8,tasks=4,results=results,robot_io=False,actual_server=True),indent=2))
print('RTC_REAL_MODEL_RECORDED_PASSED',flush=True)
