"""Real local model and recorded inputs; fake robot/sender only, no device IO."""
import json,sys,time,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from g2_rtc import Settings,Trajectory,to_base78,compile_plan,run_live_rtc
from g2_chunk_camera import G2ChunkCamera
from g2_drink import DrinkAdapter
from egoav78_contract import load_tasks
from client_deployed78 import open_client78
from paired_history import PairedHistory
from replay import BODY_NAMES,JOINT_NAMES

bundle=root/'model_servers/mid_a1_final';cf=root/'configs/deployed_mid_a1_final_local.json';cfg=json.loads(cf.read_text())
tasks=load_tasks(cf.parent/cfg['tasks_file']);out=root/'tests/g2_rtc_loop_20261111';out.mkdir(exist_ok=True)
results=[]
for item in json.loads((bundle/'verification/manifest.json').read_text()):
    if item['embodiment'].lower()!='g2':continue
    task=item['task'].split('/')[-1];task='drink' if task=='drink_head_only' else task;drink=task=='drink'
    with np.load(bundle/item['fixture'],allow_pickle=False) as z:state=z['state'].copy();frames=z['head_rgb_history'].copy()
    urdf=root/'assets/robot/agibot.urdf';adapter=DrinkAdapter(urdf) if drink else G2ChunkCamera(urdf,np.zeros(5))
    body=np.zeros(5)
    if drink:body[4]=state[27];adapter.locked=body[:4].copy()
    T=adapter.fk.head_pose(body,np.zeros(3));base=to_base78(state[None],SimpleNamespace(base_from_camera=T),drink=drink)[0]
    initial=compile_plan(Trajectory(np.array([0,10_000_000]),np.repeat(base[None],2,0)),adapter,drink=drink).native61[0]
    joints={n:0. for n in JOINT_NAMES+BODY_NAMES};joints.update(zip(BODY_NAMES,body));joints.update(zip(JOINT_NAMES[:3],initial[58:61]))
    client=open_client78('127.0.0.1',cfg['server_port'],remote_config=cf,model='c0_lr',task=task,tasks=tasks,embodiment='g2',timeout=30)
    class Capture:
        def snapshot(self):
            stamp=time.monotonic_ns()-5_000_000;ctx=adapter.begin(initial,body,sample_ns=stamp) if drink else adapter.begin(initial,sample_ns=stamp)
            h=PairedHistory(lambda:None)
            for i,rgb in enumerate(frames):h.push(dict(sample_ns=stamp+(i-4)*200_000_000,source_id=i,state=state,head_rgb=rgb,T_source_camera=T))
            client.attach_history(h)
            return dict(sample_ns=stamp,state=state,head_rgb=frames[-1],context=ctx,observation=dict(state=initial,joints=joints))
        def check(self):pass
    sent=[]
    class Sender:
        def __init__(self,*a,**kw):self.sent=0
        def send(self,plan,index):
            sent.append((int(plan.trajectory.times[index]),time.monotonic_ns()));self.sent+=1
    robot=SimpleNamespace(robot=object(),execute_sharpa61=lambda *a,**kw:None)
    log=out/(task+'-'+str(time.monotonic_ns())+'.jsonl')
    try:
        with patch('g2_rtc.LiveSender',Sender),patch('replay.read_positions',return_value=[joints[n] for n in JOINT_NAMES]),patch('g2_drink.execute_plan'):
            run_live_rtc(robot,object(),client,Capture(),adapter,task=task,steps=32,iterations=3,settings=Settings(),max_response_age=6,log_path=log)
    finally:client.close()
    events=[json.loads(l) for l in log.read_text().splitlines()]
    assert events[-1]['event']=='COMPLETE'
    assert sum(e['event']=='HANDOFF' for e in events)==2
    intervals=np.diff([s[1] for s in sent])/1e9
    assert np.all(np.diff([s[0] for s in sent])==10_000_000)
    assert intervals.max()<.1
    r=dict(task=task,sent=len(sent),handoffs=2,mean_send_interval_s=float(intervals.mean()),max_send_interval_s=float(intervals.max()),
           actual_model=True,recorded_inputs=True,robot_io=False,execution='fake_sender',log=str(log))
    results.append(r);print('LOOP_PASSED '+json.dumps(r),flush=True)
(out/'ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,tasks=4,results=results,robot_io=False),indent=2))
