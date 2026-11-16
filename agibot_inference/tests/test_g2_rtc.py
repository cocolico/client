import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import tempfile
import threading
import time
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from g2_rtc import Trajectory, mix_targets, replacement, Settings, compile_plan, to_base78, PredictionWorker, run_live_rtc


def targets(n, x=0., angle=0.):
    v=np.zeros((n,78),float)
    r=Rotation.from_euler('z',angle).as_matrix()
    for p in (0,9,18):
        v[:,p]=x;v[:,p+3:p+9]=np.r_[r[:,0],r[:,1]]
    return v


class RTCTests(unittest.TestCase):
    def test_late_result_uses_matching_time_not_first_prediction(self):
        t=np.arange(33,dtype=np.int64)*50_000_000+1_000_000_000
        v=targets(33);v[:,0]=np.arange(33)*.001
        plan=Trajectory(t,v)
        self.assertAlmostEqual(plan.sample([1_400_000_000])[0,0],.008)
        with self.assertRaises(ValueError):plan.sample([2_650_000_000])

    def test_crossfade_uses_both_moving_targets_and_endpoints(self):
        t=np.arange(33,dtype=np.int64)*50_000_000+1_000_000_000
        a=targets(33);b=targets(33,x=.02)
        a[:,0]+=np.arange(33)*.001;b[:,0]+=np.arange(33)*.002
        old,new=Trajectory(t,a),Trajectory(t,b)
        result=replacement(old,new,1_500_000_000,10_000_000,100_000_000)
        stamps=np.array([1_500_000_000,1_550_000_000,1_600_000_000])
        got=result.sample(stamps)[:,0]
        np.testing.assert_allclose(got,[.010,(.011+.042)/2,.044],atol=1e-9)
        self.assertFalse(result.values.flags.writeable)

    def test_rotation_shortest_arc_and_so3(self):
        a=targets(1,angle=np.deg2rad(179));b=targets(1,angle=np.deg2rad(-179))
        got=mix_targets(a,b,np.array([.5]))
        c=got[0,3:9].reshape(2,3).T;r=np.column_stack((c,np.cross(c[:,0],c[:,1])))
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
        self.assertAlmostEqual(abs(Rotation.from_matrix(r).as_euler('xyz')[2]),np.pi)

    def test_no_old_tail_means_no_fake_hold_or_extrapolation(self):
        old=Trajectory(np.array([0,500_000_000]),targets(2))
        new=Trajectory(np.array([100_000_000,900_000_000]),targets(2))
        with self.assertRaises(ValueError):replacement(old,new,450_000_000,10_000_000,100_000_000)

    def test_configuration_is_explicit_and_bounded(self):
        self.assertEqual(Settings().blend_s,.1)
        with self.assertRaises(ValueError):Settings(blend_s=-.1)
        with self.assertRaises(ValueError):Settings(servo_dt_s=.02)

    def test_opt_in_launcher_routes_separately_from_untouched_baseline(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'tests',prefix='rtc_route_') as d:
            p=Path(d);(p/'configs').mkdir();(p/'configs/deployed_mid_a1_final_local.json').write_text('{}')
            (p/'env.sh').write_text('export PATH="'+str(p)+':$PATH"\n')
            recorder=p/'python'
            recorder.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\nPath('+repr(str(p/'args.json'))+').write_text(json.dumps(sys.argv[1:]))\n');recorder.chmod(0o755)
            for script,target in [('run_inference.sh','run_mixed76_client.py'),('run_inference_rtc.sh','run_mixed76_client_rtc.py')]:
                (p/script).write_text((root/script).read_text())
                result=subprocess.run(['bash',str(p/script),'--model','mid_a1_final_local','--task','yifu','--steps','32'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                argv=json.loads((p/'args.json').read_text())
                self.assertEqual(Path(argv[0]).name,target)
                self.assertEqual(argv[argv.index('--model')+1],'c0_lr')
                self.assertEqual(Path(argv[argv.index('--remote-config')+1]).name,'deployed_mid_a1_final_local.json')

    def test_different_camera_anchors_are_restored_before_blending(self):
        from types import SimpleNamespace
        from g2_chunk_camera import transform71
        from egoav78_contract import g2_from78,g2_to78
        base=targets(1,x=.1)
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',.5).as_matrix();T[:3,3]=[.3,-.2,.5]
        local=g2_to78(transform71(g2_from78(base),np.linalg.inv(T)))
        restored=to_base78(local,SimpleNamespace(base_from_camera=T))
        np.testing.assert_allclose(restored,base,atol=2e-7)

    def test_four_tasks_reuse_existing_fk_and_native_mapping(self):
        from g2_chunk_camera import G2ChunkCamera
        from g2_drink import DrinkAdapter
        from egoav78_contract import g2_to78
        root=Path(__file__).resolve().parents[1];urdf=root/'assets/robot/agibot.urdf'
        native=np.zeros(61);native[[6,35]]=1;native[:3]=[.4,.2,1.];native[29:32]=[.4,-.2,1.]
        for task in ('diguan','yifu','dengpao','drink'):
            with self.subTest(task=task):
                drink=task=='drink';adapter=DrinkAdapter(urdf) if drink else G2ChunkCamera(urdf,np.zeros(5))
                context=adapter.begin(native,np.zeros(5),sample_ns=1_000_000_000) if drink else adapter.begin(native,sample_ns=1_000_000_000)
                state=context.state78.copy() if drink else g2_to78(context.state71)
                a=np.repeat(state[None],32,axis=0)
                if drink:a[:,27]=np.linspace(.001,.03,32)
                base=to_base78(np.vstack((state,a)),context,drink=drink)
                raw=Trajectory(1_000_000_000+np.arange(33)*50_000_000,base)
                times=1_000_000_000+np.arange(1,161)*10_000_000
                plan=compile_plan(Trajectory(times,raw.sample(times)),adapter,drink=drink)
                self.assertEqual(plan.native61.shape,(160,61));self.assertTrue(np.isfinite(plan.native61).all())
                if drink:
                    expected=adapter.plan(context,a)
                    np.testing.assert_allclose(plan.body,expected.body,atol=1e-6)
                    np.testing.assert_allclose(plan.head,expected.head,atol=1e-6)
                    np.testing.assert_allclose(plan.native61[:,7:29],expected.hands[:,:22],atol=1e-6)
                    for side in ('left','right'):
                        np.testing.assert_allclose(np.array(plan.arm_base[side])[:,1:4],np.array(expected.arm_base[side],dtype=object)[:,1].tolist(),atol=1e-6)
                else:
                    expected,_=adapter.decode(context,a[:,np.r_[0:9,32:54,9:18,55:77,18:27]])
                    np.testing.assert_allclose(plan.native61[4::5],expected,atol=1e-6)

    def test_worker_is_async_single_flight_and_never_a_robot_writer(self):
        state=targets(1)[0];context=SimpleNamespace(base_from_camera=np.eye(4))
        gate=threading.Event();main_thread=threading.get_ident();calls=[]
        class Client:
            last_response={'request_id':'test'}
            def infer(self,*a,**kw):
                calls.append(threading.get_ident());gate.wait(1);return np.repeat(state[None],32,0)
        class Capture:
            def snapshot(self):return dict(state=state,head_rgb=np.zeros((32,32,3),np.uint8),sample_ns=time.monotonic_ns(),context=context)
        origin=time.monotonic_ns();old=Trajectory(np.array([origin,origin+2_000_000_000]),targets(2))
        worker=PredictionWorker(Client(),Capture(),None,Settings(),task='yifu',steps=32,drink=False,max_response_age=6,emit=lambda *a,**k:None)
        with patch('g2_rtc.compile_plan',side_effect=lambda t,*a,**k:SimpleNamespace(trajectory=t)):
            worker.submit(SimpleNamespace(trajectory=old),origin)
            with self.assertRaises(RuntimeError):worker.submit(SimpleNamespace(trajectory=old),origin)
            self.assertIsNone(worker.poll());gate.set();self.assertTrue(worker.job['done'].wait(2))
            job=worker.poll();self.assertGreater(job['plan'].trajectory.times[0],job['packet']['sample_ns'])
        worker.close();self.assertEqual(len(calls),1);self.assertNotEqual(calls[0],main_thread)

    def test_complete_loop_keeps100hz_and_switches_without_robot_io(self):
        self.simulate_loop('yifu',.08)
        self.simulate_loop('drink',.08)

    def test_slow_prediction_stops_instead_of_replaying_an_expired_chunk(self):
        self.simulate_loop('yifu',.85,expect_failure=True)

    def simulate_loop(self,task,latency,expect_failure=False):
        from g2_chunk_camera import G2ChunkCamera
        from g2_drink import DrinkAdapter
        from egoav78_contract import g2_to78
        from replay import JOINT_NAMES,BODY_NAMES
        root=Path(__file__).resolve().parents[1];drink=task=='drink'
        adapter=DrinkAdapter(root/'assets/robot/agibot.urdf') if drink else G2ChunkCamera(root/'assets/robot/agibot.urdf',np.zeros(5))
        native=np.zeros(61);native[[6,35]]=1;native[:3]=[.4,.2,1.];native[29:32]=[.4,-.2,1.]
        joints={n:0. for n in JOINT_NAMES+BODY_NAMES}
        class Capture:
            def snapshot(self):
                stamp=time.monotonic_ns()
                ctx=adapter.begin(native,np.zeros(5),sample_ns=stamp) if drink else adapter.begin(native,sample_ns=stamp)
                state=ctx.state78.copy() if drink else g2_to78(ctx.state71)
                return dict(sample_ns=stamp,state=state,head_rgb=np.zeros((32,32,3),np.uint8),context=ctx,observation=dict(state=native,joints=joints))
            def check(self):pass
        class Client:
            last_response={}
            calls=0
            def infer(self,state,*a,**kw):
                self.calls+=1;time.sleep(latency);self.last_response={'request_id':str(self.calls)}
                v=np.repeat(state[None],32,0);v[:,0]+=min(self.calls,4)*.0001
                if drink:v[:,27]=.005
                return v
        sent=[]
        class Sender:
            def __init__(self,*a,**kw):self.sent=0
            def send(self,plan,index):
                sent.append((int(plan.trajectory.times[index]),time.monotonic_ns(),plan.native61[index].copy()))
                self.sent+=1
        robot=SimpleNamespace(robot=object(),execute_sharpa61=lambda *a,**k:None)
        tests=root/'tests';tests.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tests,prefix='rtc_sim_') as d:
            log=Path(d)/'events.jsonl'
            with patch('g2_rtc.LiveSender',Sender),patch('replay.read_positions',return_value=[0.]*len(JOINT_NAMES)),patch('g2_drink.execute_plan'):
                if expect_failure:
                    with self.assertRaises((RuntimeError,ValueError,TimeoutError)):
                        run_live_rtc(robot,object(),Client(),Capture(),adapter,task=task,steps=32,iterations=2,settings=Settings(),max_response_age=6,log_path=log)
                else:
                    run_live_rtc(robot,object(),Client(),Capture(),adapter,task=task,steps=32,iterations=2,settings=Settings(),max_response_age=6,log_path=log)
            events=[json.loads(l) for l in log.read_text().splitlines()]
            if expect_failure:self.assertEqual(events[-1]['event'],'STOP')
            else:
                self.assertEqual(events[-1]['event'],'COMPLETE')
                self.assertEqual(sum(e['event']=='HANDOFF' for e in events),1)
                self.assertGreater(next(e for e in events if e['event']=='NEXT_READY')['skipped_prediction_frames'],0)
                self.assertGreater(len(sent),160)
            if len(sent)>1:
                self.assertTrue(np.all(np.diff([r[0] for r in sent])==10_000_000))
                self.assertLess(max(np.diff([r[1] for r in sent]))/1e9,.1)


if __name__=='__main__':unittest.main()
