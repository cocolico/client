"""Drink-only waist contract and per-subframe kinematics. No hardware IO."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from test_g2_chunk_camera import native
from test_deployed78 import Endpoint, tasks
from test_inference_launcher import LauncherTests
from egoav78_contract import validate_tasks, validate78

PROMPT = '制作饮料：左手取杯，右手按饮品柜水龙头接饮料；随后转身用右手取冰铲加冰，最后将杯子放回桌面。'

def registry():
    r = tasks()
    s = dict(instruction=PROMPT, embodiment='g2', validity=r['yifu']['validity'].copy())
    s['validity'][27] = 1
    r['drink'] = s
    return r

class DrinkTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('g2_drink'), 'Drink adapter not implemented')
        return importlib.import_module('g2_drink')

    def test_online_roll_lower_limit_is_minus_25_degrees(self):
        from replay_public_eef import HeadKinematics
        urdf=ROOT/'assets/robot/agibot.urdf'
        fk=self.api().DrinkAdapter(urdf).fk
        strict=HeadKinematics(urdf)
        self.assertAlmostEqual(fk.bounds[1,0],np.deg2rad(-25.))
        self.assertAlmostEqual(strict.bounds[1,0],-.3491)
        np.testing.assert_array_equal(fk.bounds[[0,2]],strict.bounds[[0,2]])
        self.assertEqual(fk.bounds[1,1],strict.bounds[1,1])
        body=np.array([-.3,.6,-.3,.05,.1])
        for degrees in [-20.133453058847383,-24.9,-25.]:
            q=np.array([.1,np.deg2rad(degrees),.2])
            solved=fk.solve(fk.head_pose(body,q)[None],body,position_diagnostic_only=True)
            np.testing.assert_allclose(solved[0],q,atol=1e-12)
        for degrees in [-25.1,20.1]:
            q=[.1,np.deg2rad(degrees),.2]
            with self.assertRaisesRegex(ValueError,'online drink joint limits'):
                fk.solve(fk.head_pose(body,q)[None],body,position_diagnostic_only=True)

    def test_task_mask_allows_only_drink_yaw(self):
        r = registry()
        try: result = validate_tasks(r)
        except ValueError as e: self.fail(f'Drink task missing: {e}')
        self.assertEqual(sum(result['drink']['validity']), 72)
        self.assertEqual(sum(result['yifu']['validity']), 71)
        for task, index in [('yifu',27), ('drink',28), ('drink',54), ('basket',27)]:
            bad = registry(); bad[task]['validity'][index] = 1
            with self.assertRaises(ValueError): validate_tasks(bad)

    def test_network_preserves_waist_and_maps_drink_wire_name(self):
        from client_deployed78 import Deployed78Client
        e = Endpoint()
        try:
            try:
                c = Deployed78Client(*e.server.server_address, model='joint_relative',
                    task='drink', tasks=registry(), embodiment='g2',
                    expected_deployment=e.info['deployment_id'])
            except ValueError as ex: self.fail(f'Drink handshake missing: {ex}')
            try:
                x = np.zeros(78, np.float32)
                for s in [0,9,18]: x[s+3:s+9] = [1,0,0,0,1,0]
                x[27] = .23
                a = c.infer(x, np.zeros((32,32,3),np.uint8), task='drink',embodiment='g2',sample_ns=1)
                np.testing.assert_array_equal(a[:,27], np.full(32,np.float32(.23)))
                self.assertEqual(e.requests[-1]['task'], 'g2/drink_head_only')
                with self.assertRaises(ValueError): validate78(x, 'g2')
            finally: c.close()
        finally: e.close()

    def make_context(self):
        m = self.api()
        a = m.DrinkAdapter(ROOT/'assets/robot/agibot.urdf')
        body = np.array([-.3,.6,-.3,.05,.1])
        c = a.begin(native(), body, sample_ns=123)
        return m,a,c,body

    def test_measured_yaw_updates_anchor_and_old_context_is_immutable(self):
        m,a,c,b = self.make_context()
        self.assertAlmostEqual(c.state78[27], .1)
        np.testing.assert_allclose(c.state78[18:27],[0,0,0,1,0,0,0,1,0],atol=1e-7)
        b[4] = .3; d = a.begin(native(),b,sample_ns=124)
        self.assertFalse(np.allclose(c.base_from_camera,d.base_from_camera))
        with self.assertRaises(ValueError):c.body[4]=0
        b[0] += .2
        with self.assertRaises(ValueError):a.begin(native(),b,sample_ns=125)

    def test_every_subframe_uses_target_not_measured_or_final_yaw(self):
        m,a,c,b = self.make_context()
        x = c.state78[None].copy(); x[0,27] = .2
        plan = a.plan(c,x,startup_transition=False)
        np.testing.assert_allclose(plan.body[:,4],[.12,.14,.16,.18,.20],atol=1e-7)
        np.testing.assert_allclose(plan.body[:,:4],np.tile(b[:4],(5,1)))
        # Constant head EEF in base while waist yaw increases: head compensates.
        np.testing.assert_allclose(plan.head[:,0],[.18,.16,.14,.12,.10],atol=2e-7)
        np.testing.assert_allclose(plan.head[:,1:],np.tile(native()[59:],(5,1)),atol=2e-7)
        # Independently reconstruct the full end pose from each moving arm base.
        for i,body in enumerate(plan.body):
            B = a.fk.pose(dict(zip(a.body_names,body)), 'arm_base_link')
            for side,offset in [('left',0),('right',29)]:
                p = plan.arm_base[side][i]
                T = np.eye(4);T[:3,3]=p[1];T[:3,:3]=Rotation.from_quat(p[2]).as_matrix()
                rebuilt=B@T
                np.testing.assert_allclose(rebuilt[:3,3],native()[offset:offset+3],atol=2e-7)
                np.testing.assert_allclose(rebuilt[:3,:3],np.eye(3),atol=2e-7)

    def test_first_entry_smoothing_and_locked_slots(self):
        m,a,c,b = self.make_context()
        x=np.tile(c.state78,(2,1));x[:,27]=[.2,.25]
        p=a.plan(c,x,startup_transition=True)
        self.assertGreaterEqual(p.startup_ticks,300)
        self.assertEqual(len(p.body),p.startup_ticks+10)
        self.assertGreater(p.body[0,4],.1);self.assertLess(p.body[0,4],.101)
        self.assertAlmostEqual(p.body[p.startup_ticks-1,4],.2,places=6)
        np.testing.assert_allclose(p.body[p.startup_ticks:p.startup_ticks+5,4],.2,atol=1e-7)
        np.testing.assert_allclose(p.body[-5:,4],[.21,.22,.23,.24,.25],atol=1e-7)
        x[:,28]=.1
        with self.assertRaises(ValueError):a.plan(c,x)

    def test_non_yaw_targets_stay_fixed_after_small_measurement_drift(self):
        m,a,c,b=self.make_context();b[0]+=.05
        next_context=a.begin(native(),b,sample_ns=124)
        plan=a.plan(next_context,next_context.state78[None])
        np.testing.assert_allclose(plan.body[:,:4],np.tile([-.3,.6,-.3,.05],(5,1)))

    def test_invalid_yaw_or_head_fails_before_any_send(self):
        m,a,c,b=self.make_context();x=c.state78[None].copy()
        x[0,27]=99
        with self.assertRaises(ValueError):a.plan(c,x)
        x[0,27]=np.nan
        with self.assertRaises(ValueError):a.plan(c,x)

    def test_executor_serializes_target_body_head_and_arm_base_together(self):
        m,a,c,b=self.make_context();x=c.state78[None].copy();x[0,27]=.2
        p=a.plan(c,x)
        sent=[];hand_sent=[]
        class Publisher:
            def publish(self,msg,info): sent.append(msg);return True
        class Clock:
            now=0.
            def monotonic(self):return self.now
            def sleep(self,t):self.now+=t
        clock=Clock()
        joints=dict(zip(m.BODY_NAMES,b));joints.update(dict(zip(m.JOINT_NAMES,np.r_[native()[58:],np.zeros(14)])))
        status=NS(error_code=0,collision_pairs_1=[],collision_pairs_2=[])
        def tf(frame):
            offset=0 if frame=='arm_l_end_link' else 29
            return NS(translation=NS(**dict(zip(('x','y','z'),native()[offset:offset+3]))),
                      rotation=NS(x=0.,y=0.,z=0.,w=1.))
        robot=NS(robot=NS(get_motion_control_status=lambda:status),
            tf=NS(get_tf_from_base_link=tf,get_latest_timestamp=lambda frame:m.time.time_ns()),
            gdk=NS(Publisher=lambda *a:Publisher(),QoS=lambda:None,PubMsgInfo=NS))
        hands=NS(require_connected=lambda:None,send_qpos=lambda l,r,**kw:hand_sent.append(np.r_[l,r]))
        def feedback(robot,names):return [joints[n] for n in names]
        with patch.object(m,'read_positions',feedback),patch.object(m.time,'monotonic',clock.monotonic),patch.object(m.time,'sleep',clock.sleep):
            for indices in ((0,)*5,((0,0.5),)*5):
                with self.assertRaises(ValueError):
                    m.execute_plan(robot,replace(p,frame_indices=indices),hands,joints)
                self.assertEqual(sent,[])
            status.error_code=1
            with self.assertRaisesRegex(RuntimeError,'fault/collision'):
                m.execute_plan(robot,p,hands,joints)
            self.assertEqual(sent,[])
            status.error_code=0
            m.execute_plan(robot,p,hands,joints)
        self.assertEqual(len(sent),5);self.assertEqual(len(hand_sent),5)
        from genie_msgs_pb.msg.RetargetInfo_pb2 import RetargetInfo as RI
        for i,msg in enumerate(sent):
            groups={g.group_id:g for g in msg.retarget_groups}
            waist=groups[RI.GROUP_WAIST]
            self.assertEqual(waist.control_type,RI.ABS_JOINT)
            np.testing.assert_allclose(waist.target_joint_positions,p.body[i])
            self.assertEqual(groups[RI.GROUP_LEFT_ARM].frame_id,'arm_base_link')
            self.assertAlmostEqual(groups[RI.GROUP_HEAD_YAW].target_joint_positions[0],p.head[i,0])
        self.assertAlmostEqual(clock.now,.05)
        # A controller fault that occurs AFTER starting must stop at the next
        # existing 10Hz monitoring tick, before sending that tick's target.
        sent.clear();hand_sent.clear();clock.now=0.
        long_plan=a.plan(c,np.tile(c.state78,(4,1)))
        robot.robot.get_motion_control_status=lambda:NS(
            error_code=int(len(sent)>=10),collision_pairs_1=[],collision_pairs_2=[])
        with patch.object(m,'read_positions',feedback),patch.object(m.time,'monotonic',clock.monotonic),patch.object(m.time,'sleep',clock.sleep):
            with self.assertRaisesRegex(RuntimeError,'subframe 10.*fault/collision'):
                m.execute_plan(robot,long_plan,hands,joints)
        self.assertEqual(len(sent),10);self.assertEqual(len(hand_sent),10)

class DrinkLauncherTests(LauncherTests):
    def test_drink_routes_existing_launcher_to_new_task(self):
        r=self.launch(['--model','joint','--task','drink','--steps','32'])
        self.assertEqual(r.returncode,0,r.stderr)
        args=json.loads(self.capture.read_text())['argv']
        self.assertEqual(args[args.index('--task')+1],'drink')

if __name__=='__main__':unittest.main()
