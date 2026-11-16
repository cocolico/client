"""Offline protobuf fixtures and fake hardware boundaries; never connects hardware."""
import contextlib
import importlib
import importlib.util
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from genie_msgs_pb.msg.JointState_pb2 import JointState
from tf2_msgs_pb.msg.TFMessage_pb2 import TFMessage
from agibot_tasks import SHARPA_JOINT_NAMES
import replay_full_eef as replay


def write_records(path, messages):
    with path.open('wb') as stream:
        stream.write(struct.pack('<q', len(messages)))
        for i, msg in enumerate(messages):
            raw = msg.SerializeToString()
            # Deliberately different from source timestamps: playback must use source time.
            stream.write(struct.pack('<5q', 900000000000+i, 0, 0, 0, len(raw)))
            stream.write(raw)


def joint(stamp, q, names=SHARPA_JOINT_NAMES):
    msg = JointState()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
    msg.name.extend(names); msg.position.extend(q)
    msg.motor_position.extend([9.]*len(q))
    return msg


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = self.root/'record'; self.record.mkdir()

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('replay_feedback'),
                             'feedback replay reader not implemented')
        return importlib.import_module('replay_feedback')

    def fixture(self, stamps=(1000000000, 1020000000, 1040000000), reverse=False):
        for side, offset in (('left', 0.), ('right', .5)):
            rows = []
            for i, stamp in enumerate(stamps):
                names = SHARPA_JOINT_NAMES[::-1] if reverse else SHARPA_JOINT_NAMES
                q = [offset+i*.02+j*.001 for j in range(22)]
                rows.append(joint(stamp, q[::-1] if reverse else q, names))
            write_records(self.record/f'-sharpa-{side}-state.pbdat', rows)
        # Target stream is intentionally wrong; it must never be used for qpos replay.
        (self.record/'-sharpa-left-target.pbdat').write_bytes(b'not a protobuf')

    def test_actual_position_reordered_by_name_and_interpolated_on_source_clock(self):
        self.fixture(reverse=True); m=self.module()
        streams=m.load_feedback(self.root)
        rows, report=m.align_feedback(streams, [1000000000,1010000000,1040000000])
        self.assertAlmostEqual(rows[1][0][0], .01)
        self.assertAlmostEqual(rows[1][0][21], .031)
        self.assertAlmostEqual(rows[1][1][0], .51)
        self.assertAlmostEqual(rows[-1][1][-1], .561)
        self.assertEqual(report['left']['samples'],3)
        self.assertEqual(report['left']['max_gap_ms'],20.)

    def test_missing_side_invalid_schema_nonfinite_and_reversed_time_rejected(self):
        m=self.module()
        for rows in ([joint(10**9,[0.]*21)],
                     [joint(10**9,[0.]*22, ['x']*22)],
                     [joint(10**9,[math.nan]+[0.]*21)],
                     [joint(10**9,[0.]*22),joint(10**9,[0.]*22)]):
            with self.subTest(rows=len(rows)):
                self.fixture()
                write_records(self.record/'-sharpa-left-state.pbdat',rows)
                with self.assertRaises(ValueError):m.load_feedback(self.root)
        self.fixture(); (self.record/'-sharpa-right-state.pbdat').unlink()
        with self.assertRaises(FileNotFoundError):m.load_feedback(self.root)

    def test_large_gap_and_uncovered_tail_are_not_silently_filled(self):
        self.fixture(stamps=(10**9,1010000000,1300000000)); m=self.module()
        with self.assertRaisesRegex(ValueError,'gap'):m.load_feedback(self.root)
        self.fixture(); streams=m.load_feedback(self.root)
        with self.assertRaisesRegex(ValueError,'coverage'):
            m.align_feedback(streams,[10**9,1200000000])

    def test_small_sampling_phase_offset_is_reported(self):
        self.fixture();m=self.module()
        rows,report=m.align_feedback(m.load_feedback(self.root),[995000000,1045000000])
        self.assertEqual(report['left']['edge_hold_ms'],5.)
        self.assertAlmostEqual(rows[0][0][0],0.)
        self.assertAlmostEqual(rows[-1][0][0],.04)

    def test_full_tf_not_truncated_at_36_seconds(self):
        m=self.module(); messages=[]
        for i in range(3701):
            msg=TFMessage()
            for frame in ('arm_l_end_link','arm_r_end_link'):
                t=msg.transforms.add();t.header.frame_id='base_link';t.child_frame_id=frame
                t.header.stamp.sec,t.header.stamp.nanosec=divmod(10**9+i*10**7,10**9)
                t.transform.rotation.w=1.
            messages.append(msg)
        write_records(self.record/'-tf.pbdat',messages)
        paths=m.load_full_paths(self.root)
        self.assertEqual(len(paths['left']),3701)
        self.assertAlmostEqual(paths['left'][-1][0]-paths['left'][0][0],37.)

    def test_selector_supports_name_and_legacy_path_rejects_escape_and_ambiguity(self):
        self.assertTrue(hasattr(replay,'resolve_episode'),'episode selector missing')
        ep=self.root/'试验_001';ep.mkdir();(ep/'record').mkdir()
        self.assertEqual(replay.resolve_episode(None,'试验_001',self.root),ep)
        self.assertEqual(replay.resolve_episode(ep,None,self.root),ep)
        for old,name in ((ep,'试验_001'),(None,'../other'),(None,None)):
            with self.assertRaises(ValueError):replay.resolve_episode(old,name,self.root)

    def test_dry_run_uses_real_files_without_importing_or_connecting_hardware(self):
        self.fixture()
        msgs=[]
        for i in range(5):
            msg=TFMessage()
            for frame in ('arm_l_end_link','arm_r_end_link'):
                t=msg.transforms.add();t.header.frame_id='base_link';t.child_frame_id=frame
                t.header.stamp.sec=1;t.header.stamp.nanosec=i*10**7
                t.transform.rotation.w=1.
            msgs.append(msg)
        write_records(self.record/'-tf.pbdat',msgs)
        body=JointState();body.name.extend(replay.BODY_NAMES+replay.JOINT_NAMES)
        body.motor_position.extend([0.]*22)
        write_records(self.record/'-hal-joint_state.pbdat',[body])
        output=io.StringIO()
        with patch.dict(sys.modules,{'agibot_gdk':None,'sharpa':None}), \
             patch.object(replay,'execute',side_effect=AssertionError('dry-run moved hardware')), \
             patch.object(replay,'control_locks',side_effect=AssertionError('dry-run took control')), \
             contextlib.redirect_stdout(output):
            replay.main([str(self.root)])
        self.assertIn('22+22',output.getvalue())
        self.assertIn('预检',output.getvalue())

    def test_nonfinite_first_body_frame_fails_before_hardware(self):
        msg=JointState();msg.name.extend(replay.BODY_NAMES+replay.JOINT_NAMES)
        msg.motor_position.extend([math.nan]+[0.]*21)
        write_records(self.record/'-hal-joint_state.pbdat',[msg])
        with self.assertRaisesRegex(ValueError,'Nonfinite'):replay.first_joint_frame(self.root)


class LifecycleTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('replay_feedback'))
        return importlib.import_module('replay_feedback')

    def test_existing_control_locks_exclude_replay_and_are_released_on_error(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            paths=[Path(root)/'pico.lock',Path(root)/'hands.lock']
            with m.control_locks(paths):
                with self.assertRaisesRegex(RuntimeError,'owned'):
                    with m.control_locks(paths):self.fail('concurrent owner allowed')
            with self.assertRaisesRegex(ValueError,'fixture'):
                with m.control_locks(paths):raise ValueError('fixture')
            with m.control_locks(paths):pass
            self.assertTrue(all(p.exists() for p in paths))

    def test_smooth_hand_entry_starts_at_measured_qpos_and_reaches_target(self):
        m=self.module()
        from fake_sharpa_sdk import SDK
        from sharpa_hands import SharpaHands
        hands=SharpaHands.from_config(Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json',sdk=SDK())
        hands.connect();self.addCleanup(hands.close)
        clock=[0.]
        with patch.object(m.time,'monotonic',lambda:clock[0]), \
             patch.object(m.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt)):
            m.move_hands_to_first(hands,([.2]*22,[-.1]*22),lambda:None)
        self.assertAlmostEqual(hands.read_qpos()[0][0],.2)
        self.assertAlmostEqual(hands.read_qpos()[1][0],-.1)
        commands=hands._sdk.waves['C8519039C852'].commands
        self.assertEqual(commands[0][0],[0.]*22)
        self.assertLessEqual(max(abs(b[0][0]-a[0][0]) for a,b in zip(commands,commands[1:])),.00301)

    def test_execute_closes_hands_if_body_replay_fails(self):
        self.assertTrue(hasattr(replay,'execute_connected'),'combined execution missing')
        class Hands:
            closed=False
            def connect(self):pass
            def close(self):self.closed=True
        hands=Hands()
        with patch('sharpa_hands.SharpaHands.from_config',return_value=hands), \
             patch.object(replay,'execute_connected',side_effect=RuntimeError('body failed')):
            with self.assertRaisesRegex(RuntimeError,'body failed'):
                replay.execute({},(),(),[],Path('unused'))
        self.assertTrue(hands.closed)

    def test_execute_closes_hands_on_keyboard_interrupt(self):
        class Hands:
            closed=False
            def connect(self):pass
            def close(self):self.closed=True
        hands=Hands()
        with patch('sharpa_hands.SharpaHands.from_config',return_value=hands), \
             patch.object(replay,'execute_connected',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):replay.execute({},(),(),[],Path('unused'))
        self.assertTrue(hands.closed)


class CombinedExecutionTests(unittest.TestCase):
    """Exercise the real combined loop and Sharpa adapter with SDK-only doubles."""
    def setUp(self):
        from fake_sharpa_sdk import SDK
        from sharpa_hands import SharpaHands
        self.sdk=SDK()
        self.hands=SharpaHands.from_config(Path(__file__).resolve().parents[1]/'configs/sharpa_hands.json',sdk=self.sdk)
        self.hands.connect();self.addCleanup(self.hands.close)
        self.now=0.;self.sent=[];self.body=[];self.first=[];self.fail_arm=False
        self.paths={s:[(1.+i*.01,(0.,0.,0.),(0.,0.,0.,1.)) for i in range(5)] for s in ('left','right')}
        self.hand_path=[([i*.01]*22,[-i*.02]*22) for i in range(5)]
        def pose():return NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))
        def arm(command):
            self.sent.append((self.now,command))
            return 1 if self.fail_arm else 0
        def body(req):self.body.append(req);return 0
        robot=NS(get_motion_control_status=lambda:NS(error_code=0,collision_pairs_1=[],collision_pairs_2=[]),
                 joint_control_request=body,end_effector_pose_control=arm)
        gdk=NS(gdk_init=lambda:0,GDKRes=NS(kSuccess=0),Robot=lambda:robot,TF=lambda:object(),
               JointControlReq=lambda:NS(),EndEffectorControlGroup=NS(kBothArms=12),
               EndEffectorPose=lambda:NS(left_end_effector_pose=pose(),right_end_effector_pose=pose()))
        def sleep(dt):self.now+=dt
        for p in (patch.dict(sys.modules,{'agibot_gdk':gdk}),
                  patch.object(replay.time,'monotonic',lambda:self.now),
                  patch.object(replay.time,'sleep',sleep),
                  patch.object(replay,'read_positions',lambda robot,names:(0.,)*len(names)),
                  patch.object(replay,'move_to_first_frame',lambda robot,q:self.first.append(tuple(q))),
                  patch.object(replay,'live_pose',lambda *a:((0.,0.,0.),(0.,0.,0.,1.)))):
            p.start();self.addCleanup(p.stop)

    def run_path(self):
        replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,self.hands)

    def test_body_head_first_frame_then_synchronized_dual_arm_and_44_hand_qpos(self):
        self.run_path()
        self.assertEqual(self.body[0].joint_names,list(replay.BODY_NAMES))
        self.assertEqual(self.first,[(0.,)*17])
        self.assertEqual(len(self.sent),5)
        self.assertEqual([round(t-self.sent[0][0],3) for t,_ in self.sent],[0.,.01,.02,.03,.04])
        self.assertTrue(all(cmd.group==12 for _,cmd in self.sent))
        for serial,expected in (('C8519039C852',.04),('CE569038CE55',-.08)):
            self.assertEqual(self.sdk.waves[serial].commands[-1][0],[expected]*22)

    def test_arm_failure_stops_hand_stream_before_that_frame(self):
        self.fail_arm=True
        with self.assertRaisesRegex(RuntimeError,'EEF command failed'):self.run_path()
        self.assertEqual(len(self.sent),1)
        # Only the 401 startup samples, not the failed EEF frame.
        self.assertEqual(len(self.sdk.waves['C8519039C852'].commands),401)

    def test_hand_failure_stops_further_arm_frames(self):
        from fake_sharpa_sdk import error
        original=self.hands.send_qpos
        def send(*a,**kw):
            if kw.get('frame_index')==1:self.sdk.waves['CE569038CE55'].send_error=error(7,'fixture')
            return original(*a,**kw)
        with patch.object(self.hands,'send_qpos',send), self.assertLogs('sharpa_hands',level='ERROR'):
            with self.assertRaisesRegex(RuntimeError,'SHARPA_SEND_FAILED'):self.run_path()
        self.assertEqual(len(self.sent),2)

    def test_hand_first_frame_not_reached_prevents_eef_replay(self):
        self.hand_path=[([.2]*22,[.2]*22)]*5
        with patch.object(self.hands,'read_qpos',return_value=([0.]*22,[0.]*22)):
            with self.assertRaisesRegex(RuntimeError,'first-frame error'):self.run_path()
        self.assertEqual(self.sent,[])

    def test_hand_startup_deadline_miss_prevents_eef_replay(self):
        def stalled_sleep(delay):self.now+=delay+.11
        with patch.object(replay.time,'sleep',stalled_sleep):
            with self.assertRaisesRegex(RuntimeError,'startup deadline'):self.run_path()
        self.assertEqual(self.sent,[])


if __name__=='__main__':unittest.main()
