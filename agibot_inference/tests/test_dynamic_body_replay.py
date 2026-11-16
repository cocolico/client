import unittest
import sys
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from replay_public_eef import HeadKinematics, prepare
from robot_wrapper import AgiBotRobot, RetargetInfo
import replay_full_eef
import test_full_replay as fixtures

DATA=Path('/media/descfly/新加卷/sharpa_chanbing_20261021/trials/body_fk_004_20261021/chanbing_004_public71')

class DynamicBodyTests(unittest.TestCase):
    def test_arm_tracking_position_limit_is_point_one_meter(self):
        replay_full_eef.validate_arm_tracking('left',.1,30.)
        with self.assertRaisesRegex(RuntimeError,'0.101m'):
            replay_full_eef.validate_arm_tracking('left',.101,10.)
        with self.assertRaisesRegex(RuntimeError,'31.0deg'):
            replay_full_eef.validate_arm_tracking('right',.01,31.)

    def test_preflight_allows_recorded_eef_jumps_but_keeps_clock_checks(self):
        paths={side:[(0.,(0.,0.,0.),(0.,0.,0.,1.)),
                     (.01,(.1,0.,0.),(0.,0.,1.,0.))]
               for side in ('left','right')}
        replay_full_eef.preflight(paths)
        bad={side:list(points) for side,points in paths.items()}
        bad['right'][1]=(.04,bad['right'][1][1],bad['right'][1][2])
        with self.assertRaises(ValueError):replay_full_eef.preflight(bad)

    def test_004_original_speed_preflight_is_complete(self):
        plan=prepare(DATA,'action',speed=1.)
        self.assertEqual(len(plan['body_path']),6051)
        self.assertEqual(plan['paths']['left'][-1][0],60.5)
        self.assertEqual(plan['report']['effective_speed'],1.)

    def test_head_inverse_tracks_body_instead_of_first_frame(self):
        fk=HeadKinematics(DATA/'meta/robot.urdf')
        body=np.array([[-.3,.6,-.3,0,0],[-.3,.6,-.3,0,.2]])
        head=np.array([[.1,0,.1],[.1,0,.1]])
        poses=np.array([fk.head_pose(b,h) for b,h in zip(body,head)])
        np.testing.assert_allclose(fk.solve(poses,body),head,atol=1e-10)

    def test_public_base_targets_are_reexpressed_in_moving_arm_base(self):
        plan=prepare(DATA,'action',speed=1.)
        fk=HeadKinematics(DATA/'meta/robot.urdf')
        for i in (0,100,500,1000,3000,6050):
            values=dict(zip(replay_full_eef.BODY_NAMES,plan['body_path'][i]))
            base_from_arm=fk.pose(values,'arm_base_link')
            for side in ('left','right'):
                point=plan['arm_base_paths'][side][i]
                arm_from_target=np.eye(4)
                arm_from_target[:3,3]=point[1]
                arm_from_target[:3,:3]=Rotation.from_quat(point[2]).as_matrix()
                reconstructed=base_from_arm@arm_from_target
                original=plan['paths'][side][i]
                expected=np.eye(4)
                expected[:3,3]=original[1]
                expected[:3,:3]=Rotation.from_quat(original[2]).as_matrix()
                np.testing.assert_allclose(reconstructed,expected,atol=1e-9)

    def test_bad_body_alignment_rejected(self):
        self.assertTrue(hasattr(replay_full_eef,'validate_body_path'))
        validate_body_path=replay_full_eef.validate_body_path
        paths={'left':[0,1]}
        with self.assertRaises(ValueError):validate_body_path(paths,[0]*5,[[0]*5])
        with self.assertRaises(ValueError):validate_body_path(paths,[0]*5,[[.1]*5]*2)
        validate_body_path(paths,[0]*5,[[0]*5,[.001]*5])

    def test_004_action_replay_preserves_recorded_body_and_head(self):
        plan=prepare(DATA,'action',speed=.25)
        with np.load(DATA/'offline_replay.npz') as z:
            names=list(z['joint_names']);q=z['target_joints_rad']
            body=q[:,[names.index('idx0%d_body_joint%d'%(i,i)) for i in range(1,6)]]
            head=q[:,[names.index('idx%d_head_joint%d'%(10+i,i)) for i in range(1,4)]]
        np.testing.assert_allclose(np.array(plan['body_path'])[::20],body,atol=1e-12)
        np.testing.assert_allclose(np.array(plan['head_path'])[::20],head,atol=1e-6)
        self.assertEqual(len(plan['body_path']),24201)
        self.assertEqual(plan['paths']['left'][-1][0],242.)
        self.assertGreater(np.ptp(np.array(plan['body_path'])[:,4]),.01)

class DynamicExecutionTests(fixtures.CombinedExecutionTests):
    def test_each_frame_freezes_four_waist_joints_and_moves_only_yaw(self):
        gdk=sys.modules['agibot_gdk'];messages=[];topics=[]
        def publisher(topic,*args):
            topics.append(topic)
            return NS(publish=lambda msg,info:messages.append(msg) or True)
        gdk.Publisher=publisher
        gdk.PubMsgInfo=lambda:NS();gdk.QoS=lambda:None
        body=[(i*.02,i*.03,i*.04,i*.05,i*.01) for i in range(5)]
        head=[(i*.01,0.,0.) for i in range(5)]
        arm_base={side:list(points) for side,points in self.paths.items()}
        replay_full_eef.execute_connected(self.paths,(0.,)*5,(0.,)*17,
            self.hand_path,self.hands,head_path=head,body_path=body,
            arm_base_paths=arm_base)
        self.assertEqual(topics,['/egoav/teleop/retarget'])
        self.assertEqual(len(messages),5)
        self.assertEqual(self.body[0].joint_names,list(replay_full_eef.BODY_NAMES))
        for i,msg in enumerate(messages):
            groups={p.group_id:p for p in msg.retarget_groups}
            self.assertIn(RetargetInfo.GROUP_WAIST,groups)
            np.testing.assert_allclose(
                groups[RetargetInfo.GROUP_WAIST].target_joint_positions,
                [0.,0.,0.,0.,i*.01])
            self.assertAlmostEqual(groups[RetargetInfo.GROUP_HEAD_YAW].target_joint_positions[0],i*.01)
            self.assertIn(RetargetInfo.GROUP_LEFT_ARM,groups)
            self.assertIn(RetargetInfo.GROUP_RIGHT_ARM,groups)
            self.assertEqual(groups[RetargetInfo.GROUP_LEFT_ARM].frame_id,'arm_base_link')
            self.assertEqual(groups[RetargetInfo.GROUP_RIGHT_ARM].frame_id,'arm_base_link')
        self.assertEqual(self.sdk.waves['C8519039C852'].commands[-1][0],[.04]*22)

if __name__=='__main__':unittest.main()
