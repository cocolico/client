"""No hardware tests for public-frame 71D replay."""
import importlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

DATA=Path('/home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu')


class PublicReplayTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('replay_public_eef'),
                             'Missing 71D public replay adapter')
        return importlib.import_module('replay_public_eef')

    def test_complete_sample_is_prepared_offline_with_full_head_motion(self):
        m=self.api()
        with patch.dict(sys.modules,{'agibot_gdk':None,'sharpa':None}):
            plan=m.prepare(DATA)
        self.assertEqual(len(plan['head_path']),2696)
        self.assertEqual(len(plan['paths']['left']),2696)
        self.assertEqual(len(plan['hand_path']),2696)
        source=np.load(DATA/'offline_replay.npz')
        q=np.asarray(plan['head_path'])[::5]
        # Derived from saved EEF, independently compared to recorded named joints.
        np.testing.assert_allclose(q,source['measured_joints_rad'][:,5:8],atol=.001,rtol=0)
        self.assertTrue(np.all(np.ptp(q,axis=0)>.1))
        np.testing.assert_array_equal(np.asarray(plan['hand_path'])[::5,0],source['public_state'][:,9:31])
        np.testing.assert_array_equal(np.asarray(plan['hand_path'])[::5,1],source['public_state'][:,40:62])
        np.testing.assert_allclose(np.array([p[1] for p in plan['paths']['left']])[::5],
                                   source['public_state'][:,:3],atol=1e-12)
        self.assertLess(plan['report']['head_max_position_error_mm'],1.)
        self.assertLess(plan['report']['head_max_rotation_error_deg'],.1)

    def test_head_inverse_uses_eef_not_saved_joint_trace(self):
        m=self.api()
        fk=m.HeadKinematics(DATA/'meta/robot.urdf')
        body=np.array([-.3,.6,-.3,.05,.1])
        expected=np.array([.2,-.1,.3])
        target=fk.head_pose(body,expected)
        actual=fk.solve(target[None],body)
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)
        wrong=target.copy();wrong[0,3]+=.1
        with self.assertRaises(ValueError):fk.solve(wrong[None],body)
        for q in ([2.,0,0],[0.,.5,0],[0,0,.6]):
            with self.assertRaises(ValueError):fk.solve(fk.head_pose(body,q)[None],body)

    def test_head_limit_error_reports_joint_frame_and_overshoot(self):
        m=self.api()
        fk=m.HeadKinematics(DATA/'meta/robot.urdf')
        body=np.array([-.3,.6,-.3,.05,.1])
        cases=[(0,2.,'idx11_head_joint1',-1.5708,1.5708),
               (0,-2.,'idx11_head_joint1',-1.5708,1.5708),
               (1,.5,'idx12_head_joint2',-.3491,.3491),
               (1,-.5,'idx12_head_joint2',-.3491,.3491),
               (2,.6,'idx13_head_joint3',-.5236,.5236),
               (2,-.6,'idx13_head_joint3',-.5236,.5236)]
        for column,angle,joint,low,high in cases:
            with self.subTest(joint=joint,angle=angle):
                q=np.zeros(3);q[column]=angle
                targets=np.stack([fk.head_pose(body,np.zeros(3)),fk.head_pose(body,q)])
                original=targets.copy()
                with self.assertRaisesRegex(ValueError,'Head EEF inverse exceeds URDF joint limits') as caught:
                    fk.solve(targets,body,position_diagnostic_only=True)
                detail=getattr(caught.exception,'diagnostic',None)
                self.assertIsInstance(detail,dict,'Missing actionable head limit diagnostic')
                self.assertEqual(detail['frame_index_base'],0)
                self.assertEqual(detail['first_bad_frame'],1)
                self.assertEqual(detail['joint'],joint)
                self.assertAlmostEqual(detail['predicted_rad'],angle)
                self.assertAlmostEqual(detail['predicted_deg'],float(np.rad2deg(angle)))
                np.testing.assert_allclose(detail['allowed_rad'],[low,high])
                self.assertAlmostEqual(detail['excess_rad'],abs(angle)-high)
                self.assertEqual(detail['violating_frames'],1)
                self.assertEqual(detail['violating_joint_values'],1)
                np.testing.assert_allclose(detail['frame_body_rad'],body)
                np.testing.assert_allclose(detail['frame_head_rad'],q,atol=1e-12)
                np.testing.assert_array_equal(targets,original)
                # The diagnostic must also appear in the uncaught traceback.
                import json
                rendered=str(caught.exception).split(': ',1)[1]
                self.assertEqual(json.loads(rendered),detail)

    def test_rot6d_order_and_invalid_values(self):
        m=self.api()
        np.testing.assert_allclose(m.rotation6d([[0,1,0,-1,0,0]])[0],
                                  [[0,-1,0],[1,0,0],[0,0,1]],atol=1e-12)
        for a in ([0]*6,[1,0,0,1,0,0],[float('nan'),0,0,0,1,0]):
            with self.assertRaises(ValueError):m.rotation6d([a])


if __name__=='__main__':unittest.main()
