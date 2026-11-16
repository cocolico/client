"""Real existing replay loop + fake SDK transport. Never initializes real GDK."""
import math
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from test_full_replay import CombinedExecutionTests
import replay_full_eef as replay
# Import this pure wrapper before fixtures snapshot sys.modules. Reloading
# OpenCV after patch.dict restores sys.modules breaks its native bootstrap.
import robot_wrapper
from genie_msgs_pb.msg.RetargetInfo_pb2 import RetargetInfo


class HeadExecutionTests(CombinedExecutionTests):
    def setup_combined(self,failed=False):
        self.published=[]
        gdk=sys.modules['agibot_gdk']
        def publish(msg,info):
            self.published.append((self.now,msg))
            return not failed
        gdk.Publisher=lambda *args:NS(publish=publish)
        gdk.PubMsgInfo=lambda:NS()
        gdk.QoS=lambda:NS()
        self.heads=[(i*.001,0.,0.) for i in range(5)]

    def test_new_head_path_uses_single_combined_packet_and_original_hand_clock(self):
        self.setup_combined()
        replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,
                                 self.hands,head_path=self.heads)
        self.assertEqual(self.sent,[])  # No competing end-effector channel.
        self.assertEqual(len(self.published),5)
        for i,(_,msg) in enumerate(self.published):
            groups={p.group_id:p for p in msg.retarget_groups_trajectory[0].retarget_infos}
            self.assertEqual(len(groups),5)
            self.assertEqual(groups[RetargetInfo.GROUP_HEAD_YAW].target_joint_positions[0],self.heads[i][0])
            for g in (RetargetInfo.GROUP_LEFT_ARM,RetargetInfo.GROUP_RIGHT_ARM):
                self.assertEqual(groups[g].control_type,RetargetInfo.ABS_POSE)
                self.assertEqual(groups[g].target_frame_poses[0].orientation.w,1.)
        self.assertEqual([round(t-self.published[0][0],3) for t,_ in self.published],[0,.01,.02,.03,.04])
        self.assertEqual(self.sdk.waves['C8519039C852'].commands[-1][0],[.04]*22)

    def test_combined_publish_failure_stops_before_hand_frame(self):
        self.setup_combined(failed=True)
        with self.assertRaisesRegex(RuntimeError,'combined'):
            replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,
                                     self.hands,head_path=self.heads)
        self.assertEqual(len(self.published),1)
        self.assertEqual(len(self.sdk.waves['C8519039C852'].commands),401)

    def test_wrong_head_length_rejected_before_gdk_init(self):
        self.setup_combined()
        with patch.object(sys.modules['agibot_gdk'],'gdk_init',
                          side_effect=AssertionError('initialized hardware for invalid path')):
            with self.assertRaises(ValueError):
                replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,
                                         self.hands,head_path=self.heads[:-1])

    def test_head_tracking_error_below_fifteen_degrees_is_allowed(self):
        self.setup_combined()
        self.heads=[(0.,0.,0.)]*4+[(math.radians(14.9),0.,0.)]
        replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,
                                 self.hands,head_path=self.heads)

    def test_head_tracking_error_above_fifteen_degrees_is_rejected(self):
        self.setup_combined()
        self.heads=[(0.,0.,0.)]*4+[(math.radians(15.1),0.,0.)]
        with self.assertRaisesRegex(RuntimeError,'exceeds 15.0deg'):
            replay.execute_connected(self.paths,(0.,)*5,(0.,)*17,self.hand_path,
                                     self.hands,head_path=self.heads)


if __name__=='__main__':unittest.main()
