"""Historical replay reads synthetic logs; no hardware or network dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import write_json,SIDES
from tests.test_pipeline import setup_case
from trajectory_replay import Recordings


class ReplayTests(unittest.TestCase):
    def test_full_model_sequence_and_two_cycle_time_alignment(self):
        cfg,models,frame=setup_case()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'monitor').mkdir()
            write_json(root/'configuration.json',cfg)
            write_json(root/'session.json',dict(status='failed'))
            lines=[]
            for cycle in range(2):
                name=f'cycle_{cycle:04d}';folder=root/name;folder.mkdir()
                np.savez(folder/'prediction.npz',keypoints=np.tile(frame['keypoints'],(32,1,1,1)))
                np.savez(folder/'observation.npz',head_rgb=frame['head_rgb'],keypoints=frame['keypoints'])
                arrays=dict(time_s=np.arange(33)*.0325)
                for s in SIDES:
                    arrays[s+'_joints']=np.tile(frame['initial'][s]['q'],(33,1))
                    arrays[s+'_poses']=np.tile(frame['initial'][s]['camera_T_root'],(33,1,1))
                np.savez(folder/'retarget.npz',**arrays)
                for step in range(33 if cycle==0 else 10):
                    lines.append(json.dumps(dict(cycle=name,unix_s=1000+cycle*10+step*.2,step=step,
                        points={'actual':frame['keypoints'].tolist()},joint_error_deg={s:[0]*20 for s in SIDES})))
            (root/'monitor/feedback.jsonl').write_text('\n'.join(lines)+'\n')
            records=Recordings(root);index=records.index()
            self.assertTrue(index['cycles'][0]['complete'])
            self.assertFalse(index['cycles'][1]['complete'])
            clip=records.clip('cycle_0001',2)
            self.assertEqual(clip['forecast_frames'],64)
            self.assertEqual(len(clip['raw']),66)
            self.assertEqual(np.asarray(clip['normalized']).shape,(66,2,21,3))
            self.assertEqual(np.asarray(clip['target']).shape,(66,2,21,3))
            self.assertEqual(len(clip['actual']),43)
            self.assertEqual(clip['steps'][33],33)
            self.assertEqual(clip['time_s'][33],10.)
            self.assertEqual(clip['actual_joint_rad'],[None]*43)
            full=records.clip('all')
            self.assertEqual(full['forecast_frames'],64)
            self.assertEqual(len(full['segments']),2)
            self.assertEqual(full['segments'][1]['start_s'],10.)
            self.assertEqual(full['steps'][33],33)

    def test_partial_final_log_line_is_not_treated_as_a_complete_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'monitor').mkdir()
            write_json(root/'session.json',dict(status='executing'))
            path=root/'monitor/feedback.jsonl'
            row=json.dumps(dict(cycle='cycle_0000',unix_s=1.,step=0,points={}))
            path.write_text(row[:20]);records=Recordings(root)
            self.assertEqual(records.index()['cycles'],[])
            with path.open('a') as f:f.write(row[20:]+'\n')
            self.assertEqual(records.index()['cycles'][0]['samples'],1)
            self.assertEqual(records.index()['cycles'][0]['samples'],1)


if __name__=='__main__':unittest.main()
