"""No hardware: explicit endpoint selection and native contract routing."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
G2=(ROOT/'scripts/client_deployed78.py').exists()
COMMON=ROOT/'scripts' if G2 else ROOT/'src/infer/mixed78_common'
CONFIG=ROOT/'configs' if G2 else ROOT/'src/infer/mixed78_configs'
sys.path[:0]=[str(COMMON),str(ROOT)]
from client_deployed78 import open_client78
from egoav78_contract import load_tasks
class ReachedNetwork(RuntimeError):pass

class Routes(unittest.TestCase):
    def test_all_four_native_profiles_select_the_new_client_before_network(self):
        task='yifu' if G2 else 'book';kind='g2' if G2 else '3b'
        tasks=load_tasks(CONFIG/('g2_tasks78.json' if G2 else 'tasks.json'))
        for model in ['pi05','alpha','fastwam','groot']:
            with self.subTest(model=model),patch('socket.create_connection',side_effect=ReachedNetwork):
                with self.assertRaises(ReachedNetwork):
                    open_client78('127.0.0.1',1,remote_config=CONFIG/f'deployed_{model}.json',
                        model=model,task=task,tasks=tasks,embodiment=kind)
    @unittest.skipIf(G2,'Piper parser only')
    def test_fastwam_default_routes_to_pinned_deployment_and_preserves_executor_settings(self):
        from src.infer.mixed78_client import parse_args
        args=parse_args(['--model','fastwam','--task','book','--steps','32'])
        self.assertEqual(args.remote_config,CONFIG/'deployed_fastwam.json')
        self.assertTrue(args.execute);self.assertEqual(args.cycles,10000)
        self.assertEqual(args.control_hz,100);self.assertEqual(args.joint_filter_hz,6.)
if __name__=='__main__':unittest.main()
