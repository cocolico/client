"""Exercise real shell/parser routing using inert sinks; no SDK or hardware."""
import json,sys,tempfile,unittest,subprocess,os
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
PROFILES=['pi05','alpha','groot','fastwam','c0_lr','mid_a0_final','mid_a1_final']
class ReachedBoundary(RuntimeError):pass

class G2Routes(unittest.TestCase):
    def test_seven_profiles_use_pinned_configs_and_original_execution_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'args.json'
            (root/'run_inference.sh').write_text((ROOT/'run_inference.sh').read_text())
            (root/'env.sh').write_text('export PATH="'+str(root)+':$PATH"\n')
            recorder=root/'python';recorder.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\nPath('+repr(str(output))+').write_text(json.dumps(sys.argv[1:]))\n');recorder.chmod(0o755)
            (root/'configs').mkdir()
            for profile in PROFILES:(root/'configs'/f'deployed_{profile}.json').write_text('{}')
            for profile in PROFILES:
                for task in ['yifu','drink']:
                    with self.subTest(profile=profile,task=task):
                        p=subprocess.run(['bash',str(root/'run_inference.sh'),'--model',profile,'--task',task,'--steps','32'],capture_output=True,text=True)
                        self.assertEqual(p.returncode,0,p.stderr)
                        args=json.loads(output.read_text())
                        self.assertEqual(Path(args[args.index('--remote-config')+1]),root/'configs'/f'deployed_{profile}.json')
                        self.assertEqual(args[args.index('--model')+1],'c0_lr' if profile.startswith('mid_') else profile)
                        self.assertIn('--execute',args)
                        self.assertEqual(args[args.index('--iterations')+1],'100000')
                        self.assertEqual(args[args.index('--steps')+1],'32')
    def test_fastwam_drink_reaches_connection_boundary_without_changing_deadline(self):
        import run_mixed76_client as client
        seen=[]
        def boundary(args):seen.append(args);raise ReachedBoundary()
        argv=['client','--representation','g2-camera78','--model','fastwam','--embodiment','g2','--task','drink','--execute']
        with patch.object(sys,'argv',argv),patch.object(client,'inference_connection',boundary):
            with self.assertRaises(ReachedBoundary):client.main()
        self.assertEqual(seen[0].remote_config,ROOT/'configs/deployed_fastwam.json')
        self.assertEqual(seen[0].max_response_age,6.)
if __name__=='__main__':unittest.main()
