"""Shell launcher routing with a recorder executable, never a robot client."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class DeploymentLauncherTests(unittest.TestCase):
    def test_models_dispatch_to_matching_profile_and_preserve_execution_options(self):
        piper=(ROOT/'src/infer/mixed78_client.py').exists()
        source=ROOT/'run_inference.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); recorder=root/'python'; output=root/'args.json'
            recorder.write_text('#!'+sys.executable+'\nimport json,sys\n'
                +'open('+repr(str(output))+',"w").write(json.dumps(sys.argv[1:]))\n')
            recorder.chmod(0o755)
            shell=source.read_text().replace('/home/descfly/.venvs/fastwam-piper-5090/bin/python',str(recorder))
            (root/'run_inference.sh').write_text(shell)
            (root/'env.sh').write_text('export PATH="'+str(root)+':$PATH"\n')
            for model in ('tri_relative','joint_relative','joint_relative_history','joint','tri','alpha','pi05'):
                result=subprocess.run(['bash',str(root/'run_inference.sh'),
                    '--model',model,'--task','basket' if piper else 'yifu','--steps','32'],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                argv=json.loads(output.read_text())
                self.assertEqual(argv[argv.index('--model')+1],model)
                self.assertEqual(argv[argv.index('--steps')+1],'32')
                if piper:
                    sys.path.insert(0,str(ROOT))
                    from src.infer.mixed78_client import parse_args
                    args=parse_args(['--model',model,'--task','basket'])
                    self.assertTrue(args.execute)
                    self.assertEqual(args.control_hz,100)
                    self.assertEqual(args.joint_filter_hz,6.)
                    config=args.remote_config
                else:
                    self.assertIn('--execute',argv)
                    self.assertEqual(argv[argv.index('--iterations')+1],'100000')
                    config=ROOT/'configs'/Path(argv[argv.index('--remote-config')+1]).name
                info=json.loads(config.read_text())
                self.assertEqual(info['server_port'],dict(tri_relative=8888,joint_relative=18868,
                    joint_relative_history=8886,joint=8887,tri=18870,alpha=8885,pi05=18873)[model])
                self.assertEqual(info['server_host'],'10.0.45.36' if model in
                    ('alpha','joint','tri_relative','joint_relative_history') else '172.31.68.233')


if __name__=='__main__':unittest.main()
