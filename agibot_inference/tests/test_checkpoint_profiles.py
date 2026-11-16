"""Run the real shell router with a recorder instead of Python; no robot IO.

用真实启动脚本验证步数路由；替换执行器边界，不启动网络或机器人。
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CheckpointProfileTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        recorder=root/'python';result=root/'argv.json';env_hit=root/'env_hit'
        recorder.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\n'
            +'Path('+repr(str(result))+').write_text(json.dumps(sys.argv[1:]))\n')
        recorder.chmod(0o755)
        script=(ROOT/'run_inference.sh').read_text();piper='CLIENT_PYTHON=' in script
        script=script.replace('/home/descfly/.venvs/fastwam-piper-5090/bin/python',str(recorder))
        (root/'run_inference.sh').write_text(script)
        (root/'env.sh').write_text('export PATH="'+str(root)+':$PATH"\ntouch "'+str(env_hit)+'"\n')
        configs=root/('src/infer/mixed78_configs' if piper else 'configs');configs.mkdir(parents=True)
        return root,configs,result,env_hit,piper

    def test_step_profiles_route_to_c0_lr_backend_and_exact_config(self):
        root,configs,result,env_hit,piper=self.fixture()
        for alias,filename in [('c0_lr_30k','deployed_c0_lr_30k.json'),
                               ('c0_lr_40k','deployed_c0_lr_40k.json'),
                               ('c0_lr_50k','deployed_c0_lr_50k.json'),
                               ('mid_a0','deployed_mid_a0.json'),
                               ('mid_a0_final','deployed_mid_a0_final.json'),
                               ('mid_a1_final','deployed_mid_a1_final.json'),
                               ('mid_b1_final','deployed_mid_b1_final.json'),
                               ('mid_a1','deployed_mid_a1.json'),
                               ('mid_b1','deployed_mid_b1.json'),
                               ('c0_lr','deployed_c0_lr.json')]:
            with self.subTest(alias=alias):
                (configs/filename).write_text('{}')
                output=subprocess.run(['bash',str(root/'run_inference.sh'),'--model',alias,
                    '--task','book' if piper else 'drink','--steps','32'],capture_output=True,text=True)
                self.assertEqual(output.returncode,0,output.stderr)
                argv=json.loads(result.read_text())
                self.assertEqual(argv[argv.index('--model')+1],'c0_lr')
                self.assertEqual(argv[argv.index('--steps')+1],'32')
                if not piper or alias!='c0_lr':
                    self.assertEqual(Path(argv[argv.index('--remote-config')+1]),configs/filename)
                if not piper:
                    self.assertIn('--execute',argv)
                    self.assertEqual(argv[argv.index('--iterations')+1],'100000')

    def test_missing_step_config_never_initializes_environment_or_client(self):
        root,configs,result,env_hit,piper=self.fixture()
        for alias in ('c0_lr_30k','c0_lr_40k','c0_lr_50k','mid_a0','mid_a1','mid_b1','mid_a0_final','mid_a1_final','mid_b1_final'):
            with self.subTest(alias=alias):
                output=subprocess.run(['bash',str(root/'run_inference.sh'),'--model',alias,
                    '--task','book' if piper else 'yifu','--steps','32'],capture_output=True,text=True)
                self.assertEqual(output.returncode,2,output.stderr)
                self.assertIn('部署配置',output.stderr)
                self.assertFalse(env_hit.exists())
                self.assertFalse(result.exists())


if __name__=='__main__':unittest.main()
