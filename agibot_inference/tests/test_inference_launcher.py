"""Exercise the launcher with a fake Python boundary; never import robot code."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[1] / 'run_inference.sh'
        self.assertTrue(self.source.is_file(), 'root run_inference.sh is missing')
        self.tmp = tempfile.TemporaryDirectory(prefix='g2 launcher ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copy2(self.source, self.root / 'run_inference.sh')
        (self.root / 'bin').mkdir()
        (self.root / 'env.sh').write_text(
            'export LAUNCH_ENV_LOADED=yes\nexport PATH="$PWD/bin:$PATH"\n')
        fake = self.root / 'bin/python'
        fake.write_text('#!' + sys.executable + '\n'
            'import json, os, sys\nfrom pathlib import Path\n'
            'Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv":sys.argv[1:], '
            '"env":os.environ.get("LAUNCH_ENV_LOADED"), "cwd":os.getcwd()}))\n'
            'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\n')
        fake.chmod(0o755)
        self.capture = self.root / 'capture.json'

    def launch(self, args, exit_code=0):
        return subprocess.run(['bash', str(self.root / 'run_inference.sh'), *args],
            cwd='/', env={**os.environ, 'CAPTURE':str(self.capture),
                          'FAKE_EXIT':str(exit_code)}, capture_output=True, text=True)

    def test_forwards_three_options_and_fixed_execution_defaults(self):
        result = self.launch(['--model','pi05','--task','yifu','--steps','32'])
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(self.capture.read_text())
        self.assertEqual(payload['env'], 'yes')
        self.assertEqual(payload['cwd'], str(self.root))
        self.assertEqual(payload['argv'], [
            str(self.root / 'scripts/run_mixed76_client.py'),
            '--representation','g2-camera78','--model','pi05','--embodiment','g2',
            '--task','yifu','--inference','remote','--remote-config',
            str(self.root / 'configs/deployed_pi05.json'),
            '--iterations','100000','--steps','32','--execute'])

    def test_accepts_reordered_arguments_and_propagates_client_exit(self):
        result = self.launch(['--steps','8','--task','diguan','--model','fastwam'], 7)
        self.assertEqual(result.returncode, 7, result.stderr)
        argv = json.loads(self.capture.read_text())['argv']
        self.assertEqual(argv[argv.index('--steps')+1], '8')
        self.assertEqual(argv[argv.index('--task')+1], 'diguan')
        self.assertEqual(argv[argv.index('--model')+1], 'fastwam')

    def test_invalid_options_do_not_load_environment_or_start_client(self):
        (self.root / 'env.sh').write_text('touch env_was_loaded\nreturn 99\n')
        valid = ['--model','pi05','--task','yifu','--steps','32']
        for args in [[], valid[:-1], valid + ['--execute'], valid + ['--model','pi05'],
                     valid[:-1]+['0'], valid[:-1]+['33'], valid[:-1]+['x'],
                     ['--model','bad','--task','yifu','--steps','8']]:
            with self.subTest(args=args):
                result = self.launch(args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.capture.exists())
                self.assertFalse((self.root / 'env_was_loaded').exists())

    def test_help_does_not_load_environment(self):
        (self.root / 'env.sh').write_text('return 99\n')
        result = self.launch(['--help'])
        self.assertEqual(result.returncode, 0)
        self.assertIn('--model', result.stdout)
        self.assertFalse(self.capture.exists())

    def test_environment_failure_prevents_client_start(self):
        (self.root / 'env.sh').write_text('return 9\n')
        result = self.launch(['--model','pi05','--task','yifu','--steps','32'])
        self.assertEqual(result.returncode, 9)
        self.assertFalse(self.capture.exists())


if __name__ == '__main__':
    unittest.main()
