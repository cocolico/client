"""Offline launch routing; fake Python never imports hardware or clients."""
import argparse
import ast
from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ABCDRealRoutingTests(unittest.TestCase):
    def test_factory_passes_pinned_configuration_and_rejects_mask_changes(self):
        tree = ast.parse((ROOT/'scripts/client_deployed78.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'open_client78')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-factory>', 'exec'), namespace)
        tasks = json.loads((ROOT/'configs/deployed_tasks78.json').read_text())['tasks']
        modules = dict(egoav78_contract=types.SimpleNamespace(load_tasks=lambda path: tasks),
            client_abcd_history=types.SimpleNamespace(ABCDClient=lambda *args, **kwargs: types.SimpleNamespace(factory_kwargs=kwargs)))
        with patch.dict(sys.modules, modules):
            for model in ('model_a', 'model_b', 'model_c', 'model_d'):
                config_path = ROOT/'configs'/('deployed_'+model+'.json')
                config = json.loads(config_path.read_text())
                result = namespace['open_client78']('offline', config['server_port'],
                    remote_config=config_path, model=model, task='yifu', tasks=tasks, embodiment='g2')
                self.assertEqual(result.factory_kwargs['expected_config'], config)
                self.assertFalse(result.lossless_png)  # Direct/raw peer has no gateway handshake.
                changed = json.loads(json.dumps(tasks))
                changed['yifu']['validity'][0] = 0
                with self.assertRaisesRegex(ValueError, 'embodiment/mask'):
                    namespace['open_client78']('offline', config['server_port'],
                        remote_config=config_path, model=model, task='yifu', tasks=changed, embodiment='g2')

    def test_python_routes_four_models_and_rejects_unsupported_drink_before_connection(self):
        tree = ast.parse((ROOT/'scripts/run_mixed76_client.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        boundary = next(i for i, n in enumerate(main.body) if isinstance(n, ast.With))
        main.body = main.body[:boundary]+[ast.Return(value=ast.Name(id='args', ctx=ast.Load()))]
        namespace = dict(argparse=argparse, Path=Path, __doc__='offline argument routing',
            __file__=str(ROOT/'scripts/run_mixed76_client.py'), np=types.SimpleNamespace(isfinite=math.isfinite))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])),
                     '<actual-main-before-network>', 'exec'), namespace)
        modules = dict(egoav78_contract=types.SimpleNamespace(
            load_tasks=lambda path: {name: {'embodiment': 'g2'} for name in ('yifu', 'drink')}),
            client_deployed78=types.SimpleNamespace(open_client78=None))
        with patch.dict(sys.modules, modules):
            for model in ('model_a', 'model_b', 'model_c', 'model_d'):
                for task in ('yifu', 'drink'):
                    argv = ['client', '--representation', 'g2-camera78', '--model', model,
                            '--embodiment', 'g2', '--task', task]
                    with self.subTest(model=model, task=task), patch.object(sys, 'argv', argv):
                        if task == 'drink' and model in ('model_a', 'model_b'):
                            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                                namespace['main']()
                            self.assertEqual(caught.exception.code, 2)
                        else:
                            result = namespace['main']()
                            self.assertEqual(result.remote_config, ROOT/'configs'/('deployed_'+model+'.json'))
                            self.assertEqual(result.max_response_age, 6.)

    def launch(self, model, task):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT/'run_inference.sh', root/'run_inference.sh')
            (root/'env.sh').write_text('export PATH="$PWD/bin:$PATH"\n')
            (root/'bin').mkdir()
            executable = root/'bin/python'
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            result = subprocess.run(['bash', str(root/'run_inference.sh'), '--model', model,
                '--task', task, '--steps', '32'], capture_output=True, text=True)
            return result.returncode, result.stdout.replace(str(root), '<ROOT>'), result.stderr

    def test_four_models_use_existing_g2_execution_path(self):
        for model in ('model_a', 'model_b', 'model_c', 'model_d'):
            with self.subTest(model=model):
                code, stdout, stderr = self.launch(model, 'yifu')
                self.assertEqual(code, 0, stderr)
                self.assertEqual(stdout.splitlines(), [
                    '<ROOT>/scripts/run_mixed76_client.py', '--representation', 'g2-camera78',
                    '--model', model, '--embodiment', 'g2', '--task', 'yifu',
                    '--inference', 'remote', '--remote-config',
                    '<ROOT>/configs/deployed_'+model+'.json', '--iterations', '100000',
                    '--steps', '32', '--execute'])

    def test_only_c_and_d_accept_drink(self):
        for model in ('model_a', 'model_b'):
            with self.subTest(model=model):
                code, stdout, _ = self.launch(model, 'drink')
                self.assertEqual(code, 2)
                self.assertEqual(stdout, '')
        for model in ('model_c', 'model_d'):
            with self.subTest(model=model):
                code, _, stderr = self.launch(model, 'drink')
                self.assertEqual(code, 0, stderr)


if __name__ == '__main__':
    unittest.main()
