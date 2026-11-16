"""Offline launcher/routing checks. Real env, client and hardware never start."""
import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class C13RoutingTests(unittest.TestCase):
    def test_existing_five_model_launcher_arguments_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'env.sh').write_text('export PATH="$PWD/bin:$PATH"\n')
            (root/'bin').mkdir()
            python = root/'bin/python'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            for model in ('joint_relative', 'joint', 'tri', 'tri_relative', 'alpha'):
                command = ['--model', model, '--task', 'yifu', '--steps', '32']
                outputs = []
                for source in (ROOT/'before/run_inference.sh', ROOT/'run_inference.sh'):
                    # The baseline snapshot is needed only in this staging tree.
                    if not source.exists():
                        self.skipTest('baseline snapshot is local staging evidence')
                    shutil.copyfile(source, root/'run_inference.sh')
                    result = subprocess.run(['bash', str(root/'run_inference.sh'), *command],
                        text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs.append(result.stdout)
                self.assertEqual(outputs[0], outputs[1], model)

    def test_factory_selects_c13_adapter_and_preserves_task_mask_validation(self):
        tree = ast.parse((ROOT/'scripts/client_deployed78.py').read_text())
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'open_client78')
        namespace = {}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), '<actual-factory>', 'exec'), namespace)
        selected = json.loads((ROOT/'configs/deployed_tasks78.json').read_text())['tasks']
        contract = types.ModuleType('egoav78_contract')
        contract.load_tasks = lambda path: selected
        adapter = types.ModuleType('client_joint_relative_c13')
        adapter.C13Client = lambda *args, **kwargs: (args, kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,
                egoav78_contract=contract, client_joint_relative_c13=adapter):
            config = Path(directory)/'deployed_joint_relative_c13.json'
            config.write_text(json.dumps(dict(wire_protocol='egoav78_c13_v4',
                model='joint_relative_c13', tasks_file='tasks.json',
                deployment_id='reviewed-c13', expected_experiment='relative')))
            args, kwargs = namespace['open_client78']('127.0.0.1', 18874,
                remote_config=config, model='joint_relative_c13', task='yifu',
                tasks=selected, embodiment='g2')
            self.assertEqual(args, ('127.0.0.1', 18874))
            self.assertEqual(kwargs['expected_deployment'], 'reviewed-c13')
            self.assertEqual(kwargs['expected_experiment'], 'relative')
            changed = json.loads(json.dumps(selected))
            changed['yifu']['validity'][0] = 0
            with self.assertRaisesRegex(ValueError, 'embodiment/mask'):
                namespace['open_client78']('127.0.0.1', 18874, remote_config=config,
                    model='joint_relative_c13', task='yifu', tasks=changed, embodiment='g2')

    def test_launcher_routes_c13_tasks_through_existing_execution_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT/'run_inference.sh', root/'run_inference.sh')
            (root/'env.sh').write_text('export PATH="$PWD/bin:$PATH"\n')
            (root/'bin').mkdir()
            python = root/'bin/python'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            for task in ('yifu', 'drink'):
                with self.subTest(task=task):
                    result = subprocess.run(['bash', str(root/'run_inference.sh'),
                        '--model', 'joint_relative_c13', '--task', task, '--steps', '32'],
                        text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), [
                        str(root/'scripts/run_mixed76_client.py'),
                        '--representation', 'g2-camera78', '--model', 'joint_relative_c13',
                        '--embodiment', 'g2', '--task', task, '--inference', 'remote',
                        '--remote-config', str(root/'configs/deployed_joint_relative_c13.json'),
                        '--iterations', '100000', '--steps', '32', '--execute'])

    def test_python_argument_routing_selects_c13_configuration_before_network(self):
        # Execute the actual main() up to its network boundary. Imports below
        # that boundary are never reached; pure registry loading uses JSON only.
        tree = ast.parse((ROOT/'scripts/run_mixed76_client.py').read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        first_connection = next(i for i, node in enumerate(main.body) if isinstance(node, ast.With))
        main.body = main.body[:first_connection] + [ast.Return(value=ast.Name(id='args', ctx=ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
        namespace = dict(argparse=argparse, Path=Path, __file__=str(ROOT/'scripts/run_mixed76_client.py'),
                         __doc__='offline routing check', np=types.SimpleNamespace(isfinite=math.isfinite))
        exec(compile(module, '<actual-main-before-network>', 'exec'), namespace)
        contract = types.ModuleType('egoav78_contract')
        contract.load_tasks = lambda path: json.loads((ROOT/'configs/deployed_tasks78.json').read_text())['tasks']
        factory = types.ModuleType('client_deployed78')
        factory.open_client78 = None
        with patch.dict(sys.modules, egoav78_contract=contract, client_deployed78=factory):
            for task in ('yifu', 'drink'):
                with self.subTest(task=task), patch.object(sys, 'argv', ['client', '--representation',
                        'g2-camera78', '--model', 'joint_relative_c13', '--embodiment', 'g2',
                        '--task', task, '--steps', '32']):
                    args = namespace['main']()
                    self.assertEqual(args.remote_config, ROOT/'configs/deployed_joint_relative_c13.json')
                    self.assertEqual(args.steps, 32)


if __name__ == '__main__':
    unittest.main()
