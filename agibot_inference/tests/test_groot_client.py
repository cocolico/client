"""GR00T G2 current-only TCP and argument routing; no robot imports or IO."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import test_deployed78 as existing
from client_deployed78 import Deployed78Client
ROOT=Path(__file__).resolve().parents[1]


def groot_info(info):
    info.update(model='groot',model_action_mode='absolute_in_optical_chunk_camera',
        experiment='aligned_camera_c0',conditioning='current_image_current_proprio',
        model_pose_convention='aligned_eef_axes_optical_c0',
        wire_pose_convention='native_eef_axes_head_end_anchor',history_images=False,history_proprio=False)


def registry(info):
    return {short:dict(embodiment='g2',instruction=info['tasks'][full]['prompt'],
            validity=list(map(int,info['tasks'][full]['mask'])))
        for short,full in [('yifu','g2/yifu'),('drink','g2/drink_head_only')]}


class GrootG2Tests(unittest.TestCase):
    def test_current_only_yifu_and_drink_use_native_actions_and_original_mask(self):
        e=existing.Endpoint(8869,info_edit=groot_info)
        try:
            for task in ('yifu','drink'):
                c=Deployed78Client(*e.server.server_address,model='groot',task=task,
                    tasks=registry(e.info),embodiment='g2',expected_deployment=e.info['deployment_id'])
                try:
                    state=existing.Deployed78Tests().state()
                    if task=='drink':state[27]=.15
                    actions=c.infer(state,np.zeros((32,32,3),np.uint8),task=task,embodiment='g2',sample_ns=123)
                    np.testing.assert_array_equal(actions[:,0],np.full(32,.25,np.float32))
                    np.testing.assert_array_equal(actions[:,27],np.full(32,.15 if task=='drink' else 0,np.float32))
                    self.assertEqual(set(e.requests[-1]['observation']),{'state','head_rgb'})
                    self.assertEqual(e.requests[-1]['task'],'g2/drink_head_only' if task=='drink' else 'g2/yifu')
                    self.assertEqual(c.last_response['action_mask'],[bool(x) for x in registry(e.info)[task]['validity']])
                finally:c.close()
        finally:e.close()

    def test_wrong_geometry_history_or_selected_drink_mask_rejected_before_reset(self):
        mutations=[lambda i:i.update(model_action_mode='absolute_in_chunk_camera'),
            lambda i:i.update(history_images=True),lambda i:i.update(history_proprio=True),
            lambda i:i.update(conditioning='image_history_current_proprio'),
            lambda i:i.update(wire_pose_convention='wrong'),
            lambda i:i['tasks']['g2/drink_head_only']['mask'].__setitem__(27,False)]
        for mutation in mutations:
            e=existing.Endpoint(8869,info_edit=groot_info)
            tasks=registry(e.info);mutation(e.info)
            try:
                with self.assertRaises(ValueError):
                    Deployed78Client(*e.server.server_address,model='groot',task='drink',tasks=tasks,
                        embodiment='g2',expected_deployment=e.info['deployment_id'])
                self.assertEqual([r['cmd'] for r in e.requests],['info'])
            finally:e.close()

    def test_legacy_drink_alias_still_applies(self):
        e=existing.Endpoint(8869)
        try:
            c=Deployed78Client(*e.server.server_address,model='joint',task='drink',tasks=registry(e.info),
                embodiment='g2',expected_deployment=e.info['deployment_id'])
            try:self.assertEqual(c.wire_task,'g2/drink_head_only')
            finally:c.close()
        finally:e.close()

    def test_python_routes_groot_drink_without_history_capture(self):
        tree=ast.parse((ROOT/'scripts/run_mixed76_client.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        boundary=next(i for i,n in enumerate(main.body) if isinstance(n,ast.With))
        main.body=main.body[:boundary]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='args',ctx=ast.Load()),ast.Name(id='paired_history',ctx=ast.Load())],ctx=ast.Load()))]
        namespace=dict(argparse=argparse,Path=Path,__doc__='offline routing',__file__=str(ROOT/'scripts/run_mixed76_client.py'),np=types.SimpleNamespace(isfinite=math.isfinite))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'<real-before-network>','exec'),namespace)
        modules=dict(egoav78_contract=types.SimpleNamespace(load_tasks=lambda p:{'drink':{'embodiment':'g2'}}),client_deployed78=types.SimpleNamespace(open_client78=None))
        argv=['client','--representation','g2-camera78','--model','groot','--embodiment','g2','--task','drink']
        with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv):
            args,history=namespace['main']()
        self.assertEqual(args.remote_config,ROOT/'configs/deployed_groot.json')
        self.assertFalse(history)

    def test_shell_routes_groot_drink_through_existing_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'args.json';recorder=root/'python'
            recorder.write_text('#!'+sys.executable+'\nimport json,sys\nopen('+repr(str(output))+',"w").write(json.dumps(sys.argv[1:]))\n');recorder.chmod(0o755)
            (root/'run_inference.sh').write_text((ROOT/'run_inference.sh').read_text())
            (root/'env.sh').write_text('export PATH="'+str(root)+':$PATH"\n')
            (root/'configs').mkdir();(root/'configs/deployed_groot.json').write_text('{}')
            result=subprocess.run(['bash',str(root/'run_inference.sh'),'--model','groot','--task','drink','--steps','32'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            argv=json.loads(output.read_text())
            self.assertEqual(argv[argv.index('--model')+1],'groot')
            self.assertEqual(Path(argv[argv.index('--remote-config')+1]),root/'configs/deployed_groot.json')
            self.assertIn('--execute',argv)
            self.assertEqual(argv[argv.index('--iterations')+1],'100000')


if __name__=='__main__':unittest.main()
