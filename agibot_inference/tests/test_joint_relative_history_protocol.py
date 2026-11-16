"""Native history profile: real socket tests without robot/camera imports.

只验证协议和已有配对采集的复用，不导入或启动真机。
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COMMON=ROOT/'scripts' if (ROOT/'scripts/client_deployed78.py').exists() else ROOT/'src/infer/mixed78_common'
CONFIG=ROOT/'configs' if (ROOT/'configs').exists() else ROOT/'src/infer/mixed78_configs'
sys.path.insert(0,str(COMMON))
from test_abcd_protocol import Endpoint,ProtocolTests


class JointRelativeHistoryTests(unittest.TestCase):
    def info(self):
        return json.loads((ROOT/'tests/fixtures/joint_relative_history_info.json').read_text())

    def create(self, endpoint, task='book', factory=False):
        self.assertIsNotNone(importlib.util.find_spec('client_joint_relative_history'), 'paired history profile missing')
        from egoav78_contract import load_tasks
        tasks=load_tasks(CONFIG/'deployed_tasks78.json')
        args=dict(model='joint_relative_history',task=task,tasks=tasks,
                  embodiment='3b' if task in ('book','basket') else 'g2',timeout=2)
        if factory:
            from client_deployed78 import open_client78
            return open_client78(*endpoint.server.server_address,
                remote_config=CONFIG/'deployed_joint_relative_history.json',**args)
        from client_joint_relative_history import JointRelativeHistoryClient
        info=self.info()
        return JointRelativeHistoryClient(*endpoint.server.server_address,
            expected_config=dict(model='joint_relative_history',deployment_id=info['deployment_id'],
                                 checkpoint_sha256=info['checkpoint_sha256']),**args)

    def test_existing_paired_rpc_keeps_absolute_actions_for_both_robots(self):
        for task in (('book','yifu','drink') if COMMON==ROOT/'scripts' else ('book','basket')):
            endpoint=Endpoint(self.info())
            try:
                client=self.create(endpoint,task)
                try:
                    capture=ProtocolTests().history();packet=capture.snapshot()
                    client.attach_history(capture)
                    result=client.infer(packet['state'],packet['head_rgb'],task=task,
                        embodiment='3b' if task in ('book','basket') else 'g2',sample_ns=packet['sample_ns'])
                    np.testing.assert_array_equal(result[:,0],np.full(32,.25,np.float32))
                    self.assertEqual(client.last_history['valid_history_frames'],4)
                    self.assertEqual([r['cmd'] for r in endpoint.requests],['info','reset','infer'])
                finally:client.close()
            finally:endpoint.close()

    def test_wrong_history_or_rotation_or_weight_is_rejected_before_reset(self):
        for key,value in [('history_proprio',False),('variant','joint_relative_continuous13_current'),
                          ('checkpoint_sha256','a'*64),('rotation','first_two_matrix_rows')]:
            info=self.info();info[key]=value;endpoint=Endpoint(info)
            try:
                with self.assertRaises(ValueError):self.create(endpoint)
                self.assertEqual([r['cmd'] for r in endpoint.requests],['info'])
            finally:endpoint.close()

    def test_history_required_without_single_frame_fallback(self):
        endpoint=Endpoint(self.info())
        try:
            client=self.create(endpoint)
            try:
                packet=ProtocolTests().history().snapshot()
                with self.assertRaises(ValueError):
                    client.infer(packet['state'],packet['head_rgb'],task='book',embodiment='3b',sample_ns=packet['sample_ns'])
                self.assertEqual([r['cmd'] for r in endpoint.requests],['info','reset'])
            finally:client.close()
        finally:endpoint.close()

    def test_existing_factory_selects_native_history_profile(self):
        endpoint=Endpoint(self.info())
        try:
            client=self.create(endpoint,factory=True)
            try:self.assertTrue(client.info['history_proprio'])
            finally:client.close()
        finally:endpoint.close()

    def test_public_entry_routes_new_name_to_existing_execution_client(self):
        if COMMON==ROOT/'scripts':
            from test_abcd_real_routing import ABCDRealRoutingTests
            code,stdout,stderr=ABCDRealRoutingTests().launch('joint_relative_history','drink')
            self.assertEqual(code,0,stderr)
            self.assertIn('deployed_joint_relative_history.json',stdout)
            self.assertIn('--representation\ng2-camera78',stdout)
        else:
            sys.path.insert(0,str(ROOT))
            from src.infer.mixed78_client import parse_args
            args=parse_args(['--model','joint_relative_history','--task','book',
                             '--dry-run','--observation','unused-recorded.npz'])
            self.assertEqual(args.model,'joint_relative_history')
            self.assertEqual(args.remote_config.name,'deployed_joint_relative_history.json')
            self.assertFalse(args.execute)


if __name__=='__main__':unittest.main()
