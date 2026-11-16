"""Run the actual main() with fake hardware boundaries; never initialize a robot."""
import argparse
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class RealHistoryLifecycleTests(unittest.TestCase):
    def run_main(self, model, task, failed_capture=False):
        events = []
        state78 = np.zeros(78, np.float32)
        rgb = np.zeros((32, 32, 3), np.uint8)
        obs = dict(state=np.zeros(61, np.float32), images={'head_color': rgb}, joints={'body': .2})
        context = types.SimpleNamespace(state71=np.zeros(71), state78=state78, base_from_camera=np.eye(4))
        class Adapter:
            def __init__(self, *args): pass
            def begin(self, *args, **kwargs): return context
            def plan(self, used_context, actions, **kwargs):
                self_outer.assertIs(used_context, context)
                return types.SimpleNamespace(report={})
        self_outer = self
        class Robot:
            def __init__(self, **kwargs):
                self.robot = object(); self.tf = object()
                self.last_image_timestamps = {'head_color': 17}
            def get_observation61(self, hands):
                events.append('observe')
                return obs
            def execute_sharpa61(self, actions, state, joints, hands, *, check, **kwargs):
                self_outer.assertIs(state, obs['state'])
                check()
                events.append('send')
        class Hands:
            @classmethod
            def from_config(cls, path): return cls()
            def connect(self): events.append('hands-connect')
            def close(self): events.append('hands-close')
        class Capture:
            def __init__(self, reader): self.reader = reader
            def __enter__(self):
                self.packet = self.reader(); events.append('capture-start'); return self
            def snapshot(self): return self.packet
            def check(self):
                events.append('capture-check')
                if failed_capture: raise RuntimeError('capture failed')
            def close(self): events.append('capture-close')
        class Client:
            info = {'horizon': 32}
            last_response = {'inference_seconds': .1}
            def attach_history(self, capture): self.capture = capture
            def infer(self, state, image, *, sample_ns, **kwargs):
                self_outer.assertIs(state, self.capture.packet['state'])
                self_outer.assertIs(image, self.capture.packet['head_rgb'])
                self_outer.assertEqual(sample_ns, self.capture.packet['sample_ns'])
                events.append('infer')
                return np.zeros((32, 78), np.float32)
            def close(self): events.append('client-close')
        @contextmanager
        def connection(args): yield ('offline', 1)
        def module(**kwargs): return types.SimpleNamespace(**kwargs)
        def drink_execute(robot, plan, hands, joints, *, check):
            check(); events.append('send')
        modules = dict(
            egoav78_contract=module(load_tasks=lambda path: {task: {'embodiment': 'g2'}},
                                   g2_to78=lambda state: state78),
            client_deployed78=module(open_client78=lambda *args, **kwargs: Client()),
            g2_chunk_camera=module(G2ChunkCamera=Adapter),
            g2_drink=module(DrinkAdapter=Adapter, execute_plan=drink_execute),
            replay=module(BODY_NAMES=('body',)),
            run_ours_client=module(startup_reset=lambda *args: events.append('startup-reset')),
            robot_wrapper=module(AgiBotRobot=Robot),
            sharpa_hands=module(SharpaHands=Hands),
            hand_startup=module(load_hand_reset=lambda task: None,
                               reset_hands_to_recording=lambda *args, **kwargs: None),
            replay_full_eef=module(assert_ready=lambda *args: None),
            paired_history=module(PairedHistory=Capture))
        tree = ast.parse((ROOT/'scripts/run_mixed76_client.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        namespace = dict(argparse=argparse, json=json, time=time, Path=Path, np=np,
            __file__=str(ROOT/'scripts/run_mixed76_client.py'), __doc__='offline lifecycle',
            inference_connection=connection, read_reset_body_target=lambda *args: [0]*5,
            restore_g2_actions=lambda *args, **kwargs: (np.zeros((1, 61)), None),
            record_controller_fault=lambda *args, **kwargs: events.append('fault'),
            CONTROLLER_FAULT_DIRECTORY=ROOT/'unused')
        exec(compile(ast.Module(body=[main], type_ignores=[]), '<actual-main>', 'exec'), namespace)
        argv = ['client', '--representation', 'g2-camera78', '--model', model,
                '--embodiment', 'g2', '--task', task, '--execute', '--steps', '1']
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv):
            if failed_capture:
                with self.assertRaisesRegex(RuntimeError, 'capture failed'): namespace['main']()
            else:
                namespace['main']()
        self.assertEqual(events.count('observe'), 1)
        self.assertEqual(events.count('capture-check'), 1)
        self.assertLess(events.index('capture-close'), events.index('hands-close'))
        self.assertLess(events.index('hands-close'), events.index('client-close'))
        self.assertEqual(events.count('send'), 0 if failed_capture else 1)

    def test_ordinary_and_drink_use_one_paired_packet_and_close_in_order(self):
        self.run_main('model_a', 'yifu')
        self.run_main('model_c', 'drink')

    def test_capture_failure_reaches_execution_check_before_any_command(self):
        self.run_main('model_d', 'yifu', failed_capture=True)


if __name__ == '__main__': unittest.main()
