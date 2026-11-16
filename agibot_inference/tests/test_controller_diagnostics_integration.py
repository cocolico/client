"""Fault propagation at real executor/CLI boundaries with fake device sinks only."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from controller_diagnostics import ControllerFault
from replay_full_eef import assert_ready
from test_controller_diagnostics import FakeRobot
from test_g2_chunk_camera import native
import g2_drink


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ControllerDiagnosticsIntegrationTests(unittest.TestCase):
    def drink_fixture(self, *, fault_after_sends):
        m = g2_drink
        adapter = m.DrinkAdapter(ROOT / 'assets/robot/agibot.urdf')
        body = np.array([-.3, .6, -.3, .05, .1])
        context = adapter.begin(native(), body, sample_ns=123)
        plan = adapter.plan(context, np.tile(context.state78, (4, 1)))
        joints = dict(zip(m.BODY_NAMES, body))
        joints.update(dict(zip(m.JOINT_NAMES, np.r_[native()[58:], np.zeros(14)])))
        sent, hand_sent = [], []
        clock = Clock()
        passive = FakeRobot(error_code=0)

        def status():
            passive.status.error_code = 2 if len(sent) >= fault_after_sends else 0
            return passive.status

        passive.get_motion_control_status = status

        class Publisher:
            def publish(self, message, info):
                sent.append(message)
                return True

        def tf(frame):
            offset = 0 if frame == 'arm_l_end_link' else 29
            return NS(translation=NS(**dict(zip(('x', 'y', 'z'), native()[offset:offset+3]))),
                      rotation=NS(x=0., y=0., z=0., w=1.))

        robot = NS(robot=passive,
                   tf=NS(get_tf_from_base_link=tf,
                         get_latest_timestamp=lambda frame: m.time.time_ns()),
                   gdk=NS(Publisher=lambda *args: Publisher(), QoS=lambda: None, PubMsgInfo=NS))
        hands = NS(require_connected=lambda: None,
                   send_qpos=lambda left, right, **kw: hand_sent.append((left, right, kw)))
        return NS(robot=robot, hands=hands, plan=plan, joints=joints,
                  sent=sent, hand_sent=hand_sent, clock=clock)

    def execute(self, fixture):
        with patch.object(g2_drink, 'read_positions',
                          side_effect=lambda robot, names: [fixture.joints[name] for name in names]), \
                patch.object(g2_drink.time, 'monotonic', fixture.clock.monotonic), \
                patch.object(g2_drink.time, 'sleep', fixture.clock.sleep):
            g2_drink.execute_plan(fixture.robot, fixture.plan, fixture.hands, fixture.joints)

    def test_drink_preflight_fault_sends_nothing_and_has_no_last_subframe(self):
        fixture = self.drink_fixture(fault_after_sends=0)
        with self.assertRaises(ControllerFault) as caught:
            self.execute(fixture)
        self.assertEqual(fixture.sent, [])
        self.assertEqual(fixture.hand_sent, [])
        self.assertFalse(hasattr(fixture.robot, '_drink_publisher'))
        self.assertEqual(caught.exception.execution['phase'], 'drink_preflight')
        self.assertIsNone(caught.exception.execution['failed_subframe'])
        self.assertIsNone(caught.exception.execution['last_sent_subframe'])

    def test_joint_drift_up_to_point_four_allows_execution(self):
        for delta in (.3, .4, -.4):
            with self.subTest(delta=delta):
                fixture = self.drink_fixture(fault_after_sends=999)
                current = dict(fixture.joints)
                current[g2_drink.JOINT_NAMES[3]] += delta
                with patch.object(g2_drink, 'read_positions',
                                  side_effect=lambda robot, names: [current[n] for n in names]), \
                        patch.object(g2_drink.time, 'monotonic', fixture.clock.monotonic), \
                        patch.object(g2_drink.time, 'sleep', fixture.clock.sleep):
                    g2_drink.execute_plan(fixture.robot, fixture.plan, fixture.hands, fixture.joints)
                self.assertEqual(len(fixture.sent), len(fixture.plan.body))
                self.assertEqual(len(fixture.hand_sent), len(fixture.plan.body))

    def test_joint_drift_reports_changed_joint_and_angles_before_any_send(self):
        fixture = self.drink_fixture(fault_after_sends=999)
        name = g2_drink.JOINT_NAMES[3]
        current = dict(fixture.joints)
        current[name] += .41
        with patch.object(g2_drink, 'read_positions',
                          side_effect=lambda robot, names: [current[n] for n in names]):
            with self.assertRaisesRegex(RuntimeError, 'Robot moved while inference was running') as caught:
                g2_drink.execute_plan(fixture.robot, fixture.plan, fixture.hands, fixture.joints)
        self.assertIn(name, str(caught.exception))
        self.assertIn('observed=0.000000rad', str(caught.exception))
        self.assertIn('current=0.410000rad', str(caught.exception))
        self.assertIn('delta=0.410000rad', str(caught.exception))
        self.assertIn('limit=0.400000rad', str(caught.exception))
        self.assertEqual(fixture.sent, [])
        self.assertEqual(fixture.hand_sent, [])
        self.assertFalse(hasattr(fixture.robot, '_drink_publisher'))

    def test_drink_monitoring_fault_stops_before_failing_subframe_target(self):
        fixture = self.drink_fixture(fault_after_sends=10)
        with self.assertRaisesRegex(RuntimeError, 'Drink subframe 10:.*fault/collision') as caught:
            self.execute(fixture)
        self.assertEqual(len(fixture.sent), 10)
        self.assertEqual(len(fixture.hand_sent), 10)
        self.assertEqual([item[2] for item in fixture.hand_sent], [
            {'frame_index': frame, 'subframe_index': subframe}
            for frame in range(2) for subframe in range(5)
        ])
        fault = caught.exception.__cause__
        self.assertIsInstance(fault, ControllerFault)
        self.assertEqual(fault.diagnostic['status']['error_code'], 2)
        self.assertEqual(fault.execution['phase'], 'drink_execution')
        self.assertEqual(fault.execution['failed_subframe'], 10)
        self.assertEqual(fault.execution['last_sent_subframe'], 9)
        self.assertEqual(fault.execution['action_frame'], 2)
        self.assertEqual(fault.execution['action_subframe'], 0)
        self.assertAlmostEqual(fault.execution['monitoring_interval_s'], .1)
        self.assertAlmostEqual(fixture.clock.now, .11)
        # The real validated DrinkPlan must survive JSON conversion, so both
        # the last sent and rejected target can be recovered after shutdown.
        from controller_diagnostics import record_controller_fault
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            path = record_controller_fault(caught.exception,
                context={'plan': fixture.plan}, directory=tmp)
            payload = json.loads(path.read_text())
        saved = payload['context']['plan']
        self.assertEqual(len(saved['body']), 20)
        self.assertEqual(saved['frame_indices'][10], [2, 0])
        for index in (payload['execution']['last_sent_subframe'],
                      payload['execution']['failed_subframe']):
            np.testing.assert_allclose(saved['body'][index], fixture.plan.body[index])
            np.testing.assert_allclose(saved['head'][index], fixture.plan.head[index])
            np.testing.assert_allclose(saved['hands'][index], fixture.plan.hands[index])
            for side in ('left', 'right'):
                np.testing.assert_allclose(saved['arm_base'][side][index][1],
                                           fixture.plan.arm_base[side][index][1])
        self.assertIn('ControllerFault', str(payload['exception_chain']))

    def test_cli_records_context_reraises_original_fault_and_closes_connections(self):
        import run_mixed76_client as runner
        import controller_diagnostics as diagnostics

        for storage in ('success', 'write_failure', 'serialization_failure'):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as tmp:
                events = []
                passive = FakeRobot()
                try:
                    assert_ready(passive)
                except ControllerFault as detected:
                    original_fault = detected
                else:
                    self.fail('Fixture did not produce a controller fault')
                state = native()
                obs = {'state': state, 'joints': {'fixture_joint': .125},
                       'images': {'head_color': np.zeros((8, 9, 3), np.uint8)}}

                def observe(hands):
                    events.append('observe')
                    return obs

                def execute(actions, measured, joints, hands, **kwargs):
                    events.append('execute')
                    np.testing.assert_allclose(actions, np.tile(state, (4, 1)), atol=1e-6)
                    raise original_fault

                robot = NS(robot=passive, get_observation61=observe,
                           execute_sharpa61=execute, last_image_timestamps={'head_color': 456})
                hands = NS(connect=lambda: events.append('hands_connect'),
                           close=lambda: events.append('hands_close'))

                def infer(model_state, image, **kwargs):
                    events.append('infer')
                    return np.tile(model_state, (32, 1))

                client = NS(info={'deployment_id': 'offline-fixture', 'horizon': 32},
                            last_response={'request_id': 'fixture-request', 'inference_seconds': .25},
                            infer=infer, close=lambda: events.append('client_close'))

                @contextmanager
                def connection(args):
                    events.append('connection_enter')
                    try:
                        yield ('offline-fixture', 0)
                    finally:
                        events.append('connection_exit')

                fake_modules = {
                    'run_ours_client': NS(startup_reset=lambda *args: None),
                    'robot_wrapper': NS(AgiBotRobot=lambda **kwargs: robot),
                    'sharpa_hands': NS(SharpaHands=NS(from_config=lambda *args: hands)),
                    'hand_startup': NS(load_hand_reset=lambda task: None,
                                       reset_hands_to_recording=lambda *args, **kwargs: None),
                }
                directory = Path(tmp) / 'records'
                if storage == 'write_failure':
                    directory.write_text('fixture obstacle')
                real_dumps = json.dumps

                def dumps(value, *args, **kwargs):
                    if (storage == 'serialization_failure' and isinstance(value, dict)
                            and value.get('event') == 'controller_fault'):
                        raise RuntimeError('fixture JSON encoding unavailable')
                    return real_dumps(value, *args, **kwargs)

                argv = ['client', '--inference', 'local', '--model', 'joint',
                        '--embodiment', 'g2', '--task', 'yifu', '--execute',
                        '--steps', '4', '--iterations', '2']
                stderr = io.StringIO()
                with ExitStack() as stack:
                    for manager in (
                        patch.object(sys, 'argv', argv),
                        patch.dict(sys.modules, fake_modules),
                        patch.object(runner, 'inference_connection', connection),
                        patch.object(runner, 'Mixed76Client', return_value=client),
                        patch.object(runner, 'CONTROLLER_FAULT_DIRECTORY', directory),
                        patch.object(diagnostics.json, 'dumps', side_effect=dumps),
                        redirect_stdout(io.StringIO()), redirect_stderr(stderr),
                    ):
                        stack.enter_context(manager)
                    with self.assertRaises(ControllerFault) as caught:
                        runner.main()
                self.assertIs(caught.exception, original_fault)
                self.assertEqual(events, ['connection_enter', 'hands_connect', 'observe', 'infer',
                                          'execute', 'hands_close', 'client_close', 'connection_exit'])
                if storage == 'success':
                    records = list(directory.glob('*.json'))
                    self.assertEqual(len(records), 1)
                    payload = json.loads(records[0].read_text())
                    self.assertEqual(payload['event'], 'controller_fault')
                    self.assertEqual(payload['diagnostic']['status']['error_code'], 2)
                    context = payload['context']
                    self.assertEqual(context['phase'], 'native61_execution')
                    self.assertEqual(context['model'], 'joint')
                    self.assertEqual(context['task'], 'yifu')
                    self.assertEqual(context['iteration'], 0)
                    self.assertEqual(context['steps'], 4)
                    self.assertEqual(context['max_response_age_s'], 6.)
                    self.assertEqual(context['server_info']['deployment_id'], 'offline-fixture')
                    self.assertEqual(context['observation']['joints'], {'fixture_joint': .125})
                    self.assertEqual(context['observation']['image_shapes']['head_color'], [8, 9, 3])
                    self.assertEqual(context['observation']['image_timestamps_ns']['head_color'], 456)
                    self.assertEqual(context['inference_response']['request_id'], 'fixture-request')
                    self.assertEqual(len(context['actions']), 32)
                    self.assertEqual(len(context['actions61']), 4)
                    self.assertIn('execute', payload['traceback'])
                    self.assertIn('ControllerFault', payload['traceback'])
                    self.assertIn('CONTROLLER_FAULT_RECORDED', stderr.getvalue())
                else:
                    self.assertIn('CONTROLLER_FAULT_RECORD_FAILED', stderr.getvalue())
                    if storage == 'write_failure':
                        self.assertEqual(directory.read_text(), 'fixture obstacle')


if __name__ == '__main__':
    unittest.main()
