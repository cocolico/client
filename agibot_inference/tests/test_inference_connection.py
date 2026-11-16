"""Connection selection and protocol regression tests; no robot/SDK access."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_ours_client as cli
from _ours_protocol import recv_message, send_message
from agibot_tasks import interface_spec
from inference_connection import inference_connection, SSHTunnel


class ConnectionTests(unittest.TestCase):
    def args(self, **kw):
        data = dict(inference='local', ip='127.0.0.1', port=19000, remote_config=None)
        data.update(kw)
        return NS(**data)

    def test_local_preserves_explicit_endpoint_without_ssh(self):
        with patch('inference_connection.subprocess.run', side_effect=AssertionError('local must not invoke SSH')):
            with inference_connection(self.args(ip='192.0.2.10', port=19009)) as address:
                self.assertEqual(address, ('192.0.2.10', 19009))

    def test_remote_unknown_server_port_fails_before_ssh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'remote.json'
            path.write_text(json.dumps(dict(ssh_host='b_dev', server_host='127.0.0.1', server_port=None)))
            with patch('inference_connection.subprocess.run', side_effect=AssertionError('no SSH before configuration')):
                with self.assertRaisesRegex(ValueError, 'server_port'):
                    with inference_connection(self.args(inference='remote', remote_config=path)):
                        self.fail('incomplete config must not connect')

    def test_failed_ssh_does_not_fall_back_to_local_or_retry(self):
        calls = []
        def failed(command, **kwargs):
            calls.append(command)
            return NS(returncode=255)
        with patch('inference_connection.subprocess.run', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'SSH'):
                with SSHTunnel('b_dev', '127.0.0.1', 19000):
                    self.fail('failed SSH must not yield an endpoint')
        self.assertEqual(len(calls), 1)

    def test_tunnel_cleanup_on_client_error_and_no_added_deadline(self):
        calls = []
        def ssh(command, **kwargs):
            self.assertNotIn('timeout', kwargs)
            self.assertFalse(any('ConnectTimeout=' in arg or 'ServerAlive' in arg for arg in command))
            calls.append(command)
            control = Path(command[command.index('-S') + 1])
            if '-O' not in command:
                control.touch()
            return NS(returncode=0)
        with patch('inference_connection.subprocess.run', side_effect=ssh):
            with self.assertRaisesRegex(RuntimeError, 'client failed'):
                with SSHTunnel('b_dev', '127.0.0.1', 19000) as address:
                    self.assertEqual(address[0], '127.0.0.1')
                    self.assertGreater(address[1], 0)
                    raise RuntimeError('client failed')
        self.assertEqual(len(calls), 2)
        self.assertIn('exit', calls[1])
        self.assertFalse(Path(calls[0][calls[0].index('-S') + 1]).parent.exists())

    def test_existing_client_reuses_one_socket_and_preserves_three_images_state_action(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(5)  # Test harness deadline, not a runtime policy.
        self.addCleanup(listener.close)
        expected = np.arange(50 * 61, dtype=np.float32).reshape(50, 61) / 1000
        obs = dict(state=np.arange(61, dtype=np.float32), images={
            key: np.full((24, 32, 3), value, np.uint8)
            for key, value in [('head_color', 23), ('left_wrist_camera', 57), ('right_wrist_camera', 91)]})
        received, errors = [], []
        def serve():
            try:
                conn, _ = listener.accept()
                with conn:
                    conn.settimeout(5)
                    self.assertEqual(recv_message(conn), {'cmd': 'info'})
                    send_message(conn, dict(interface_spec('yifu', 'pi05'), step=50000))
                    for _ in range(3):
                        request = recv_message(conn)
                        received.append(request['observation'])
                        send_message(conn, dict(actions=expected, inference_seconds=.01))
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with inference_connection(self.args(port=listener.getsockname()[1])) as address:
            client = cli.OursClient(*address, task='yifu', model='pi05')
            try:
                for _ in range(3):
                    np.testing.assert_array_equal(client.policy_infer(obs)['actions'], expected)
            finally:
                client._disconnect()
        thread.join(6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(received), 3)
        for item in received:
            np.testing.assert_array_equal(item['state'], obs['state'])
            for key in obs['images']:
                np.testing.assert_array_equal(item['images'][key], obs['images'][key])

    def test_failed_cleanup_keeps_control_path_for_manual_recovery(self):
        calls = []
        def ssh(command, **kwargs):
            calls.append(command)
            control = Path(command[command.index('-S') + 1])
            if '-O' not in command:
                control.touch()
            return NS(returncode=255 if len(calls) == 2 else 0)
        tunnel = SSHTunnel('b_dev', '127.0.0.1', 19000)
        with patch('inference_connection.subprocess.run', side_effect=ssh):
            tunnel.__enter__()
            try:
                with self.assertWarns(RuntimeWarning):
                    tunnel.__exit__(None, None, None)
                self.assertTrue(tunnel.control.exists())
            finally:
                tunnel.__exit__(None, None, None)


if __name__ == '__main__':
    unittest.main()
