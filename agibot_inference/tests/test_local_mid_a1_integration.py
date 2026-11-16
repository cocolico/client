"""Repository-local endpoint routing. No model, GDK or robot startup."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('local_connection_target', ROOT/'scripts/inference_connection.py')
connection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connection)


class LocalIntegrationTests(unittest.TestCase):
    def args(self, config):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)/'config.json'
        path.write_text(json.dumps(config))
        return SimpleNamespace(inference='remote', remote_config=path)

    def test_direct_tcp_does_not_invoke_ssh_or_gateway(self):
        args = self.args(dict(transport='tcp', server_host='127.0.0.1', server_port=19011))
        with patch.object(connection, 'SSHTunnel', side_effect=AssertionError('unexpected SSH')):
            with connection.inference_connection(args) as address:
                self.assertEqual(address, ('127.0.0.1', 19011))

    def test_original_non_gateway_ssh_unchanged(self):
        args = self.args(dict(ssh_host='b_dev', server_host='10.0.0.84', server_port=19011))
        with patch.object(connection, 'SSHTunnel') as tunnel:
            tunnel.return_value.__enter__.return_value = ('127.0.0.1', 45678)
            with connection.inference_connection(args) as address:
                self.assertEqual(address, ('127.0.0.1', 45678))
            tunnel.assert_called_once_with('b_dev','10.0.0.84',19011,compression=False)

    def test_invalid_port_fails_before_connect(self):
        for port in (0,65536,True,'19011'):
            with self.subTest(port=port), self.assertRaises(ValueError):
                with connection.inference_connection(self.args(dict(
                        transport='tcp',server_host='127.0.0.1',server_port=port))):
                    self.fail('invalid port accepted')

    def test_unknown_transport_never_falls_back_to_ssh(self):
        args = self.args(dict(transport='typo',ssh_host='b_dev',server_host='127.0.0.1',server_port=19011))
        with patch.object(connection,'SSHTunnel'), self.assertRaises(ValueError):
            with connection.inference_connection(args):
                self.fail('unknown transport accepted')

    def test_local_profile_and_remote_profile_use_original_g2_client(self):
        for profile in ('mid_a1_final', 'mid_a1_final_local'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                # Source a recorder env and replace exec before any real Python/GDK.
                (root/'run_inference.sh').write_text((ROOT/'run_inference.sh').read_text())
                (root/'env.sh').write_text("exec() { printf 'RECORDED_ARGV\\n'; printf '%s\\0' \"$@\"; exit 0; }\n")
                (root/'configs').mkdir()
                (root/f'configs/deployed_{profile}.json').write_text('{}')
                result = subprocess.run(['bash',str(root/'run_inference.sh'),'--model',profile,
                    '--task','yifu','--steps','32'],capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr.decode())
                argv = result.stdout.split(b'RECORDED_ARGV\n',1)[1].split(b'\0')[:-1]
                argv = [value.decode() for value in argv]
                self.assertEqual(Path(argv[1]).name,'run_mixed76_client.py')
                self.assertEqual(argv[argv.index('--model')+1],'c0_lr')
                self.assertEqual(Path(argv[argv.index('--remote-config')+1]).name,f'deployed_{profile}.json')
                self.assertIn('--execute',argv)
                self.assertEqual(argv[argv.index('--iterations')+1],'100000')


if __name__ == '__main__':
    unittest.main()
