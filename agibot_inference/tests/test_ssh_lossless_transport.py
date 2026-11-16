"""SSH transport configuration only; never import hardware or start a robot."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inference_connection import inference_connection,SSHTunnel

class LosslessTransportTests(unittest.TestCase):
    def exercise(self,config):
        calls=[]
        def launch(command,**kwargs):
            calls.append(command)
            if '-O' not in command:Path(command[command.index('-S')+1]).touch()
            return NS(returncode=0)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json';path.write_text(json.dumps(config))
            args=NS(inference='remote',remote_config=path)
            with patch('inference_connection.subprocess.run',side_effect=launch):
                with inference_connection(args) as address:
                    self.assertEqual(address[0],'127.0.0.1')
        self.assertEqual(len(calls),2)
        self.assertIn('exit',calls[1])
        return calls

    def test_history_profile_uses_lossless_ssh_compression(self):
        commands=self.exercise(dict(ssh_host='zw',server_host='10.0.0.153',server_port=8893,ssh_compression=True))
        self.assertIn('-C',commands[0])
        self.assertIn('127.0.0.1:',commands[0][-2])

    def test_legacy_profiles_keep_original_transport(self):
        commands=self.exercise(dict(ssh_host='zw',server_host='10.0.0.153',server_port=8893))
        self.assertNotIn('-C',commands[0])

if __name__=='__main__':unittest.main()
