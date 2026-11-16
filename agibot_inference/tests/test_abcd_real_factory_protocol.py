"""Actual factory and pinned server info, using an offline fake TCP endpoint."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class ABCDRealFactoryProtocolTests(unittest.TestCase):
    def test_real_factory_supports_all_allowed_g2_tasks_with_paired_proprio(self):
        import test_abcd_protocol as protocol
        from client_deployed78 import open_client78
        from egoav78_contract import load_tasks
        registry = load_tasks(ROOT/'configs/g2_tasks78.json')
        for port, model in ((8881, 'model_a'), (8882, 'model_b'),
                            (8883, 'model_c'), (8884, 'model_d')):
            info = protocol.fixture(port)
            info.update(proprio_history=False,conditioning='image_history_current_proprio')
            info['history'].update(proprio_history=False,
                observation_fields=['head_rgb_history','history_is_pad'],
                state_frame='current state uses the current request camera anchor; no past proprio')
            endpoint = protocol.Endpoint(info)
            try:
                for task in ('diguan', 'yifu', 'dengpao')+(() if port<8883 else ('drink',)):
                    with self.subTest(model=model, task=task):
                        # Use an isolated test deployment, never assume a live pod/IP or identity.
                        config=json.loads((ROOT/'configs'/('deployed_'+model+'.json')).read_text())
                        config.update(deployment_id=info['deployment_id'],checkpoint_sha256=info['checkpoint_sha256'],
                            proprio_history=False,conditioning='image_history_current_proprio',
                            tasks_file=str(ROOT/'configs/deployed_tasks78.json'))
                        with tempfile.TemporaryDirectory() as directory:
                            path=Path(directory)/'config.json';path.write_text(json.dumps(config))
                            client = open_client78(*endpoint.server.server_address,
                                remote_config=path,model=model,task=task,tasks=registry,embodiment='g2',timeout=2)
                        try:
                            history = protocol.ProtocolTests().history()
                            packet = history.snapshot()
                            client.attach_history(history)
                            result = client.infer(packet['state'], packet['head_rgb'],
                                task=task, embodiment='g2', sample_ns=packet['sample_ns'])
                            np.testing.assert_array_equal(result[:,0], np.full(32,.25,np.float32))
                            self.assertEqual(client.last_history['valid_history_frames'], 4)
                        finally:
                            client.close()
            finally:
                endpoint.close()


if __name__ == '__main__': unittest.main()
