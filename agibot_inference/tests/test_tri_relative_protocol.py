"""Protocol-only regression: relative Tri must not compose the delta twice."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_deployed78 import Endpoint, Deployed78Tests, tasks


class TriRelativeTests(unittest.TestCase):
    def test_relative_tri_returns_absolute_actions_unchanged(self):
        helper = Deployed78Tests()
        endpoint = Endpoint(8868, info_edit=lambda info: info.update(model='tri'))
        try:
            for task, spec in tasks().items():
                with self.subTest(task=task):
                    client = helper.client(endpoint, 'tri_relative', task)
                    try:
                        result = client.infer(helper.state(), np.zeros((32, 32, 3), np.uint8),
                                              task=task, embodiment=spec['embodiment'])
                        self.assertEqual(result.shape, (32, 78))
                        np.testing.assert_array_equal(result[:, 0], np.full(32, .25, np.float32))
                    finally:
                        client.close()
        finally:
            endpoint.close()

    def test_absolute_tri_endpoint_rejected_for_relative_selection(self):
        endpoint = Endpoint(8870)
        try:
            with self.assertRaisesRegex(ValueError, 'model_action_mode'):
                Deployed78Tests().client(endpoint, 'tri_relative')
            self.assertEqual([r['cmd'] for r in endpoint.requests], ['info'])
        finally:
            endpoint.close()


if __name__ == '__main__':
    unittest.main()
