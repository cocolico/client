"""Match deployment identities to the user's 2026-10-15 discovery evidence."""
import json
from pathlib import Path
import unittest


class DeploymentTests(unittest.TestCase):
    def test_config_selects_the_discovered_left_and_right_hand(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/sharpa_hands.json').read_text())
        discovered = {'C8519039C852': 'LEFT', 'CE569038CE55': 'RIGHT'}
        self.assertEqual(discovered.get(config['left']['serial']), 'LEFT')
        self.assertEqual(discovered.get(config['right']['serial']), 'RIGHT')


if __name__ == '__main__':
    unittest.main()
