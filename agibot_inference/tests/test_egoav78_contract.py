import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class Contract78Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('egoav78_contract'), '78D contract is missing')
        import egoav78_contract as c
        self.c=c

    def test_g2_named_groups_roundtrip(self):
        x=np.arange(71,dtype=np.float32)+1
        y=self.c.g2_to78(x)
        np.testing.assert_array_equal(y[:9],x[:9])
        np.testing.assert_array_equal(y[9:18],x[31:40])
        np.testing.assert_array_equal(y[18:27],x[62:71])
        np.testing.assert_array_equal(y[32:54],x[9:31])
        np.testing.assert_array_equal(y[55:77],x[40:62])
        np.testing.assert_array_equal(y[[27,28,29,30,31,54,77]],0)
        np.testing.assert_array_equal(self.c.g2_from78(y),x)
        np.testing.assert_array_equal(self.c.g2_from78(self.c.g2_to78(np.tile(x,(32,1)))),np.tile(x,(32,1)))

    def test_piper_roundtrip_does_not_change_matrices_or_gripper_units(self):
        x=np.zeros((4,76),np.float32)
        x[:,:9]=np.arange(9)+1
        x[:,12:30]=np.arange(18)+101
        x[:,52]=17.5;x[:,75]=62.5
        y=self.c.piper_to78(x)
        np.testing.assert_array_equal(y[:,:9],x[:,12:21])
        np.testing.assert_array_equal(y[:,9:18],x[:,21:30])
        np.testing.assert_array_equal(y[:,18:27],x[:,:9])
        np.testing.assert_array_equal(y[:,[54,77]],x[:,[52,75]])
        np.testing.assert_array_equal(self.c.piper_from78(y),x)

    def test_registry_binds_instruction_mask_and_embodiment(self):
        tasks=self.c.load_tasks(ROOT/'configs/g2_tasks78.json')
        for name in ('yifu','diguan','dengpao'):
            self.assertEqual(tasks[name]['embodiment'],'g2')
            self.assertEqual(sum(tasks[name]['validity']),71)
        self.assertEqual(tasks['yifu']['instruction'],'晾衣服')
        self.assertEqual(tasks['basket']['embodiment'],'3b')
        self.assertEqual(sum(tasks['basket']['validity']),29)
        self.assertEqual(tasks['basket']['validity'],tasks['book']['validity'])

    def test_invalid_or_mismatched_masks_are_rejected(self):
        tasks=self.c.load_tasks(ROOT/'configs/g2_tasks78.json')
        tasks['yifu']['validity'][54]=1
        with self.assertRaises(ValueError):self.c.validate_tasks(tasks)
        with self.assertRaises(ValueError):self.c.g2_to78(np.zeros(78))
        x=np.zeros(78,np.float32);x[54]=1
        with self.assertRaises(ValueError):self.c.g2_from78(x)
        with self.assertRaises(ValueError):self.c.g2_to78(np.full(71,np.nan))


if __name__=='__main__':unittest.main()
