"""Current v4 deployments must share one transport default, without model whitelists."""
import importlib,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class TransportDefaults(unittest.TestCase):
    def resolve(self,config):
        module=importlib.import_module('rpc_png_transport')
        self.assertTrue(callable(getattr(module,'transport_config',None)), 'unified transport default is missing')
        return module.transport_config(config)

    def test_new_history_and_native_models_do_not_fall_back_to_raw_upload(self):
        for wire in ['egoav78_aligned_c0_v4','egoav78_native_post_v4','egoav78_v4','egoav78_c13_v4','egoav78_abcd_history_v4','egoav78_paired_history_v4']:
            for name in ['ours_direct_post','mid_a2_final','arbitrary_future_model']:
                cfg=dict(ssh_host='zw',wire_protocol=wire,model=name)
                actual=self.resolve(cfg)
                self.assertIsInstance(actual,dict)
                self.assertEqual(actual['codec'],'rgb_png_lossless_v1')
                self.assertEqual(actual['congestion'],'bbr')
                self.assertNotIn('lossless_transport',cfg)

    def test_explicit_transport_remains_authoritative_and_is_not_mutated(self):
        config=dict(ssh_host='zw',wire_protocol='egoav78_aligned_c0_v4',lossless_transport=dict(
            codec='rgb_png_lossless_v1',congestion='cubic',gateway_script='/reviewed/custom.py'))
        actual=self.resolve(config);self.assertEqual(actual,config['lossless_transport'])
        actual['congestion']='bbr';self.assertEqual(config['lossless_transport']['congestion'],'cubic')

    def test_explicit_disable_and_legacy_or_other_hosts_keep_original_transport(self):
        for cfg in [dict(ssh_host='zw',wire_protocol='egoav78_aligned_c0_v4',lossless_transport=False),
                    dict(ssh_host='zw'),dict(ssh_host='b_dev',wire_protocol='egoav78_aligned_c0_v4')]:
            self.assertIsNone(self.resolve(cfg))

if __name__=='__main__':unittest.main()
