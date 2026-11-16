import unittest
import numpy as np
import socket
import threading
from _ours_protocol import send_message, recv_message
from client_mixed76 import Mixed76Client
from mixed76_contract import encode_state, decode_actions, embodiment_mask, validate_actions


class ContractTests(unittest.TestCase):
    def test_tcp_client_keeps_76d_response(self):
        # Real framing/socket/client; only the expensive model is a fixture.
        listener=socket.socket(); listener.bind(('127.0.0.1',0)); listener.listen(1)
        expected=np.zeros((32,76),np.float32); expected[:,9:12]=[.1,.2,.3]
        def peer():
            with listener:
                connection,_=listener.accept()
                with connection:
                    self.assertEqual(recv_message(connection),{'cmd':'info'})
                    send_message(connection,dict(protocol_version=3,action_dim=76,state_dim=76,
                        representation='head12_wrist18_hand46_rot6d',action_units='physical',horizon=32))
                    request=recv_message(connection)
                    self.assertEqual(request['observation']['state'].shape,(76,))
                    send_message(connection,dict(actions=expected,embodiment='g2',
                        representation='head12_wrist18_hand46_rot6d',action_mask=embodiment_mask('g2').tolist()))
        thread=threading.Thread(target=peer,daemon=True);thread.start()
        client=Mixed76Client('127.0.0.1',listener.getsockname()[1])
        try:
            result=client.infer(np.zeros(76,np.float32),np.zeros((480,640,3),np.uint8),task='yifu',embodiment='g2')
            np.testing.assert_array_equal(result,expected)
        finally:
            client.close();thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_g2_identity_roundtrip_and_finger_slots(self):
        x = np.arange(61, dtype=np.float32) / 100
        x[3:7] = x[32:36] = [0, 0, 0, 1]
        y = encode_state(x, 'g2')
        np.testing.assert_array_equal(y[15:21], [1,0,0,0,1,0])
        np.testing.assert_allclose(y[30:52], x[7:29])
        np.testing.assert_allclose(decode_actions(y[None], 'g2')[0], x, atol=1e-6)
        self.assertEqual(embodiment_mask('g2').sum(), 65)

    def test_piper_rotation_roundtrip(self):
        x = np.zeros(23, np.float32)
        for start in [0,8,16]:
            x[start+3:start+7] = [0,0,np.sqrt(.5),np.sqrt(.5)]
        x[7], x[15] = 13, 42
        y = encode_state(x, '3b')
        np.testing.assert_allclose(y[15:21], [0,1,0,-1,0,0], atol=1e-6)
        np.testing.assert_allclose(decode_actions(y[None], '3b')[0], x, atol=1e-6)
        self.assertEqual(embodiment_mask('3b').sum(), 29)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            encode_state(np.zeros(61), 'g2')
        with self.assertRaises(ValueError):
            decode_actions(np.zeros((32,76)), 'g2')
        with self.assertRaises(ValueError):
            validate_actions(np.zeros((50,76)), 'g2', 32)
        x = np.zeros((32,76)); x[:,0] = 1
        with self.assertRaises(ValueError):
            validate_actions(x, 'g2', 32)
        x[:] = np.nan
        with self.assertRaises(ValueError):
            validate_actions(x, 'g2', 32)


if __name__ == '__main__':
    unittest.main()
