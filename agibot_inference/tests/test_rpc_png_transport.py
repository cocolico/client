"""Lossless transport regressions: real framing, pixels, and corruption; no SDK."""
import base64, hashlib, importlib.util, json, socket, sys, threading, unittest, subprocess, shlex
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _ours_protocol import _encode, recv_exact
from client_deployed78 import Deployed78Client


class PNGTransportTests(unittest.TestCase):
    def codec(self):
        self.assertIsNotNone(importlib.util.find_spec('rpc_png_transport'), 'lossless codec not implemented')
        import rpc_png_transport
        return rpc_png_transport

    def request(self, equal_current=True):
        history=np.random.default_rng(9).integers(0,256,(5,32,40,3),dtype=np.uint8)
        current=history[-1].copy() if equal_current else np.zeros((32,40,3),np.uint8)
        return dict(cmd='infer',request_id='fixed',sample_ns=123,seed=42,task='g2/dengpao',
            observation=dict(state=np.linspace(-1,1,78,dtype=np.float32),head_rgb=current,
                head_rgb_history=history,history_is_pad=[False]*5))

    def test_roundtrip_restores_exact_original_wire_bytes(self):
        codec=self.codec()
        for equal in (True,False):
            request=self.request(equal)
            want=json.dumps(request,default=_encode,allow_nan=False,separators=(',',':')).encode()
            encoded=codec.encode_request(request)
            self.assertEqual(codec.restore_request(encoded),want)
            self.assertEqual(request['observation']['head_rgb'].shape,(32,40,3))

    def test_changed_pixel_hash_is_rejected_before_forwarding(self):
        codec=self.codec(); envelope=json.loads(codec.encode_request(self.request()))
        envelope['request']['observation']['head_rgb_history']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'pixel'):
            codec.restore_request(json.dumps(envelope).encode())

    def test_changed_nonimage_instruction_is_rejected(self):
        codec=self.codec(); envelope=json.loads(codec.encode_request(self.request()))
        envelope['request']['task']='g2/drink'
        with self.assertRaisesRegex(ValueError,'request'):
            codec.restore_request(json.dumps(envelope).encode())

    def test_declared_oversize_png_is_rejected(self):
        codec=self.codec(); envelope=json.loads(codec.encode_request(self.request()))
        envelope['request']['observation']['head_rgb_history']['shape']=[5,4096,4096,3]
        with self.assertRaisesRegex(ValueError,'shape|large|size'):
            codec.restore_request(json.dumps(envelope).encode())

    def test_control_request_and_legacy_bytes_stay_identical(self):
        codec=self.codec(); raw=b'{"cmd":"info"}'
        self.assertEqual(codec.encode_request(dict(cmd='info')),raw)
        self.assertEqual(codec.restore_request(raw),raw)

    def test_client_optin_transmits_png_and_returns_unchanged_response(self):
        codec=self.codec()
        for enabled in (False,True):
            left,right=socket.socketpair(); left.settimeout(3);right.settimeout(3)
            client=object.__new__(Deployed78Client);client.socket=left;client.lossless_png=enabled
            request=self.request(); received=[]; errors=[]
            def peer():
                try:
                    n=int.from_bytes(recv_exact(right,4),'big'); raw=recv_exact(right,n)
                    received.append(json.loads(raw))
                    restored=codec.restore_request(raw)
                    expected=json.dumps(request,default=_encode,allow_nan=False,separators=(',',':')).encode()
                    self.assertEqual(restored,expected)
                    right.sendall(b'\x00\x00\x00\x0b'+b'{"ok":true}')
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=peer);thread.start()
            try:self.assertEqual(client._rpc(request),dict(ok=True))
            finally:left.close();thread.join(4);right.close()
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(received[0].get('_transport_codec'), 'rgb_png_lossless_v1' if enabled else None)

    def test_stdio_session_restores_pixels_and_closes_only_owned_process(self):
        codec=self.codec()
        from rpc_stdio_connection import StdioConnection
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(5)
        received=[];errors=[]
        def server():
            try:
                conn,_=listener.accept()
                with conn:
                    conn.settimeout(5)
                    for _ in range(3):
                        n=int.from_bytes(recv_exact(conn,4),'big'); received.append(recv_exact(conn,n))
                        conn.sendall(b'\x00\x00\x00\x0b'+b'{"ok":true}')
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=server);worker.start()
        original_popen=subprocess.Popen;children=[]
        def local_ssh(command,**kwargs):
            remote=shlex.split(command[-1]);remote[0]=sys.executable
            child=original_popen(remote,**kwargs);children.append(child);return child
        gateway=str(Path(__file__).resolve().parents[1]/'scripts/rpc_png_gateway.py')
        connection=StdioConnection('zw','127.0.0.1',listener.getsockname()[1],
            transport=dict(codec='rgb_png_lossless_v1',congestion='bbr',gateway_script=gateway))
        with patch('rpc_stdio_connection.subprocess.Popen',side_effect=local_ssh):
            with connection as address:
                with socket.create_connection(address,timeout=5) as client:
                    for i in range(3):
                        request=self.request();request['sample_ns']=i+1
                        codec.send_message(client,request)
                        n=int.from_bytes(recv_exact(client,4),'big');reply=json.loads(recv_exact(client,n))
                        self.assertTrue(reply['ok']);self.assertFalse(reply['transport_timing']['robot_io'])
                        self.assertEqual(reply['transport_timing']['codec'],'rgb_png_lossless_v1')
                        self.assertEqual(received[-1],json.dumps(request,default=_encode,allow_nan=False,separators=(',',':')).encode())
        worker.join(6);listener.close()
        self.assertEqual(errors,[]);self.assertEqual(len(children),1)
        self.assertIsNotNone(children[0].poll());self.assertFalse(connection.thread.is_alive())


if __name__=='__main__':unittest.main()
