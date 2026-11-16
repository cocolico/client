"""One SSH command session, one persistent GPU connection; CPU only, no open ports."""
import argparse
import json
import socket
import sys
import time
from _ours_protocol import MAX_BYTES, recv_exact
from rpc_png_transport import restore_request

def exact(stream,n):
    parts=[]
    while n:
        part=stream.read(n)
        if not part:raise EOFError
        parts.append(part);n-=len(part)
    return b''.join(parts)

def serve(source,sink,host,port):
    with socket.create_connection((host,port),timeout=300) as upstream:
        while True:
            try:
                first=exact(source,1);received_first=time.time_ns()
                n=int.from_bytes(first+exact(source,3),'big')
                if not 0<n<=MAX_BYTES:raise ValueError('Invalid request size')
                payload=exact(source,n);received_last=time.time_ns()
                start=time.monotonic();original=restore_request(payload);decode_s=time.monotonic()-start
                forwarded_first=time.time_ns()
                upstream.sendall(len(original).to_bytes(4,'big')+original)
                forwarded_last=time.time_ns()
                first=recv_exact(upstream,1);response_first=time.time_ns()
                size=int.from_bytes(first+recv_exact(upstream,3),'big')
                if not 0<size<=MAX_BYTES:raise ValueError('Invalid reply size')
                reply=json.loads(recv_exact(upstream,size));response_last=time.time_ns()
                timing=dict(received_first_ns=received_first,received_last_ns=received_last,
                    forwarded_first_ns=forwarded_first,forwarded_last_ns=forwarded_last,
                    response_first_ns=response_first,response_last_ns=response_last,
                    request_bytes=n,restored_request_bytes=len(original),lossless_decode_s=decode_s,
                    original_response_bytes=size,robot_io=False,codec='rgb_png_lossless_v1')
                reply['transport_timing']=timing
                timing['reply_encode_start_ns']=time.time_ns()
                raw=json.dumps(reply,allow_nan=False,separators=(',',':')).encode()
                if len(raw)>MAX_BYTES:raise ValueError('Reply too large')
                sink.write(len(raw).to_bytes(4,'big')+raw);sink.flush()
            except EOFError:return

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    a=p.parse_args()
    if not 1<=a.port<=65535:p.error('Invalid port')
    serve(sys.stdin.buffer,sys.stdout.buffer,a.host,a.port)

if __name__=='__main__':main()
