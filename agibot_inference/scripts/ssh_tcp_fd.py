"""Experimental SSH fd-pass helper: congestion control for this socket only."""
import argparse
import array
import socket

p = argparse.ArgumentParser()
p.add_argument('--congestion', choices=['cubic', 'bbr'], required=True)
p.add_argument('host')
p.add_argument('port', type=int)
a = p.parse_args()
if not 1 <= a.port <= 65535:
    p.error('Invalid port')
with socket.create_connection((a.host, a.port), timeout=10) as upstream:
    upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, a.congestion.encode())
    upstream.settimeout(None)
    with socket.socket(fileno=1) as channel:
        channel.sendmsg([b'\0'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                   array.array('i', [upstream.fileno()]))])
