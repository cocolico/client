"""Stop only this local Pi0.5 inference server, never a robot client."""
from pathlib import Path
import json
import os
import signal

root=Path(__file__).resolve().parent
endpoint=root/'endpoint.json'
if not endpoint.exists():raise SystemExit('No local Pi0.5 server endpoint')
pid=json.loads(endpoint.read_text())['pid']
cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
if str(root/'server.py').encode() not in cmdline:
    raise SystemExit('PID does not belong to this Pi0.5 server; refusing to signal')
os.kill(pid,signal.SIGTERM)
print('Requested local Pi0.5 server stop:',pid)
