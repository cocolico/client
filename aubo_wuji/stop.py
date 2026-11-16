"""Send SIGINT only to the exact client PID recorded in the selected run."""
import argparse
import os
from pathlib import Path
import signal
import time
from common import ROOT, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_directory',type=Path)
    args = p.parse_args(); path = args.run_directory/'session.json'
    state = read_json(path); pid = int(state['pid'])
    if state['status'] in ('completed','failed'): raise SystemExit('Session already ended: '+state['status'])
    try: command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError: raise SystemExit('Process has already exited')
    if str(ROOT/'run.py').encode() not in command or b'run' not in command:
        raise SystemExit('PID is not this client; no signal sent')
    os.kill(pid,signal.SIGINT)
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        now = read_json(path)
        if now.get('pid') != pid: raise SystemExit('Session identity changed')
        if now['status'] in ('completed','failed'):
            print('Client ended. Verify replay stop_results/cleanup_errors if execution had started.')
            return
        time.sleep(.1)
    raise SystemExit('Stop request sent; cleanup not yet confirmed. Check replay execution log.')


if __name__ == '__main__': main()
