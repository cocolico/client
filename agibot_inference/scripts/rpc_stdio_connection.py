"""Local socket bridge to a single long-lived SSH command, owned by this context."""
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from inference_connection import SSHTunnel

class StdioConnection:
    def __init__(self,ssh_host,server_host,server_port,*,transport,compression=False):
        # Retain the existing host/port/type validation.
        SSHTunnel(ssh_host,server_host,server_port,compression=compression)
        if transport.get('codec')!='rgb_png_lossless_v1':raise ValueError('Unknown lossless transport codec')
        if transport.get('congestion') not in ('bbr','cubic'):raise ValueError('Unknown SSH congestion control')
        gateway=transport.get('gateway_script')
        if not isinstance(gateway,str) or not gateway.startswith('/'):raise ValueError('Absolute gateway_script required')
        self.host=ssh_host;self.target=server_host;self.port=server_port
        self.transport=transport;self.compression=compression
        self.process=None;self.listener=None;self.peer=None;self.directory=None;self.thread=None

    def __enter__(self):
        self.directory=Path(tempfile.mkdtemp(prefix='agibot-png-'))
        self.errors=(self.directory/'ssh.log').open('wb')
        try:
            self.listener=socket.socket();self.listener.bind(('127.0.0.1',0));self.listener.listen(1)
            helper=Path(__file__).with_name('ssh_tcp_fd.py')
            proxy=shlex.join([sys.executable,str(helper),'--congestion',self.transport['congestion']])+' %h %p'
            command=['ssh','-T','-o','ControlMaster=no','-o','ControlPath=none','-o','BatchMode=yes',
                '-o','StrictHostKeyChecking=yes','-o','ProxyUseFdpass=yes','-o','ProxyCommand='+proxy]
            if self.compression:command.insert(1,'-C')
            remote=shlex.join(['python3','-u',self.transport['gateway_script'],'--host',self.target,'--port',str(self.port)])
            command += [self.host,remote]
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,bufsize=0)
            self.thread=threading.Thread(target=self._bridge,daemon=True);self.thread.start()
            print(f'远端无损推理连接：{self.host} → {self.target}:{self.port}；PNG + {self.transport["congestion"]}，复用单个 SSH 会话',flush=True)
            return self.listener.getsockname()
        except BaseException:self.__exit__(None,None,None);raise

    def _bridge(self):
        try:
            self.peer,_=self.listener.accept()
            def upload():
                try:
                    while True:
                        data=self.peer.recv(262144)
                        if not data:break
                        view=memoryview(data)
                        while view:
                            n=self.process.stdin.write(view)
                            if not n:raise BrokenPipeError
                            view=view[n:]
                except (OSError,ValueError):pass
                finally:
                    try:self.process.stdin.close()
                    except (OSError,ValueError):pass
            writer=threading.Thread(target=upload,daemon=True);writer.start()
            try:
                while True:
                    data=self.process.stdout.read(262144)
                    if not data:break
                    self.peer.sendall(data)
            finally:
                self.peer.shutdown(socket.SHUT_RDWR)
                self.peer.close();writer.join(timeout=2)
        except (OSError,ValueError):pass

    def __exit__(self,exc_type,exc,tb):
        for item in (self.peer,self.listener):
            if item is not None:
                try:item.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                item.close()
        if self.process is not None:
            if self.process.poll() is None:self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
            self.process.stdin.close();self.process.stdout.close()
        if self.thread is not None:self.thread.join(timeout=3)
        if self.directory is not None:
            self.errors.close()
            detail=(self.directory/'ssh.log').read_text(errors='replace').strip()
            if detail and self.process is not None and self.process.returncode not in (0,-15):
                print('无损 SSH 连接日志：'+detail[-1500:],file=sys.stderr)
            shutil.rmtree(self.directory);self.directory=None
