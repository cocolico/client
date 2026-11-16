#!/usr/bin/env python3
"""Loopback-only Pi0.5 -> xflow_kp21_chunk_v1 adapter. No robot SDK imports."""
from pathlib import Path
from collections import OrderedDict
import argparse
import json
import os
import signal
import socketserver
import sys
import threading
import time
import uuid
import numpy as np
from runtime import ROOT,CHECKPOINT,MODEL_NAME,TIMES,Runtime

sys.path.insert(0,str(ROOT.parent/'aubo_wuji/vendor'))
from wire_v4 import recv_message,send_message

PROTOCOL='xflow_kp21_chunk_v1'
TASKS={
    'flower':{'instruction':'Flower arrangement: Pick up the flower on the left with the left hand and insert it into the vase, then pick up the flower on the right with the right hand and insert it into the vase.'},
    'tidying':{'instruction':'Table tidying: Pick up the carrot on the left with the left hand and place it in the basket in the middle, then pick up the carrot on the right with the right hand and place it in the basket in the middle.'},
    'lemon':{'instruction':'Lemon picking with tongs: Hold the cup steady with the left hand, grab the tongs with the right hand, pick up the lemon with the tongs, and place it in the cup in the middle.'},
    'whiteboard':{'instruction':'Whiteboard wiping: Hold the whiteboard steady with the left hand, pick up the eraser with the right hand, and wipe the writing off the whiteboard.'}}


class Adapter:
    def __init__(self,runtime,record_dir):
        self.runtime=runtime;self.record_dir=Path(record_dir)
        self.record_dir.mkdir(parents=True,exist_ok=True)
        self.sessions=OrderedDict();self.lock=threading.Lock()

    def dispatch(self,msg):
        with self.lock:
            cmd=msg.get('cmd')
            if cmd=='info':
                return dict(protocol=PROTOCOL,protocol_version=4,horizon=32,default_model=MODEL_NAME,
                    models={MODEL_NAME:self.runtime.describe()},tasks=TASKS,
                    note='Pi0.5 predicts 50 steps; this adapter exposes the first 32 unchanged in space/time')
            if cmd=='reset':
                session=uuid.uuid4().hex;self.sessions[session]=dict(last_ns=-1,request_ids=set())
                while len(self.sessions)>64:self.sessions.popitem(last=False)
                return dict(request_id=msg.get('request_id'),session_id=session)
            if cmd!='infer':raise ValueError('Unsupported command')
            session=self.sessions.get(msg.get('session_id'))
            if session is None:raise ValueError('Unknown session; reset required')
            sample=msg.get('sample_ns');request=msg.get('request_id')
            if type(sample) is not int or sample<=session['last_ns'] or not isinstance(request,str) or not request or request in session['request_ids']:
                raise ValueError('Invalid, duplicate or out-of-order inference request')
            if msg.get('model')!=MODEL_NAME:raise ValueError('Model identity mismatch')
            task=msg.get('task')
            if task not in TASKS:raise ValueError('Unknown task')
            session['last_ns']=sample;session['request_ids'].add(request)
            if len(session['request_ids'])>10000:session['request_ids']={request}
            points,diagnostic=self.runtime.infer(msg['observation'],TASKS[task]['instruction'],
                msg.get('num_inference_steps',10),msg.get('seed',0))
            response=dict(protocol=PROTOCOL,protocol_version=4,horizon=32,
                request_id=request,session_id=msg['session_id'],sample_ns=sample,
                model=MODEL_NAME,deployment_id=self.runtime.deployment_id,
                weights=self.runtime.describe()['weights'],contract='humanhand',task=task,
                instruction=TASKS[task]['instruction'],hand_order=['left','right'],
                frame='cam0_color_optical_frame',units='metres',keypoints=points[:32].copy(),
                keypoint_time_s=TIMES.copy(),proprio_keypoints=msg['observation']['keypoints'],
                native_keypoints=points,native_keypoint_time_s=np.arange(1,51,dtype=np.float32)/30,
                model_family='pi05',inference_seconds=diagnostic['inference_seconds'])
            np.savez_compressed(self.record_dir/(request+'.npz'),
                keypoints_left_right=points,time_s=np.arange(1,51)/30,
                **{k:v for k,v in diagnostic.items() if isinstance(v,np.ndarray)})
            print('INFER',task,'seconds',response['inference_seconds'],flush=True)
            return response


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(300)
        while True:
            try:msg=recv_message(self.request)
            except (ConnectionError,OSError):return
            try:response=self.server.adapter.dispatch(msg)
            except Exception as error:
                response={'error':type(error).__name__+': '+str(error)}
                print('ERROR',response['error'],flush=True)
            try:send_message(self.request,response)
            except (ConnectionError,OSError):return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18805)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    args=parser.parse_args()
    os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'cache'))
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    print('Loading Pi0.5',args.checkpoint,flush=True)
    runtime=Runtime(args.checkpoint)
    run=ROOT/'runs'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=True)
    with Server(('127.0.0.1',args.port),Handler) as server:
        server.adapter=Adapter(runtime,run)
        identity=dict(pid=os.getpid(),host='127.0.0.1',port=args.port,protocol=PROTOCOL,
            model=MODEL_NAME,**runtime.describe())
        (ROOT/'endpoint.json').write_text(json.dumps(identity,indent=2)+'\n')
        (run/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        print('READY',json.dumps(identity),flush=True)
        def stop(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        try:server.serve_forever(poll_interval=.1)
        except KeyboardInterrupt:print('Pi0.5 local server stopped',flush=True)
        finally:
            endpoint=ROOT/'endpoint.json'
            if endpoint.exists() and json.loads(endpoint.read_text()).get('pid')==os.getpid():endpoint.unlink()


if __name__=='__main__':main()
