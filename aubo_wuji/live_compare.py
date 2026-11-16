"""Read-only, localhost video comparison; no robot commands are issued here."""
import base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse,parse_qs
import cv2
import numpy as np
from common import ROOT, SIDES, read_json
from hand_kinematics import camera_points


def bend_degrees(points):
    """Unsigned geometric PIP bend (thumb: analogous distal bend), not motor angles."""
    p = np.asarray(points,float)
    a = p[...,2:21:4,:]-p[...,1:20:4,:]
    b = p[...,3:21:4,:]-p[...,2:21:4,:]
    norm = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/np.maximum(norm,1e-12),-1,1)))


def project(points,info):
    p=np.asarray(points,float)
    xy,_=cv2.projectPoints(p.reshape(-1,3),np.zeros(3),np.zeros(3),
        np.asarray(info['K'],float).reshape(3,3),np.asarray(info['D'],float))
    xy=xy.reshape(*p.shape[:-1],2)
    xy[p[...,2]<=.01]=np.nan
    return xy


def draw_hands(image,points,info,color):
    xy=project(points,info)
    for hand in xy:
        for f in range(5):
            chain=[0]+list(range(1+f*4,5+f*4))
            for a,b in zip(chain[:-1],chain[1:]):
                if np.isfinite(hand[[a,b]]).all() and np.max(abs(hand[[a,b]]))<10000:
                    cv2.line(image,tuple(np.round(hand[a]).astype(int)),tuple(np.round(hand[b]).astype(int)),color,2,cv2.LINE_AA)
    return image


class LiveCompare:
    def __init__(self,cfg,hardware,camera,models,session):
        self.hardware=hardware;self.camera=camera;self.models=models;self.session=Path(session)
        self.stop=threading.Event();self.latest_command=None;self.packet={'status':'等待第一个成功发送的控制目标'}
        self.cache_key=None;self.cache=None;self.writer=None;self.readers={}
        self.actual_trail=deque(maxlen=150)
        from trajectory_replay import Recordings
        records=Recordings(self.session)
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                url=urlparse(self.path)
                if url.path in ('/index','/clip'):
                    try:
                        query=parse_qs(url.query)
                        result=records.index() if url.path=='/index' else records.clip(query.get('cycle',[''])[0],int(query.get('rounds',['1'])[0]))
                        body=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
                    except Exception as error:
                        body=json.dumps(dict(error=str(error)),ensure_ascii=False).encode()
                    kind='application/json; charset=utf-8'
                elif url.path=='/state':
                    body=json.dumps(owner.packet,ensure_ascii=False,allow_nan=False).encode()
                    kind='application/json; charset=utf-8'
                elif self.path=='/' or self.path.startswith('/?'):
                    body=(ROOT/'trajectory_replay.html').read_bytes();kind='text/html; charset=utf-8'
                elif url.path=='/live':
                    body=(ROOT/'live_compare.html').read_bytes();kind='text/html; charset=utf-8'
                else:
                    self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind)
                self.send_header('Cache-Control','no-store');self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        port=int(cfg.get('monitor',{}).get('port',8765))
        try:self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        except OSError:self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        self.web=threading.Thread(target=self.server.serve_forever,daemon=True);self.web.start()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        print('LIVE COMPARISON',self.url,flush=True)
        (self.session/'monitor_url.txt').write_text(self.url+'\n')

    def command(self,target,index,plan):
        # Called after commands succeed: only copy 52 numbers and publish a tuple.
        # No rendering, files, HTTP, SDK reads or hashing in the control thread.
        self.latest_command=(time.monotonic(),np.array(target,copy=True),int(index),plan['trajectory']['path'])

    def points(self,arms,hands):
        from dh_kinematics import fk
        out=[]
        for s in SIDES:
            root=np.linalg.inv(self.hardware.base_T_camera[s]) @ fk(arms[s],self.kinematics[s]) @ np.asarray(self.hardware.replay['sides'][s]['flange_T_wuji_base'])
            out.append(camera_points(self.models[s],hands[s],root))
        return np.asarray(out)

    def load_cycle(self,path):
        if path==self.cache_key:return self.cache
        directory=Path(path).parent
        with np.load(directory/'prediction.npz') as z:pred=z['keypoints'].copy()
        with np.load(directory/'observation.npz') as z:
            image=z['head_rgb'].copy();stamp=int(z['sample_ns']);initial=z['keypoints'].copy()
        self.cache=dict(prediction=pred,image=image,stamp=stamp,initial=initial,cycle=directory.name)
        self.cache_key=path
        return self.cache

    def loop(self):
        from device_backends import Aubo
        from runtime_plan import split
        destination=self.session/'monitor';destination.mkdir(exist_ok=True)
        log=None
        try:
            self.kinematics={}
            for s in SIDES:
                self.readers[s]=Aubo(self.hardware.replay['sides'][s])
                self.kinematics[s]=self.readers[s].config.getKinematicsParam(True)
            log=(destination/'feedback.jsonl').open('a')
            while not self.stop.is_set():
                tick=time.monotonic()
                try:
                    cmd=self.latest_command
                    with self.camera.condition:
                        frame=self.camera.frames[-1] if self.camera.frames else None
                        info=dict(self.camera.info) if self.camera.info else None
                    if frame is None or cmd is None:
                        self.stop.wait(.2);continue
                    read_begin=time.monotonic()
                    arms={s:np.asarray(self.readers[s].state.getJointPositions(),float) for s in SIDES}
                    feedback=self.hardware.hands.snapshot()
                    hands={s:np.asarray(feedback[s]['positions_rad'],float) for s in SIDES}
                    measured=self.points(arms,hands)
                    target_arms,target_hands=split(cmd[1]);target=self.points(target_arms,target_hands)
                    cycle=self.load_cycle(cmd[3]);index=max(0,min(cmd[2]-1,31))
                    raw=cycle['initial'] if cmd[2]==0 else cycle['prediction'][index]
                    rgb=frame[2].copy()
                    if self.hardware.geometry is not None:
                        rgb=self.hardware.geometry.apply(rgb);info=read_json(ROOT/'vendor/cam0_color_camera_info.json')
                    actual_bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    model_bgr=cv2.cvtColor(cycle['image'],cv2.COLOR_RGB2BGR)
                    # Full forecast wrist paths, plus forecast skeleton at the sent step.
                    paths=project(cycle['prediction'][:,:,0],info)
                    for s in range(2):
                        pts=paths[:,s];pts=pts[np.isfinite(pts).all(axis=1)]
                        if len(pts)>1 and np.max(abs(pts))<10000:
                            cv2.polylines(model_bgr,[np.round(pts).astype(np.int32)],False,(255,210,50),2)
                    draw_hands(model_bgr,raw,info,(255,210,50))
                    draw_hands(model_bgr,target,info,(0,160,255))
                    draw_hands(actual_bgr,target,info,(0,160,255))
                    draw_hands(actual_bgr,measured,info,(70,230,70))
                    self.actual_trail.append(measured[:,0].copy())
                    trail=project(np.asarray(self.actual_trail),info)
                    for side in range(2):
                        xy=trail[:,side];xy=xy[np.isfinite(xy).all(axis=1)]
                        if len(xy)>1 and np.max(abs(xy))<10000:
                            cv2.polylines(actual_bgr,[np.round(xy).astype(np.int32)],False,(70,230,70),2)
                    now=time.monotonic()
                    age=(time.monotonic_ns()-frame[1])/1e9
                    packet=dict(status='实时反馈' if age<.5 else '相机画面过期',cycle=cycle['cycle'],step=cmd[2],
                        camera_age_s=age,command_age_s=now-cmd[0],observation_age_s=(time.monotonic_ns()-cycle['stamp'])/1e9,
                        read_span_s=now-read_begin,raw_bend= bend_degrees(raw).tolist(),
                        target_bend=bend_degrees(target).tolist(),actual_bend=bend_degrees(measured).tolist(),
                        points=dict(raw=raw.tolist(),target=target.tolist(),actual=measured.tolist()),
                        actual_joint_rad={s:dict(arm=arms[s].tolist(),hand=hands[s].tolist()) for s in SIDES},
                        target_joint_rad={s:dict(arm=target_arms[s].tolist(),hand=target_hands[s].tolist()) for s in SIDES},
                        joint_error_deg={s:np.degrees(hands[s]-target_hands[s]).tolist() for s in SIDES},
                        tip_error_mm=(np.linalg.norm(measured[:,[4,8,12,16,20]]-target[:,[4,8,12,16,20]],axis=-1)*1000).tolist(),
                        hand_feedback_age_s={s:now-feedback[s]['received_monotonic'] for s in SIDES})
                    log.write(json.dumps(dict(unix_s=time.time(),**packet),ensure_ascii=False)+'\n');log.flush()
                    for name,img in [('model',model_bgr),('actual',actual_bgr)]:
                        ok,jpg=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,80])
                        if not ok:raise RuntimeError('JPEG encode failed')
                        packet[name+'_image']='data:image/jpeg;base64,'+base64.b64encode(jpg).decode()
                    combined=np.hstack([model_bgr,actual_bgr])
                    cv2.putText(combined,f'{cycle["cycle"]} step {cmd[2]}  camera age {age:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                    if self.writer is None:
                        self.writer=cv2.VideoWriter(str(destination/'comparison.avi'),cv2.VideoWriter_fourcc(*'MJPG'),5.,(combined.shape[1],combined.shape[0]))
                        if not self.writer.isOpened():raise RuntimeError('Video recording unavailable')
                    self.writer.write(combined)
                    self.packet=packet
                except Exception as error:
                    self.packet=dict(status='监视数据暂不可用',error=str(error))
                self.stop.wait(max(0.,.2-(time.monotonic()-tick)))
        except Exception as error:
            self.packet=dict(status='监视器错误；控制流程独立运行',error=str(error))
        finally:
            if log:log.close()
            if self.writer:self.writer.release()
            for reader in self.readers.values():
                try:reader.close()
                except Exception:pass

    def close(self):
        self.stop.set();self.thread.join(timeout=5.)
        self.server.shutdown();self.server.server_close();self.web.join(timeout=2.)
