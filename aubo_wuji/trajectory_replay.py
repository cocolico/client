"""Read-only historical trajectory viewer. Never imports robot SDKs or sends motion."""
import argparse
import base64
from collections import defaultdict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse,parse_qs
import cv2
import numpy as np
from common import ROOT,SIDES,load_config,read_json
from hand_kinematics import HandModel,camera_points


class Recordings:
    def __init__(self,session=None):
        self.fixed=Path(session).resolve() if session else None
        self.session=None;self.offset=0;self.rows=defaultdict(list);self.carry=b''
        self.lock=threading.Lock();self.models=None
        self.clip_cache={}

    def refresh(self):
        session=self.fixed
        if session is None:
            logs=list((ROOT/'runs').glob('*/monitor/feedback.jsonl'))
            if not logs:raise ValueError('尚无对比记录，请先启动带监视器的客户端')
            session=max(logs,key=lambda p:p.stat().st_mtime).parents[1]
        if session!=self.session:
            self.session=session;self.offset=0;self.rows=defaultdict(list);self.carry=b'';self.models=None;self.clip_cache={}
        path=session/'monitor/feedback.jsonl'
        if not path.exists():raise ValueError('此运行尚无feedback.jsonl')
        if path.stat().st_size<self.offset:
            self.offset=0;self.rows=defaultdict(list);self.carry=b''
        with path.open('rb') as f:
            f.seek(self.offset);chunk=f.read();self.offset=f.tell()
        lines=(self.carry+chunk).split(b'\n');self.carry=lines.pop()
        for line in lines:
            if not line:continue
            try:r=json.loads(line)
            except ValueError:continue
            if 'points' in r and 'cycle' in r:self.rows[r['cycle']].append(r)

    def index(self):
        with self.lock:
            self.refresh()
            status_path=self.session/'session.json'
            try:state=read_json(status_path)
            except (OSError,ValueError):state={}
            keys=sorted(self.rows)
            return dict(session=self.session.name,status=state.get('status'),cycles=[dict(
                name=k,samples=len(self.rows[k]),duration_s=self.rows[k][-1]['unix_s']-self.rows[k][0]['unix_s'],
                ended=i<len(keys)-1 or state.get('status') in ('completed','failed'),
                complete=max(r['step'] for r in self.rows[k])>=32 and
                    (i<len(keys)-1 or state.get('status') in ('completed','failed'))) for i,k in enumerate(keys)])

    def clip(self,name,count=1):
        if name=='all':
            with self.lock:
                self.refresh()
                if not self.rows:raise ValueError('暂无记录')
                name=max(self.rows);count=0
        if not name.startswith('cycle_') or not name[6:].isdigit():raise ValueError('Invalid cycle name')
        if not 0<=count<=20:raise ValueError('请选择全部记录或1到20轮')
        if count==0 or count>1:
            with self.lock:
                self.refresh();keys=[k for k in sorted(self.rows) if k<=name]
                if count:keys=keys[-count:]
            clips=[self.clip(k) for k in keys]
            if not clips:raise ValueError('暂无记录')
            merged=dict(clips[0]);offset=0
            names=('raw','normalized','target','actual','raw_xy','target_xy','actual_xy','actual_joint_rad','joint_error_deg','frame_ages_s')
            for key in names:merged[key]=[]
            merged['steps']=[];merged['time_s']=[];merged['command_time_s']=[];merged['segments']=[]
            for clip in clips:
                for key in names:merged[key].extend(clip[key])
                merged['steps'].extend(offset+s for s in clip['steps'])
                elapsed=clip['epoch_s']-clips[0]['epoch_s']
                merged['segments'].append(dict(cycle=clip['cycle'],start_s=elapsed,end_s=elapsed+clip['time_s'][-1]))
                merged['time_s'].extend(elapsed+t for t in clip['time_s'])
                merged['command_time_s'].extend(elapsed+t for t in clip['command_time_s'])
                offset+=len(clip['raw'])
            merged['cycle']=clips[0]['cycle']+' → '+clips[-1]['cycle']
            merged['forecast_frames']=sum(len(c['raw'])-1 for c in clips)
            return merged
        with self.lock:
            self.refresh();rows=list(self.rows.get(name,[]));session=self.session
            cached=self.clip_cache.get(name)
            if cached and cached[0]==len(rows):return cached[1]
        if not rows:raise ValueError('此轮暂无真实反馈采样')
        folder=session/name
        with np.load(folder/'prediction.npz',allow_pickle=False) as z:pred=z['keypoints'].copy()
        with np.load(folder/'observation.npz',allow_pickle=False) as z:
            image=z['head_rgb'].copy();initial=z['keypoints'].copy()
        with np.load(folder/'retarget.npz',allow_pickle=False) as z:ret={k:z[k].copy() for k in z.files}
        config=read_json(session/'configuration.json')
        model_root=Path(config['hand_model_root'])
        models={s:HandModel(model_root,s) for s in SIDES}
        n=len(ret['time_s']);target=[]
        for i in range(n):
            target.append([camera_points(models[s],ret[s+'_joints'][i],ret[s+'_poses'][i]).tolist() for s in SIDES])
        raw=np.concatenate([initial[None],pred[:n-1]],axis=0)
        from hand_kinematics import normalized_goals
        normalized=np.stack([np.concatenate([initial[None,si],
            ret[s+'_normalized_targets'][:n-1] if s+'_normalized_targets' in ret else
            normalized_goals(pred[:n-1,si],models[s].fk(models[s].neutral))],axis=0)
            for si,s in enumerate(SIDES)],axis=1)
        actual=np.array([r['points']['actual'] for r in rows])
        # Normalize plotting coordinates only in the browser, never in stored points.
        info=read_json(ROOT/'vendor/cam0_color_camera_info.json')
        cfgpath=session/'configuration.json'
        config=read_json(cfgpath)
        if config.get('camera',{}).get('geometry_mode')!='canonical_cam0':
            info=read_json(folder/'camera_info.json')
        def projection(points):
            p=np.asarray(points,float)
            xy,_=cv2.projectPoints(p.reshape(-1,3),np.zeros(3),np.zeros(3),
                np.asarray(info['K'],float).reshape(3,3),np.asarray(info['D'],float))
            xy=xy.reshape(*p.shape[:-1],2)
            invalid=(p[...,2]<=.01)|(~np.isfinite(xy).all(axis=-1))
            result=xy.astype(object);result[invalid]=None
            return result.tolist()
        ok,jpg=cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if not ok:raise ValueError('Cannot encode input image')
        base=rows[0]['unix_s']
        result=dict(session=session.name,cycle=name,epoch_s=base,forecast_frames=len(raw)-1,raw=raw.tolist(),normalized=normalized.tolist(),target=target,actual=actual.tolist(),
            raw_xy=projection(raw),target_xy=projection(target),actual_xy=projection(actual),
            time_s=[r['unix_s']-base for r in rows],steps=[r['step'] for r in rows],
            command_time_s=(ret['time_s']/config['execution']['speed']).tolist(),
            actual_joint_rad=[r.get('actual_joint_rad') for r in rows],
            joint_error_deg=[r['joint_error_deg'] for r in rows],
            frame_ages_s=[r.get('camera_age_s') for r in rows],
            image='data:image/jpeg;base64,'+base64.b64encode(jpg).decode())
        with self.lock:self.clip_cache[name]=(len(rows),result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path)
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args();records=Recordings(args.session)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            url=urlparse(self.path)
            try:
                if url.path=='/':
                    body=(ROOT/'trajectory_replay.html').read_bytes();kind='text/html; charset=utf-8'
                elif url.path=='/index':
                    body=json.dumps(records.index(),ensure_ascii=False,allow_nan=False).encode();kind='application/json'
                elif url.path=='/clip':
                    query=parse_qs(url.query);name=query.get('cycle',[''])[0]
                    body=json.dumps(records.clip(name,int(query.get('rounds',['1'])[0])),ensure_ascii=False,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200)
            except Exception as error:
                body=json.dumps(dict(error=str(error)),ensure_ascii=False).encode();kind='application/json';self.send_response(400)
            self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'TRAJECTORY REPLAY http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
