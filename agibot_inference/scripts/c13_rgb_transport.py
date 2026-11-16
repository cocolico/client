"""C13-only passive RGB transport. No state reads, inference, FK or motion.

Keep raw observations through RPC/execution pauses. The server owns temporal
selection and VAE preprocessing. PNG is lossless; no client resize or jitter.
"""
from collections import deque
import base64
import threading
import time
import numpy as np


class RGBTransport:
    def __init__(self, read_frame, *, period=.1, max_age_s=.3):
        if not .005<=period<=.2 or not 0<max_age_s<=1:
            raise ValueError('Invalid RGB acquisition timing')
        self.read_frame=read_frame;self.period=period;self.max_age_s=max_age_s
        self.io_lock=threading.RLock();self.lock=threading.Lock()
        self.stop=threading.Event();self.ready=threading.Event()
        self.frames=deque(maxlen=12);self.source_id=None;self.last_error=None
        self.thread=None;self.first_sample_ns=None

    def read_current(self, function):
        # Only C13 uses this lock. No concurrent SDK camera calls.
        with self.io_lock:return function()

    def _run(self):
        import cv2
        while not self.stop.is_set():
            start=time.monotonic()
            try:
                with self.io_lock:packet=self.read_frame()
                if packet is not None and packet['source_id']!=self.source_id:
                    ns=packet['sample_ns'];rgb=np.asarray(packet['head_rgb'])
                    if type(ns) is not int or not 0<=ns<=time.monotonic_ns():raise ValueError('Invalid capture time')
                    if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
                            or min(rgb.shape[:2])<32 or max(rgb.shape[:2])>640 or rgb.shape[0]*rgb.shape[1]>640*480):
                        raise ValueError('C13 passive RGB transport supports up to640x480')
                    ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1])
                    if not ok:raise RuntimeError('PNG encoding failed')
                    frame=dict(sample_ns=ns,head_rgb=base64.b64encode(png.tobytes()).decode('ascii'))
                    with self.lock:
                        if self.frames and ns<=self.frames[-1]['sample_ns']:raise ValueError('Capture clock did not advance')
                        self.frames.append(frame)
                        if self.first_sample_ns is None:self.first_sample_ns=ns
                        self.source_id=packet['source_id'];self.last_error=None
                    self.ready.set()
            except Exception as error:
                with self.lock:self.last_error=str(error)
            self.stop.wait(max(.001,self.period-(time.monotonic()-start)))

    def frames_before(self, sample_ns):
        with self.lock:
            frames=[dict(f) for f in self.frames if sample_ns-1100000000<=f['sample_ns']<sample_ns]
            error=self.last_error
        if not frames or (sample_ns-frames[-1]['sample_ns'])/1e9>self.max_age_s:
            raise RuntimeError('C13 head camera history is stale/unavailable: '+str(error))
        # Keep the wire body below16MiB, including the current raw RGB/state.
        size=0;result=[]
        for frame in reversed(frames):
            size+=len(frame['head_rgb'])
            if size>13*1024*1024:break
            result.append(frame)
        if not result:raise RuntimeError('C13 historical frame exceeds wire budget')
        return list(reversed(result))

    def clear(self):
        with self.lock:self.frames.clear();self.source_id=None;self.first_sample_ns=None
        self.ready.clear()

    def __enter__(self):
        if self.thread is not None:raise RuntimeError('RGB transport already started')
        self.thread=threading.Thread(target=self._run,name='c13-passive-head-rgb',daemon=True);self.thread.start()
        if not self.ready.wait(2):
            self.close();raise RuntimeError('C13 head camera did not produce a frame: '+str(self.last_error))
        return self

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():raise RuntimeError('C13 head reader failed to stop')

    def __exit__(self,*args):self.close()


def g2_head_reader(robot):
    """Existing camera subscriber only; never poll hands, joints or controller."""
    def read():
        import cv2
        gdk=robot.gdk;image=robot.camera.get_latest_image(gdk.CameraType.kHeadColor,100.)
        if image is None:raise RuntimeError('No G2 head RGB')
        stamp=time.monotonic_ns();raw=np.asarray(image.data,np.uint8)
        if image.encoding in (gdk.Encoding.JPEG,gdk.Encoding.PNG):
            bgr=cv2.imdecode(raw.reshape(-1),cv2.IMREAD_COLOR)
            if bgr is None:raise ValueError('Invalid G2 encoded RGB')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        elif image.encoding==gdk.Encoding.UNCOMPRESSED:
            rgb=raw.reshape(image.height,image.width,3)
            if image.color_format==gdk.ColorFormat.BGR:rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
            elif image.color_format!=gdk.ColorFormat.RGB:raise ValueError('Unsupported G2 color format')
        else:raise ValueError('Unsupported G2 image encoding')
        # Match the pre-existing current-observation decoder exactly.
        rgb=cv2.resize(rgb,(640,480),interpolation=cv2.INTER_LINEAR)
        return dict(sample_ns=stamp,source_id=int(image.timestamp_ns),head_rgb=rgb)
    return read


def piper_head_reader(hub):
    """Read the existing shared JPEG slot; do not poll CAN or mutate SourceHub."""
    def read():
        import cv2
        frame=hub.buffers['active_camera'].read(0)
        if frame is None:return None
        bgr=cv2.imdecode(np.frombuffer(frame['jpeg'],np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Invalid Piper encoded RGB')
        return dict(sample_ns=int(frame['read_monotonic_ns']),source_id=int(frame['sequence']),head_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    return read
