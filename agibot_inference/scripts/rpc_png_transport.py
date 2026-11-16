"""Opt-in lossless RGB transport. Restore byte-identical legacy JSON before inference."""
import base64
import hashlib
import io
import json
import math
import numpy as np
from _ours_protocol import MAX_BYTES, _encode

CODEC='rgb_png_lossless_v1'

def dumps(value, **kwargs):
    return json.dumps(value, allow_nan=False, separators=(',',':'), **kwargs).encode()

def bounded(payload):
    if not 0<len(payload)<=MAX_BYTES:raise ValueError('Message too large')
    return payload

def encode_request(request):
    original=bounded(dumps(request,default=_encode))
    if request.get('cmd')!='infer':return original
    import cv2
    obs=request.get('observation',{})
    wire=dict(request,observation=dict(obs))
    if ('head_rgb_history' in obs and len(obs['head_rgb_history']) and
            np.array_equal(obs.get('head_rgb'),obs['head_rgb_history'][-1])):
        wire['observation']['head_rgb']={'_current_from_history':True}
    def encode(value):
        if (isinstance(value,np.ndarray) and value.dtype==np.uint8 and
                value.ndim in (3,4) and value.shape[-1]==3):
            frames=value if value.ndim==4 else [value]
            compressed=[]
            for frame in frames:
                ok,png=cv2.imencode('.png',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,1])
                if not ok:raise ValueError('PNG encode failed')
                compressed.append(base64.b64encode(png).decode('ascii'))
            return dict(_rgb_png=True,shape=list(value.shape),frames=compressed,
                        sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return _encode(value)
    return bounded(dumps(dict(_transport_codec=CODEC,original_sha256=hashlib.sha256(original).hexdigest(),
                              request=wire),default=encode))

def restore_request(payload):
    bounded(payload)
    envelope=json.loads(payload)
    if not isinstance(envelope,dict):raise ValueError('Expected request object')
    if '_transport_codec' not in envelope:return payload
    if envelope['_transport_codec']!=CODEC:raise ValueError('Unknown transport codec')
    from PIL import Image
    def restore(value):
        if isinstance(value,list):return [restore(x) for x in value]
        if not isinstance(value,dict):return value
        if value.get('_rgb_png') is True:
            shape=value.get('shape')
            if (not isinstance(shape,list) or len(shape) not in (3,4) or shape[-1]!=3 or
                    any(type(x) is not int or x<=0 for x in shape) or math.prod(shape)>MAX_BYTES):
                raise ValueError('Invalid PNG shape or size')
            count=shape[0] if len(shape)==4 else 1
            if len(value['frames'])!=count:raise ValueError('PNG frame count mismatch')
            frames=[]
            for encoded in value['frames']:
                with Image.open(io.BytesIO(base64.b64decode(encoded,validate=True))) as picture:
                    if picture.format!='PNG' or picture.mode!='RGB' or picture.size!=(shape[-2],shape[-3]):
                        raise ValueError('PNG shape/mode mismatch')
                    frames.append(np.array(picture,dtype=np.uint8))
            pixels=np.stack(frames) if len(shape)==4 else frames[0]
            if hashlib.sha256(pixels.tobytes()).hexdigest()!=value['sha256']:
                raise ValueError('Lossless pixel verification failed')
            return _encode(pixels)
        return {key:restore(item) for key,item in value.items()}
    request=restore(envelope['request'])
    obs=request.get('observation',{})
    if isinstance(obs.get('head_rgb'),dict) and obs['head_rgb'].get('_current_from_history') is True:
        history=obs['head_rgb_history'];shape=history['shape'][1:]
        if history.get('__array__') is not True or history['dtype']!='|u1' or len(shape)!=3:
            raise ValueError('Invalid RGB history dedup reference')
        raw=base64.b64decode(history['data'],validate=True);size=math.prod(shape)
        obs['head_rgb']=dict(__array__=True,dtype='|u1',shape=shape,data=base64.b64encode(raw[-size:]).decode())
    original=bounded(dumps(request))
    if hashlib.sha256(original).hexdigest()!=envelope['original_sha256']:
        raise ValueError('Original request bytes mismatch')
    return original

def send_message(sock,request):
    payload=encode_request(request)
    sock.sendall(len(payload).to_bytes(4,'big')+payload)


def transport_config(config):
    """One model-independent default shared by endpoint selection and wire encoding.

    Reviewed current v4 deployments on zw use the installed CPU gateway.
    Explicit overrides/disable and legacy or other-host configurations keep
    their existing behavior. Never select transport by checkpoint/model name.
    """
    if 'lossless_transport' in config:
        value = config['lossless_transport']
        if value is None or value is False:
            return None
        if not isinstance(value, dict):
            raise ValueError('lossless_transport must be an object, null or false')
        return dict(value)
    if config.get('ssh_host') == 'zw' and config.get('wire_protocol') in (
            'egoav78_v4', 'egoav78_c13_v4', 'egoav78_abcd_history_v4',
            'egoav78_paired_history_v4', 'egoav78_aligned_c0_v4', 'egoav78_native_post_v4'):
        return dict(codec='rgb_png_lossless_v1', congestion='bbr',
            gateway_script='/mnt/workspace/users/haoran/workspace/deploy/rpc_png_transport_20261109/rpc_png_gateway.py')
    return None
