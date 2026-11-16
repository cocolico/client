"""Opt-in client image geometry, independent of robot control and model family.

仅按部署配置处理图像，不改变本体状态、时间戳、坐标系或执行逻辑。
"""
import copy
import numpy as np

CONTRACT=dict(schema='task_head_rgb_crop_v1',owner='client',
    tasks={'g2/diguan':dict(source_wh=[640,480],crop_xyxy=[218,130,538,450],
        output_wh=[320,320],resize='none',history='all_observed_frames')})

def configure(client,contract):
    """Pin the same visual contract on client and server / 客户端与服务端核对同一裁剪合同。"""
    if contract is not None and contract!=CONTRACT:
        raise ValueError('Unsupported client image preprocessing contract')
    if client.info.get('client_image_preprocess')!=contract:
        raise ValueError('Client/server image preprocessing contract mismatch')
    client.image_preprocess=copy.deepcopy(contract)

def prepare_request(request,contract):
    """Crop current and measured history together, once, after pair validation.

    配对检查后统一裁剪当前及所有历史帧；不修改采集缓存，不接受二次裁剪输入。
    """
    if contract is None or request.get('cmd')!='infer':return request
    if contract!=CONTRACT:raise ValueError('Unsupported image preprocessing contract')
    spec=contract['tasks'].get(request.get('task'))
    if spec is None:return request
    obs=request['observation'];rgb=np.asarray(obs['head_rgb'])
    if rgb.dtype!=np.uint8 or rgb.shape!=(480,640,3):
        raise ValueError('Diguan crop expects original uint8 RGB640x480; do not crop twice')
    images=obs.get('head_rgb_history')
    if images is not None:
        images=np.asarray(images)
        if images.dtype!=np.uint8 or images.shape!=(5,480,640,3) or not np.array_equal(images[-1],rgb):
            raise ValueError('Diguan requires paired original five-frame RGB history')
    new=dict(obs,head_rgb=np.ascontiguousarray(rgb[130:450,218:538]))
    if images is not None:new['head_rgb_history']=np.ascontiguousarray(images[:,130:450,218:538])
    return dict(request,observation=new)
