"""Physical 78D task contract. Pure indexing only: no FK, IK or robot I/O."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL_VERSION=6
SCHEMA='egoav_78d_20261021_v1'
REPRESENTATION='egoav78_chunk_first_camera_rot6d'
MODELS=('pi05','fastwam','alpha','coarse_fine','joint','tri')
WIRE_INFO=dict(protocol_version=PROTOCOL_VERSION,schema=SCHEMA,representation=REPRESENTATION,
    state_dim=78,action_dim=78,state_units='physical',action_units='physical',
    action_mode='chunk_relative',coordinate_frame='chunk_first_camera_fixed',
    rotation_order='first_two_columns',position_units='meter',joint_units='radian',
    gripper_units='millimeter',dt_s=.05,camera_keys=['head_rgb'])


def embodiment_mask(kind):
    mask=np.zeros(78,np.uint8)
    if kind not in ('g2','3b'):
        raise ValueError('Supported embodiments: g2, 3b')
    mask[:27]=1
    if kind=='g2':mask[32:54]=1;mask[55:77]=1
    else:mask[[54,77]]=1
    return mask


def task_mask(kind, task=None):
    mask = embodiment_mask(kind)
    if task == 'drink':
        if kind != 'g2':
            raise ValueError('drink waist is G2-only')
        mask[27] = 1  # idx05_body_joint5, not body_joint1
    return mask


def validate_tasks(tasks):
    if not isinstance(tasks,dict) or not tasks:
        raise ValueError('Nonempty task registry required')
    for name,spec in tasks.items():
        if not isinstance(name,str) or not name or not isinstance(spec,dict):
            raise ValueError('Invalid task entry')
        if set(spec)!={'instruction','embodiment','validity'}:
            raise ValueError('Task requires instruction, embodiment, validity')
        if not isinstance(spec['instruction'],str) or not spec['instruction'].strip():
            raise ValueError('Task instruction must match training instruction')
        expected=task_mask(spec['embodiment'],name).tolist()
        mask=spec['validity']
        if (not isinstance(mask,list) or len(mask)!=78
                or any(type(x) is not int or x not in (0,1) for x in mask)
                or mask!=expected):
            raise ValueError(f'Task validity disagrees with embodiment: {name}')
    return copy.deepcopy(tasks)


def load_tasks(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema')!=SCHEMA:
        raise ValueError('Task registry schema mismatch')
    return validate_tasks(value['tasks'])


def task_digest(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,ensure_ascii=False,
        separators=(',',':')).encode()).hexdigest()


def _array(values,width):
    x=np.asarray(values,np.float32)
    if x.ndim not in (1,2) or x.shape[-1]!=width or (x.ndim==2 and not len(x)) or not np.isfinite(x).all():
        raise ValueError(f'Expected finite [{width}] or [H,{width}]')
    return x


def validate78(values,kind,*,task=None):
    x=_array(values,78)
    if np.any(x[...,task_mask(kind,task)==0]!=0):
        raise ValueError('Inactive G2/3B physical slots must be zero')
    return x


def g2_to78(values):
    x=_array(values,71);y=np.zeros((*x.shape[:-1],78),np.float32)
    y[...,:9]=x[...,:9];y[...,9:18]=x[...,31:40];y[...,18:27]=x[...,62:71]
    y[...,32:54]=x[...,9:31];y[...,55:77]=x[...,40:62]
    return y


def g2_from78(values):
    x=validate78(values,'g2')
    return np.concatenate((x[...,:9],x[...,32:54],x[...,9:18],x[...,55:77],x[...,18:27]),axis=-1)


def piper_to78(values):
    x=_array(values,76);y=np.zeros((*x.shape[:-1],78),np.float32)
    active=np.zeros(76,bool);active[:9]=True;active[12:30]=True;active[[52,75]]=True
    if np.any(x[...,~active]!=0):raise ValueError('Inactive Piper internal76 slots must be zero')
    y[...,:9]=x[...,12:21];y[...,9:18]=x[...,21:30];y[...,18:27]=x[...,:9]
    y[...,54]=x[...,52];y[...,77]=x[...,75]
    return y


def piper_from78(values):
    x=validate78(values,'3b');y=np.zeros((*x.shape[:-1],76),np.float32)
    y[...,12:21]=x[...,:9];y[...,21:30]=x[...,9:18];y[...,:9]=x[...,18:27]
    y[...,52]=x[...,54];y[...,75]=x[...,77]
    return y
