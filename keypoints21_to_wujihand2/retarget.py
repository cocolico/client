"""Sharpa measured joint trajectories -> Wuji Hand 2 early Beta, offline only.

The output floating root is expressed in each source Sharpa root frame, NOT in
the camera/world frame. Default outputs include the user-confirmed local-Z
correction (left +90 degrees, right -90 degrees) after geometric IK.
No arm IK or robot connection is performed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
TIPS=np.array([4,8,12,16,20]); PARENTS=np.array([0]+[v for f in range(5) for v in [0,1+4*f,2+4*f,3+4*f]])
SHARPA=['thumb_CMC_FE','thumb_CMC_AA','thumb_MCP_FE','thumb_MCP_AA','thumb_IP']+[f'{f}_{j}' for f in ['index','middle','ring'] for j in ['MCP_FE','MCP_AA','PIP','DIP']]+['pinky_CMC','pinky_MCP_FE','pinky_MCP_AA','pinky_PIP','pinky_DIP']
WUJI=['thumb_cmc_flex','thumb_cmc_abd','thumb_mcp','thumb_ip']+[f'{f}_{j}' for f in ['index_finger','middle_finger','ring_finger','pinky'] for j in ['mcp_flex','mcp_abd','pip','dip']]
ALIGNMENT_VERSION='wuji_beta_local_z_left_p90_right_m90_v1'
ALIGNMENT_DEG={'left':90.,'right':-90.}


def stat(a):
    a=np.asarray(a);return dict(mean=float(a.mean()),rmse=float(np.sqrt(np.mean(a*a))),p95=float(np.percentile(a,95)),max=float(a.max()))


def pose_attrs(origin):
    if origin is None:return {}
    r=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_quat()[[3,0,1,2]]
    return dict(pos=origin.get('xyz','0 0 0'),quat=' '.join(map(str,r)))


class Hand:
    def __init__(self,kind,side,visual=False):
        self.kind,self.side=kind,side
        self.path=HERE/'assets'/('sharpa' if kind=='sharpa' else 'wuji/urdf')/f'{side}.urdf'
        self.urdf=ET.parse(self.path).getroot()
        self.joints=list(self.urdf.findall('joint'))
        children={j.find('child').get('link') for j in self.joints}
        self.root=next(l.get('name') for l in self.urdf.findall('link') if l.get('name') not in children)
        if kind=='sharpa':
            self.names=[f'{side}_{n}' for n in SHARPA]
            self.points=[self.root]
            for f in ['thumb','index','middle','ring','pinky']:
                self.points += [f'{side}_{f}_{s}' for s in (['CMC_VL','MCP_VL','DP','fingertip'] if f=='thumb' else ['MCP_VL','MP','DP','fingertip'])]
        else:
            p=side[0];self.names=[f'{p}_{n}' for n in WUJI];self.points=[f'{p}_wrist']
            for f in ['thumb','index_finger','middle_finger','ring_finger','pinky']:
                self.points += [f'{p}_{f}_{s}' for s in ['proximal_abd','middle','distal','tip']]
        jdict={j.get('name'):j for j in self.joints}
        self.lo=np.array([float(jdict[n].find('limit').get('lower')) for n in self.names]);self.hi=np.array([float(jdict[n].find('limit').get('upper')) for n in self.names])
        self.vel=np.array([float(jdict[n].find('limit').get('velocity')) for n in self.names]);self.neutral=np.clip(np.zeros(len(self.names)),self.lo,self.hi)
        doc=ET.Element('mujoco',model=f'{kind}_{side}');ET.SubElement(doc,'compiler',angle='radian',autolimits='true',fusestatic='false')
        ET.SubElement(doc,'option',gravity='0 0 0');vis=ET.SubElement(doc,'visual');ET.SubElement(vis,'global',offwidth='1280',offheight='1000');ET.SubElement(vis,'headlight',ambient='.5 .5 .5',diffuse='.7 .7 .7',specular='.1 .1 .1')
        asset=ET.SubElement(doc,'asset');world=ET.SubElement(doc,'worldbody');bodies={self.root:ET.SubElement(world,'body',name=self.root)}
        pending=list(self.joints)
        while pending:
            before=len(pending)
            for j in pending[:]:
                parent=j.find('parent').get('link');child=j.find('child').get('link')
                if parent not in bodies:continue
                b=ET.SubElement(bodies[parent],'body',name=child,**pose_attrs(j.find('origin')));bodies[child]=b
                ET.SubElement(b,'inertial',mass='.01',pos='0 0 0',diaginertia='.00001 .00001 .00001')
                if j.get('type')!='fixed':
                    assert j.get('type')=='revolute';lim=j.find('limit')
                    ET.SubElement(b,'joint',name=j.get('name'),axis=j.find('axis').get('xyz'),range=lim.get('lower')+' '+lim.get('upper'))
                pending.remove(j)
            assert len(pending)<before
        for i,name in enumerate(self.points):ET.SubElement(bodies[name],'site',name=f'point{i}',size='.001',rgba='1 .5 0 1')
        if visual:
            if kind=='sharpa':
                # Use the previously validated renderer's exact URDF body transforms and visual fallbacks.
                old=ET.parse(HERE/'assets/sharpa/original_bimanual.xml').getroot()
                for mesh in old.findall('asset/mesh'):
                    m=copy.deepcopy(mesh);m.set('file',str(HERE/'assets/sharpa/meshes'/Path(m.get('file')).name));asset.append(m)
                for ob in old.findall('.//body'):
                    if ob.get('name') in bodies:
                        for geom in ob.findall('geom'):bodies[ob.get('name')].append(copy.deepcopy(geom))
            else:
                for link in self.urdf.findall('link'):
                    for v in link.findall('visual'):
                        mesh=v.find('geometry/mesh')
                        if mesh is None:continue
                        filename=(self.path.parent/mesh.get('filename')).resolve();assert filename.exists()
                        name='mesh'+str(len(asset));ET.SubElement(asset,'mesh',name=name,file=str(filename))
                        ET.SubElement(bodies[link.get('name')],'geom',type='mesh',mesh=name,contype='0',conaffinity='0',rgba='.65 .78 .86 1',**pose_attrs(v.find('origin')))
        self.xml=ET.tostring(doc).decode();self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.qids=np.array([self.model.joint(n).qposadr[0] for n in self.names]);self.dofs=np.array([self.model.joint(n).dofadr[0] for n in self.names]);self.siteids=np.array([self.model.site(f'point{i}').id for i in range(21)])

    def fk(self,q,jac=False):
        self.data.qpos[self.qids]=q;mujoco.mj_forward(self.model,self.data)
        points=self.data.site_xpos[self.siteids].copy()
        if not jac:return points
        jj=[]
        for sid in self.siteids:
            a=np.zeros((3,self.model.nv));mujoco.mj_jacSite(self.model,self.data,a,None,sid);jj.append(a[:,self.dofs])
        return points,np.array(jj)

    def independent(self,qs):
        n=len(qs);trans={self.root:np.tile(np.eye(4),(n,1,1))};qmap=dict(zip(self.names,qs.T));pending=list(self.joints)
        while pending:
            before=len(pending)
            for j in pending[:]:
                par=j.find('parent').get('link')
                if par not in trans:continue
                o=j.find('origin');a=np.eye(4)
                if o is not None:a[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');a[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
                b=np.tile(np.eye(4),(n,1,1))
                if j.get('type')!='fixed':b[:,:3,:3]=Rotation.from_rotvec(qmap[j.get('name')][:,None]*np.fromstring(j.find('axis').get('xyz'),sep=' ')).as_matrix()
                trans[j.find('child').get('link')]=trans[par]@a@b;pending.remove(j)
            assert len(pending)<before
        return np.stack([trans[x][:,:3,3] for x in self.points],axis=1),trans


def rigid_align(a,b):
    u,_,vh=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)));fix=np.eye(3);fix[2,2]=np.linalg.det(vh.T@u.T);r=vh.T@fix@u.T
    return r,b.mean(0)-r@a.mean(0)


def solve(source, target, frames):
    p0=target.fk(target.neutral);s0=source.fk(source.neutral)
    r,t=rigid_align(p0[[1,5,9,13,17]],s0[[1,5,9,13,17]])
    prev=np.r_[target.neutral,Rotation.from_matrix(r).as_rotvec(),t]
    w=np.full(21,.05);w[0]=.1;w[TIPS]=1.;w=np.sqrt(w)[:,None]
    bounds=(np.r_[target.lo,np.full(6,-np.inf)],np.r_[target.hi,np.full(6,np.inf)])
    result=[];error=[];evals=[];status=[];optimality=[]
    for i,goal in enumerate(frames):
        previous_q=prev[:20].copy()
        temporal=1e-4 if i==0 else .005
        def fun(z):
            rot=Rotation.from_rotvec(z[20:23]).as_matrix();p=target.fk(z[:20])@rot.T+z[23:26]
            return np.r_[((p-goal)*w).ravel(),temporal*(z[:20]-previous_q)]
        def jac(z):
            rot=Rotation.from_rotvec(z[20:23]).as_matrix();p,j=target.fk(z[:20],True);out=np.zeros((83,26))
            out[:63,:20]=(np.einsum('ij,njk->nik',rot,j)*w[:,:,None]).reshape(63,20)
            for k in range(3):
                d=np.eye(3)[k]*1e-6;dr=(Rotation.from_rotvec(z[20:23]+d).as_matrix()-Rotation.from_rotvec(z[20:23]-d).as_matrix())/(2e-6)
                out[:63,20+k]=(p@dr.T*w).ravel()
            out[:63,23:]=np.tile(np.eye(3),(21,1))*np.repeat(w,3,axis=0);out[63:,:20]=temporal*np.eye(20)
            return out
        prev[:20]=np.clip(prev[:20],target.lo+1e-9,target.hi-1e-9)
        sol=least_squares(fun,prev,jac=jac,bounds=bounds,max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-8)
        prev=sol.x.copy();pred=target.fk(prev[:20])@Rotation.from_rotvec(prev[20:23]).as_matrix().T+prev[23:]
        result.append(prev.copy());error.append(np.linalg.norm(pred-goal,axis=-1));evals.append(sol.nfev);status.append(sol.status);optimality.append(sol.optimality)
        if i%200==0:print(f'{source.side} IK {i}/{len(frames)} tips {np.mean(error[-1][TIPS])*1000:.2f}mm',flush=True)
    return np.array(result),np.array(error),dict(nfev=evals,status=status,optimality=optimality,initial_alignment_rotation=r.tolist(),initial_alignment_translation=t.tolist())
