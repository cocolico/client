"""Wuji beta URDF FK and the 21-point, free-root weighted IK used by convert.py.

Uses the same links, ordering, neutral-bone normalization and tip weighting.
The implementation uses URDF transforms/Jacobians directly, without MuJoCo.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from common import SIDES, transform

PARENTS = np.array([0]+[v for f in range(5) for v in [0,1+4*f,2+4*f,3+4*f]])
TIPS = np.array([4,8,12,16,20])


def origin(joint):
    o = joint.find('origin'); out = np.eye(4)
    if o is not None:
        out[:3,3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
        out[:3,:3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return out


class HandModel:
    def __init__(self, root, side):
        p = side[0]; self.side = side
        self.path = Path(root)/'urdf'/f'{side}.urdf'
        tree = ET.parse(self.path).getroot(); joints = tree.findall('joint')
        self.names = [f'{p}_{f}_{s}' for f in ('thumb','index_finger','middle_finger','ring_finger','pinky')
            for s in (('cmc_flex','cmc_abd','mcp','ip') if f == 'thumb' else ('mcp_flex','mcp_abd','pip','dip'))]
        by_name = {j.get('name'):j for j in joints}
        self.lo = np.array([float(by_name[n].find('limit').get('lower')) for n in self.names])
        self.hi = np.array([float(by_name[n].find('limit').get('upper')) for n in self.names])
        self.neutral = np.clip(np.zeros(20), self.lo, self.hi)
        self.root = p+'_base_link'
        self.points = [p+'_wrist'] + [f'{p}_{f}_{s}'
            for f in ('thumb','index_finger','middle_finger','ring_finger','pinky')
            for s in ('proximal_abd','middle','distal','tip')]
        self.chain = []; known = {self.root}; pending = list(joints)
        self.ancestors = {self.root:[]}
        while pending:
            remaining = []
            for j in pending:
                parent = j.find('parent').get('link'); child = j.find('child').get('link')
                if parent not in known: remaining.append(j); continue
                name = j.get('name'); idx = None
                axis = None
                if j.get('type') != 'fixed':
                    if j.get('type') != 'revolute' or name not in self.names:
                        raise ValueError('Unsupported Wuji URDF joint: '+name)
                    idx = self.names.index(name)
                    axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                    axis /= np.linalg.norm(axis)
                self.chain.append((parent,child,origin(j),idx,axis))
                self.ancestors[child] = self.ancestors[parent] + ([] if idx is None else [idx])
                known.add(child)
            if len(remaining) == len(pending): raise ValueError('Disconnected Wuji URDF')
            pending = remaining

    def fk(self, q, jac=False):
        q = np.asarray(q, float)
        if q.shape != (20,) or not np.isfinite(q).all(): raise ValueError('Expected 20 finite hand angles')
        poses = {self.root:np.eye(4)}; axes = {}; pivots = {}
        for parent,child,offset,idx,axis in self.chain:
            pose = poses[parent] @ offset
            if idx is not None:
                axes[idx] = pose[:3,:3] @ axis; pivots[idx] = pose[:3,3].copy()
                motion = np.eye(4); motion[:3,:3] = Rotation.from_rotvec(axis*q[idx]).as_matrix()
                pose = pose @ motion
            poses[child] = pose
        points = np.array([poses[name][:3,3] for name in self.points])
        if not jac: return points
        result = np.zeros((21,3,20))
        for i, name in enumerate(self.points):
            for idx in self.ancestors[name]: result[i,:,idx] = np.cross(axes[idx], points[i]-pivots[idx])
        return points, result


def camera_points(hand, q, camera_T_root):
    pose = transform(camera_T_root)
    return hand.fk(q) @ pose[:3,:3].T + pose[:3,3]


def normalized_goals(points, neutral):
    points = np.asarray(points, float)
    lengths = np.linalg.norm(points-points[...,PARENTS,:], axis=-1)
    if not np.isfinite(points).all() or np.any(lengths[...,1:] < 1e-5):
        raise ValueError('Missing/degenerate predicted hand bones')
    model_lengths = np.linalg.norm(neutral-neutral[PARENTS], axis=-1)
    result = np.empty_like(points); result[...,0,:] = points[...,0,:]
    for i in range(1,21):
        parent = PARENTS[i]
        direction = (points[...,i,:]-points[...,parent,:])/lengths[...,i,None]
        result[...,i,:] = result[...,parent,:]+direction*model_lengths[i]
    return result


def retarget(hand, points, initial_q, initial_pose, limits):
    points = np.asarray(points, float)
    if points.ndim != 3 or points.shape[1:] != (21,3) or not len(points): raise ValueError('Expected [N,21,3]')
    initial_pose = transform(initial_pose)
    previous = np.r_[initial_q,Rotation.from_matrix(initial_pose[:3,:3]).as_rotvec(),initial_pose[:3,3]]
    goals = normalized_goals(points, hand.fk(hand.neutral))
    weights = np.full(21,.05); weights[0] = .1; weights[TIPS] = 1.
    weights = np.sqrt(weights)[:,None]
    lower = np.r_[hand.lo, np.full(6,-np.inf)]; upper = np.r_[hand.hi,np.full(6,np.inf)]
    solutions = []; errors = []; diagnostics = []
    rotation_limit = float(limits['max_root_rotation_rad'])
    maximum_rotation = 0.
    for goal in goals:
        previous_q = previous[:20].copy()
        def fun(z):
            rot = Rotation.from_rotvec(z[20:23]).as_matrix()
            fitted = hand.fk(z[:20]) @ rot.T + z[23:]
            return np.r_[((fitted-goal)*weights).ravel(), .005*(z[:20]-previous_q)]
        def jac(z):
            rot = Rotation.from_rotvec(z[20:23]).as_matrix()
            p, j = hand.fk(z[:20], True); out = np.zeros((83,26))
            out[:63,:20] = (np.einsum('ij,njk->nik',rot,j)*weights[:,:,None]).reshape(63,20)
            for k in range(3):
                delta = np.eye(3)[k]*1e-6
                dr = (Rotation.from_rotvec(z[20:23]+delta).as_matrix()-Rotation.from_rotvec(z[20:23]-delta).as_matrix())/2e-6
                out[:63,20+k] = (p@dr.T*weights).ravel()
            out[:63,23:] = np.tile(np.eye(3),(21,1))*np.repeat(weights,3,axis=0)
            out[63:,:20] = .005*np.eye(20)
            return out
        previous[:20] = np.clip(previous[:20],hand.lo+1e-9,hand.hi-1e-9)
        sol = least_squares(fun,previous,jac=jac,bounds=(lower,upper),max_nfev=300,
                            ftol=1e-9,xtol=1e-9,gtol=1e-8)
        if not sol.success or not np.isfinite(sol.x).all(): raise ValueError('Hand IK did not converge')
        previous = sol.x.copy(); pose = np.eye(4)
        pose[:3,:3] = Rotation.from_rotvec(previous[20:23]).as_matrix(); pose[:3,3] = previous[23:]
        fitted = camera_points(hand,previous[:20],pose)
        error = np.linalg.norm(fitted-goal,axis=-1)
        if error[TIPS].max() > limits['max_tip_fit_m'] or error.max() > limits['max_point_fit_m']:
            raise ValueError('Hand keypoint fit exceeds configured limits')
        translation_limit = limits.get('max_root_translation_m')
        if translation_limit is not None and np.linalg.norm(pose[:3,3]-initial_pose[:3,3]) > translation_limit:
            raise ValueError('Predicted root translation exceeds chunk limit')
        angle = Rotation.from_matrix(initial_pose[:3,:3].T @ pose[:3,:3]).magnitude()
        maximum_rotation = max(maximum_rotation,float(angle))
        solutions.append(previous); errors.append(error)
        diagnostics.append(dict(nfev=sol.nfev,status=sol.status,root_rotation_rad=float(angle),
            root_rotation_warning=bool(angle > rotation_limit),
            root_rotation_warning_threshold_rad=rotation_limit))
    if maximum_rotation > rotation_limit:
        print('WARNING',dict(type='predicted_root_rotation_exceeds_threshold',side=hand.side,
            maximum_rotation_rad=maximum_rotation,threshold_rad=rotation_limit,
            action='warn_only_continue'),flush=True)
    z = np.array(solutions); poses = np.tile(np.eye(4),(len(z),1,1))
    poses[:,:3,:3] = Rotation.from_rotvec(z[:,20:23]).as_matrix(); poses[:,:3,3] = z[:,23:]
    return dict(joints=z[:,:20], poses=poses, normalized_targets=goals,
                fit_error_m=np.array(errors), diagnostics=diagnostics)


def retarget_chunk(models, prediction, initial, limits, steps=32):
    if not 1 <= steps <= 32: raise ValueError('steps must be 1..32')
    out = {}
    for si, side in enumerate(SIDES):
        out[side] = retarget(models[side], prediction['keypoints'][:steps,si],
            initial[side]['q'], initial[side]['camera_T_root'], limits)
    # Include the measured start as t=0, rather than dropping the first forecast interval.
    out['time_s'] = np.r_[0., np.asarray(prediction['keypoint_time_s'][:steps],float)]
    for side in SIDES:
        out[side]['joints'] = np.vstack([initial[side]['q'],out[side]['joints']])
        out[side]['poses'] = np.concatenate([initial[side]['camera_T_root'][None],out[side]['poses']])
    return out
