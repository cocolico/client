"""Independently check saved timeline, joint limits, actions, and forward kinematics."""
from pathlib import Path
import argparse
import json

import numpy as np
from retarget import Hand, Rotation, ALIGNMENT_DEG


def verify(episode, kind):
    prefix = 'wuji2' if kind == 'wuji' else 'sharpa'
    key = 'wuji' if kind == 'wuji' else 'sharpa'
    output = episode/prefix
    source = np.load(episode/'keypoints21_camera.npz', allow_pickle=False)
    z = np.load(output/f'{prefix}_human_trajectory.npz', allow_pickle=False)
    replay = np.load(output/f'{prefix}_camera_replay_trajectory.npz', allow_pickle=False)
    n = len(source['valid_frame_indices'])
    assert n == len(z['time_s']) == len(replay['time_s'])
    np.testing.assert_array_equal(z['timestamp_ns'],source['target_timestamps_ns'])
    np.testing.assert_array_equal(z['source_timestamp_ns'],source['source_timestamps_ns'])
    np.testing.assert_array_equal(z['is_duplicate'],source['is_duplicate'])
    np.testing.assert_array_equal(z['source_frame_indices'],source['source_frame_indices'])
    np.testing.assert_array_equal(z['human_points_camera_m'],source['joints_camera_m'])
    np.testing.assert_array_equal(z['visible'],source['visible'])
    for k in ('timestamp_ns','source_timestamp_ns','is_duplicate'):
        np.testing.assert_array_equal(replay[k],z[k])
    assert np.all(np.diff(z['timestamp_ns']) > 0)
    assert abs(float(replay['fps'])-30) < 1e-5
    assert bool(np.asarray(z['visible']).all())
    action_size = 52 if kind == 'wuji' else 56
    joint_size = 40 if kind == 'wuji' else 44
    actions = np.load(output/f'{prefix}_actions_camera_right_left.npy')
    joints = np.load(output/f'{prefix}_hand_joints_left_right.npy')
    assert actions.shape == (n,action_size) and joints.shape == (n,joint_size)
    checks = dict(episode=episode.name,model=prefix,frames=n,duplicate_source_frames=int(source['is_duplicate'].sum()),
                  target_fps=30,all_solver_status_positive=True,joint_limit_violation_max_rad=0.,
                  independent_fk_max_abs_m=0.,timestamps_and_actions_exact=True)
    side_action = {}
    side_joints = {}
    for side in ('left','right'):
        hand = Hand(kind,side)
        q = z[f'{side}_{key}_joints_rad']
        pos = z[f'{side}_root_xyz_camera_m']
        rv = z[f'{side}_root_rotvec_camera_rad']
        assert list(z[f'{side}_{key}_joint_names']) == list(hand.names)
        assert (z[f'{side}_solver_status'] > 0).all()
        assert np.isfinite(q).all() and np.isfinite(pos).all() and np.isfinite(rv).all()
        violation = np.maximum(np.maximum(hand.lo-q,q-hand.hi),0).max()
        checks['joint_limit_violation_max_rad'] = max(checks['joint_limit_violation_max_rad'],float(violation))
        assert violation < 1e-9
        local,_ = hand.independent(q)
        for i in range(n):
            checks['independent_fk_max_abs_m'] = max(checks['independent_fk_max_abs_m'],
                float(np.abs(hand.fk(q[i])-local[i]).max()))
        fitted = np.einsum('nij,nkj->nki',Rotation.from_rotvec(rv).as_matrix(),local)+pos[:,None,:]
        np.testing.assert_allclose(fitted,z[f'{side}_fitted_points_camera_m'],atol=1e-9)
        target_key = 'normalized_target_points_camera_m' if kind == 'wuji' else 'target_points_camera_m'
        np.testing.assert_allclose(np.linalg.norm(fitted-z[f'{side}_{target_key}'],axis=-1),
                                   z[f'{side}_fitted_error_m'],atol=1e-9)
        if kind == 'wuji':
            corrected_r = Rotation.from_rotvec(rv).as_matrix() @ Rotation.from_euler('z',ALIGNMENT_DEG[side],degrees=True).as_matrix()
            corrected = np.einsum('nij,nkj->nki',corrected_r,local)+pos[:,None,:]
            np.testing.assert_allclose(corrected,z[f'{side}_corrected_points_camera_m'],atol=1e-9)
        action = np.c_[pos,rv,q]
        np.testing.assert_allclose(action,z[f'{side}_{key}_action_camera'],atol=1e-12)
        pose = replay[f'{side}_root_pose']
        np.testing.assert_allclose(pose[:,:3,:3],Rotation.from_rotvec(rv).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(pose[:,:3,3],pos,atol=1e-12)
        quat = replay[f'{side}_quaternion_wxyz'][:,[1,2,3,0]]
        np.testing.assert_allclose(Rotation.from_quat(quat).as_matrix(),pose[:,:3,:3],atol=1e-12)
        np.testing.assert_array_equal(replay[f'{side}_joint_positions_rad'],q)
        np.testing.assert_array_equal(replay[f'{side}_joint_names'],z[f'{side}_{key}_joint_names'])
        side_action[side] = action
        side_joints[side] = q
    np.testing.assert_array_equal(actions,np.c_[side_action['right'],side_action['left']])
    np.testing.assert_array_equal(joints,np.c_[side_joints['left'],side_joints['right']])
    if kind == 'wuji':
        optional = np.load(output/'wuji2_actions_camera_mount_reference_right_left.npy')
        np.testing.assert_array_equal(optional,np.c_[z['right_optional_mount_reference_action_camera'],
                                                   z['left_optional_mount_reference_action_camera']])
    assert checks['independent_fk_max_abs_m'] < 1e-8
    (output/'delivery_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('kind',choices=('sharpa','wuji'))
    args = parser.parse_args()
    verify(args.episode,args.kind)
