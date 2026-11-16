"""Camera-optical human 21 landmarks -> Wuji Hand 2 Beta joint+root poses."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from retarget import Hand,Rotation,solve,stat,TIPS,ALIGNMENT_DEG,ALIGNMENT_VERSION,PARENTS

HERE=Path(__file__).resolve().parent
SOURCE=HERE
SIDES=['left','right']

class GoalSource:
    def __init__(self,side,first):self.side=side;self.neutral=np.zeros(0);self.first=first
    def fk(self,_):return self.first


def normalized_goals(human,neutral):
    """Retain camera wrist, bone directions and timing; replace segment lengths."""
    n=len(human);out=np.empty_like(human,dtype=float);out[:,0]=human[:,0]
    human_lengths=np.linalg.norm(human-human[:,PARENTS],axis=-1)
    model_lengths=np.linalg.norm(neutral-neutral[PARENTS],axis=-1)
    assert np.all(human_lengths[:,1:]>1e-5)
    for i in range(1,21):
        parent=PARENTS[i]
        direction=(human[:,i]-human[:,parent])/human_lengths[:,i,None]
        out[:,i]=out[:,parent]+direction*model_lengths[i]
    return out


def run(source=SOURCE,destination=HERE):
    destination.mkdir(parents=True,exist_ok=True)
    handpose=source/'keypoints21_camera.npz';data=np.load(handpose)
    human=data['joints_camera_m'].astype(float);n=len(human)
    assert human.shape==(n,2,21,3) and n>1 and list(data['sides'])==SIDES
    assert np.isfinite(human).all() and np.asarray(data['visible']).all()
    stamp=np.asarray(data['target_timestamps_ns'],dtype=np.int64)
    source_stamp=np.asarray(data['source_timestamps_ns'],dtype=np.int64)
    frame=np.asarray(data['frame_indices'],dtype=np.int64)
    source_frame=np.asarray(data['source_frame_indices'],dtype=np.int64)
    duplicate=np.asarray(data['is_duplicate'],dtype=bool)
    assert np.array_equal(frame,data['valid_frame_indices'])
    assert np.all(np.diff(stamp)>0) and np.all(np.diff(source_stamp)>=0)
    assert np.array_equal(duplicate,np.r_[False,np.diff(source_frame)==0])
    fps=float(1e9/np.median(np.diff(stamp)))
    time_s=(stamp-stamp[0])/1e9
    output={'time_s':time_s,'timestamp_ns':stamp,'source_timestamp_ns':source_stamp,'is_duplicate':duplicate,'frame_indices':frame,'source_frame_indices':source_frame,'visible':data['visible'],
        'camera_intrinsics_fx_fy_cx_cy':data['intrinsics'],'side_order':data['sides'],
        'frame':np.array('undistorted color optical; +X image right, +Y image down, +Z forward'),
        'main_pose_alignment':np.array('direct human camera fit; no Sharpa mounting rotation'),
        'optional_mount_reference_version':np.array(ALIGNMENT_VERSION),
        'optional_mount_local_z_deg_left_right':np.array([ALIGNMENT_DEG[s] for s in SIDES]),
        'human_points_camera_m':human,'source_timestamp_origin':np.array('target_timestamps_ns: uniform 30 Hz replay timeline')}
    report={'source':str(source),'source_handpose_sha256':hashlib.sha256(handpose.read_bytes()).hexdigest(),
      'target_model':'Wuji Hand 2 early Beta v2026.6.27','frames':n,'duration_s':float(time_s[-1]),'nominal_fps':fps,
      'time_basis':'target_timestamps_ns, uniform 30 Hz; original source timestamps and duplicate flags preserved',
      'duplicate_frames':int(duplicate.sum()),
      'source_frame':'undistorted color optical; +X image right, +Y image down, +Z forward; metres',
      'method':'Human bone directions with Wuji neutral segment lengths -> bounded 21-point geometric IK with free root and temporal joint prior',
      'human_length_scaling':'Per bone, Wuji neutral segment length; wrist position unchanged in normalized goals',
      'main_output_orientation':'Direct geometric fit in this camera optical frame. No Sharpa-to-Wuji mounting correction applied.',
      'optional_mount_reference':{'left_local_z_deg':90,'right_local_z_deg':-90,'composition':'R_reference=R_fit @ Rz(angle); t_reference=t_fit',
        'scope':'Previously visually confirmed on Sharpa workcell; not calibrated to this different camera. Stored separately, not the main output.'},
      'root_pose_scope':'Camera optical frame, not FR3 base or flange; no hand-eye transform for this camera',
      'solver_source':'Human points only; no Sharpa joint data or Sharpa source model used as IK target',
      'sides':{}}
    for si,side in enumerate(SIDES):
        hand=Hand('wuji',side);target=normalized_goals(human[:,si],hand.fk(hand.neutral))
        solution,error,diag=solve(GoalSource(side,target[0]),hand,target)
        q=solution[:,:20];Rfit=Rotation.from_rotvec(solution[:,20:23]).as_matrix();position=solution[:,23:]
        correction=Rotation.from_euler('z',ALIGNMENT_DEG[side],degrees=True).as_matrix();Rcorrected=Rfit@correction
        joints_local,links=hand.independent(q)
        fitted=np.einsum('nij,nkj->nki',Rfit,joints_local)+position[:,None,:]
        corrected=np.einsum('nij,nkj->nki',Rcorrected,joints_local)+position[:,None,:]
        assert np.max(abs(fitted-target))-np.max(error)<1e-8
        assert np.all(np.isfinite(solution)) and np.all(q>=hand.lo-1e-9) and np.all(q<=hand.hi+1e-9)
        q_step=np.rad2deg(abs(np.diff(q,axis=0)));root_step=np.linalg.norm(np.diff(position,axis=0),axis=-1)
        fitted_errors=np.linalg.norm(fitted-target,axis=-1)
        raw_errors=np.linalg.norm(fitted-human[:,si],axis=-1)
        corrected_errors=np.linalg.norm(corrected-human[:,si],axis=-1)
        output.update({f'{side}_wuji_joint_names':np.array(hand.names),f'{side}_wuji_joints_rad':q,
            f'{side}_root_xyz_camera_m':position,
            f'{side}_root_rotvec_camera_rad':solution[:,20:23],
            f'{side}_optional_mount_reference_rotvec_camera_rad':Rotation.from_matrix(Rcorrected).as_rotvec(),
            f'{side}_wuji_action_camera':np.c_[position,solution[:,20:23],q],
            f'{side}_optional_mount_reference_action_camera':np.c_[position,Rotation.from_matrix(Rcorrected).as_rotvec(),q],
            f'{side}_normalized_target_points_camera_m':target,
            f'{side}_fitted_points_camera_m':fitted,
            f'{side}_corrected_points_camera_m':corrected,
            f'{side}_fitted_error_m':fitted_errors,
            f'{side}_raw_human_error_m':raw_errors,
            f'{side}_corrected_to_human_error_m':corrected_errors,
            f'{side}_solver_status':np.array(diag['status']),f'{side}_solver_nfev':np.array(diag['nfev'])})
        report['sides'][side]={'normalized_tip_fit_mm':stat(fitted_errors[:,TIPS]*1000),
           'raw_human_tip_distance_before_orientation_correction_mm':stat(raw_errors[:,TIPS]*1000),
           'raw_human_tip_distance_after_orientation_correction_mm':stat(corrected_errors[:,TIPS]*1000),
           'wrist_fit_mm':stat(fitted_errors[:,0]*1000),
           'joint_step_deg':stat(q_step),'root_step_mm':stat(root_step*1000),
           'solver_budget_exhausted_frames':int(np.sum(np.array(diag['status'])==0)),
           'max_nfev':int(np.max(diag['nfev'])),'joint_limit_violation_max_rad':float(np.maximum(np.maximum(hand.lo-q,q-hand.hi),0).max()),
           'human_segment_lengths_mean_m':np.linalg.norm(human[:,si]-human[:,si,PARENTS],axis=-1).mean(0).tolist(),
           'wuji_neutral_segment_lengths_m':np.linalg.norm(hand.fk(hand.neutral)-hand.fk(hand.neutral)[PARENTS],axis=-1).tolist()}
        print(side,report['sides'][side]['normalized_tip_fit_mm'],flush=True)
    np.savez_compressed(destination/'wuji2_human_trajectory.npz',**output)
    np.save(destination/'wuji2_hand_joints_left_right.npy',np.c_[output['left_wuji_joints_rad'],output['right_wuji_joints_rad']])
    np.save(destination/'wuji2_actions_camera_right_left.npy',np.c_[output['right_wuji_action_camera'],output['left_wuji_action_camera']])
    np.save(destination/'wuji2_actions_camera_mount_reference_right_left.npy',np.c_[output['right_optional_mount_reference_action_camera'],output['left_optional_mount_reference_action_camera']])
    replay={'time_s':time_s,'timestamp_ns':stamp,'source_timestamp_ns':source_stamp,'is_duplicate':duplicate,'fps':np.array(fps),
        'side_order':np.array(SIDES),'frame':np.array('undistorted_color_optical'),
        'pose_target':np.array('Wuji l_base_link / r_base_link; not FR3 flange'),
        'quaternion_order':np.array('wxyz'),'position_unit':np.array('metre'),
        'joint_unit':np.array('radian'),'alignment_version':np.array('human_camera_direct_fit_v1'),
        'source_stream':np.array('keypoints21_camera.npz:joints_camera_m'),
        'model':np.array('Wuji hand2_beta early v2026.6.27'),
        'target_arm_ik_available':np.array(False)}
    for side in SIDES:
        xyz=output[f'{side}_root_xyz_camera_m']
        rotation=Rotation.from_rotvec(output[f'{side}_root_rotvec_camera_rad'])
        matrices=rotation.as_matrix()
        quats_xyzw=rotation.as_quat()
        quats_wxyz=quats_xyzw[:,[3,0,1,2]]
        for i in range(1,n):
            if np.dot(quats_wxyz[i],quats_wxyz[i-1])<0:quats_wxyz[i]*=-1
        pose=np.tile(np.eye(4),(n,1,1));pose[:,:3,:3]=matrices;pose[:,:3,3]=xyz
        replay.update({f'{side}_root_pose':pose,f'{side}_position_m':xyz,
            f'{side}_quaternion_wxyz':quats_wxyz,
            f'{side}_joint_names':output[f'{side}_wuji_joint_names'],
            f'{side}_joint_positions_rad':output[f'{side}_wuji_joints_rad']})
    np.savez_compressed(destination/'wuji2_camera_replay_trajectory.npz',**replay)
    (destination/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('completed',n,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--output',type=Path,default=HERE);a=ap.parse_args();run(a.source,a.output)
