"""Render each 30 FPS human-keypoint sample against its Sharpa or Wuji2 fit."""
from pathlib import Path
import argparse
import copy
import os
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import imageio_ffmpeg
import mujoco
import numpy as np

from retarget import Hand, Rotation, PARENTS

COLORS = [(249,151,93),(248,218,85),(94,223,174),(87,172,255),(210,145,250)]


def label(im, message, xy, scale=.58):
    cv2.putText(im, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (240, 240, 245), 1, cv2.LINE_AA)


def draw_points(im, points, intrinsics, side):
    fx, fy, cx, cy = intrinsics
    uv = np.round(np.c_[points[:,0]/points[:,2]*fx+cx,
                         points[:,1]/points[:,2]*fy+cy]).astype(int)
    for finger in range(5):
        for j in range(1+4*finger,5+4*finger):
            cv2.line(im, tuple(uv[PARENTS[j]]), tuple(uv[j]), COLORS[finger], 2, cv2.LINE_AA)
        cv2.circle(im, tuple(uv[4+4*finger]), 3, COLORS[finger], -1, cv2.LINE_AA)
    label(im, side.upper(), tuple(uv[0]+[7,-7]), .5)


def viewer_model(kind, width, height):
    doc = ET.Element('mujoco')
    ET.SubElement(doc, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(ET.SubElement(doc, 'visual'), 'global', offwidth=str(width), offheight=str(height))
    asset = ET.SubElement(doc, 'asset')
    world = ET.SubElement(doc, 'worldbody')
    for side in ('left','right'):
        tree = ET.fromstring(Hand(kind, side, True).xml)
        for mesh in tree.findall('asset/mesh'):
            mesh.set('name',side+'_'+mesh.get('name'))
            asset.append(mesh)
        body = copy.deepcopy(tree.find('worldbody/body'))
        body.insert(0, ET.Element('freejoint', name=side+'_floating'))
        ET.SubElement(body, 'inertial', mass='.1', pos='0 0 0', diaginertia='.001 .001 .001')
        for geom in body.findall('.//geom'):
            if geom.get('mesh'):
                geom.set('mesh',side+'_'+geom.get('mesh'))
        for site in body.findall('.//site'):
            site.set('name',side+'_'+site.get('name'))
        world.append(body)
    return mujoco.MjModel.from_xml_string(ET.tostring(doc).decode())


def render(episode, kind, preview=False):
    prefix = 'wuji2' if kind == 'wuji' else 'sharpa'
    output = episode/prefix
    z = np.load(output/f'{prefix}_human_trajectory.npz', allow_pickle=False)
    n = len(z['time_s'])
    rgb_dir = Path((episode/'source_rgb_frames.txt').read_text().strip())
    first = cv2.imread(str(rgb_dir/'frame_000000.jpg'))
    assert first is not None
    height, width = first.shape[:2]
    intrinsics = z['camera_intrinsics_fx_fy_cx_cy']
    model = viewer_model(kind, width, height)
    data = mujoco.MjData(model)
    for gi in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[gi])).name
        if body_name.startswith('left_') or (kind == 'wuji' and body_name.startswith('l_')):
            model.geom_rgba[gi,:3] = [.35,.7,1.]
        elif body_name.startswith('right_') or (kind == 'wuji' and body_name.startswith('r_')):
            model.geom_rgba[gi,:3] = [1.,.62,.32]
    model.vis.global_.fovy = np.rad2deg(2*np.arctan(height/(2*intrinsics[1])))
    model.vis.headlight.ambient[:] = [.68,.68,.68]
    model.vis.headlight.diffuse[:] = [.6,.6,.6]
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.
    camera.lookat[:] = [0,0,.7]
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    selected = set(np.linspace(0,n-1,6).round().astype(int))
    snapshots = []
    writer = None
    staging = output/f'human_{prefix}_camera_replay.staging.mp4'
    if not preview:
        writer = imageio_ffmpeg.write_frames(str(staging), (width,2*height), fps=30,
            codec='libx264', pix_fmt_in='rgb24', output_params=['-crf','19','-preset','fast'])
        writer.send(None)
    for fi in range(n):
        if preview and fi not in selected:
            continue
        bgr = cv2.imread(str(rgb_dir/f'frame_{fi:06d}.jpg'))
        assert bgr is not None and bgr.shape[:2] == (height,width), fi
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        for si,side in enumerate(('left','right')):
            draw_points(rgb,z['human_points_camera_m'][fi,si],intrinsics,side)
            adr = model.joint(side+'_floating').qposadr[0]
            data.qpos[adr:adr+3] = z[f'{side}_root_xyz_camera_m'][fi]
            data.qpos[adr+3:adr+7] = Rotation.from_rotvec(z[f'{side}_root_rotvec_camera_rad'][fi]).as_quat()[[3,0,1,2]]
            for name,value in zip(z[f'{side}_{prefix if kind == "sharpa" else "wuji"}_joint_names'],
                                  z[f'{side}_{prefix if kind == "sharpa" else "wuji"}_joints_rad'][fi]):
                data.qpos[model.joint(str(name)).qposadr[0]] = value
        mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera,scene_option=option)
        for view in renderer.scene.camera:
            view.pos[:] = [0,0,0]
            view.forward[:] = [0,0,1]
            view.up[:] = [0,-1,0]
        lower = renderer.render().copy()
        frame = np.concatenate([rgb,lower])
        label(frame,f'Human 21 points | frame {fi+1}/{n} | t={z["time_s"][fi]:.2f}s',(10,23))
        label(frame,f'{prefix.upper()} | left blue / right orange | camera frame',(10,height+23))
        label(frame,'Camera frame only; no robot-base calibration or arm IK',(10,2*height-12),.52)
        if writer is not None:
            writer.send(np.ascontiguousarray(frame))
        if fi in selected:
            cv2.imwrite(str(output/f'preview_{fi:04d}.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            snapshots.append(cv2.resize(frame,(320,400)))
        if fi % 100 == 0:
            print(episode.name,prefix,'render',fi,n,flush=True)
    renderer.close()
    assert len(snapshots) == 6
    sheet = np.concatenate([np.concatenate(snapshots[i:i+3],axis=1) for i in (0,3)])
    cv2.imwrite(str(output/'contact_sheet.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
    if writer is not None:
        writer.close()
        cap = cv2.VideoCapture(str(staging))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        for fi in (0,n//2,n-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
            okay,_ = cap.read()
            assert okay
        cap.release()
        staging.replace(output/f'human_{prefix}_camera_replay.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('kind',choices=('sharpa','wuji'))
    parser.add_argument('--preview',action='store_true')
    args = parser.parse_args()
    render(args.episode,args.kind,args.preview)
