# 21 keypoints → wuji2

这个压缩包包含可独立运行的转换脚本、手模型资源、核验脚本和可选的原图对照视频脚本。目标模型：Wuji Hand 2 early Beta v2026.6.27。

## 输入

`EP/keypoints21_camera.npz`，至少包含：

- `joints_camera_m`: `(N,2,21,3)`，米；双手顺序为 `left,right`。
- `sides`, `visible`, `intrinsics`, `frame_indices`, `valid_frame_indices`。
- `source_frame_indices`, `source_timestamps_ns`, `target_timestamps_ns`, `is_duplicate`。

目标时间戳必须严格递增。光学相机坐标系为 X 向右、Y 向下、Z 向前。输入样本是 30 FPS 时间轴；脚本根据 `target_timestamps_ns` 计算实际 FPS，并保留源时间戳和重复帧标记。

## 运行

安装 `requirements.txt` 中的依赖，然后将 EP 指向输入 episode 目录：

```bash
EP=/absolute/path/to/episode
python convert.py --source "$EP" --output "$EP/wuji2"
python verify.py "$EP" wuji
```

示例输入可参考原始样本目录 `HumanHand_A_30fps_pose_20261114_samples/episode1299`。如果有 RGB JPG 序列，再创建 `EP/source_rgb_frames.txt`，内容为 JPG 目录的绝对路径，文件名需为 `frame_000000.jpg` 等，可运行：

```bash
python visualize.py "$EP" wuji
```

## 结果

- `wuji2_camera_replay_trajectory.npz`：左右手根 4×4 位姿、四元数、20 关节角及时间戳。
- `wuji2_actions_camera_right_left.npy`：右手后左手，每手 `[xyz_m(3), rotvec_rad(3), joints_rad(20)]`，总 52 列。
- `wuji2_hand_joints_left_right.npy`：仅关节角，左手后右手。
- `wuji2_human_trajectory.npz` 与 `validation.json`：拟合点、误差和求解状态。

算法使用 free-root：手根 3D 平移、3D 旋转与手指关节逐帧联合求解。Wuji2 先用 Wuji 中性骨长替换每段人手骨长，保留指骨方向，再做加权 IK；另存的左 +90° / 右 −90° 安装方向仅是参考动作，主回放不使用。这些根位姿仍在相机坐标系中；机械臂回放需另外求相机到机器人基座及手根到法兰的变换。
