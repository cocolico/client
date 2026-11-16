# 公共坐标系 XYZ＋6D 样本回放

在 `real` 的 `/home/descfly/code/agibot_inference` 使用。原来的
`replay_full_eef.py` 原始记录回放入口保持兼容。

## 命令

```bash
cd /home/descfly/code/agibot_inference
source env.sh

# 离线检查：不初始化 GDK，不连接 Sharpa，不启动运动。
python scripts/replay_public_eef.py \
  --dataset /home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu

# 由操作者执行：先恢复本条首帧，再回放完整头部、双臂、双手。
python scripts/replay_public_eef.py \
  --dataset /home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu \
  --execute
```

启动前退出遥操、推理及其他控制程序，确认运动区域和实体急停。
Ctrl+C 停止本循环继续发送，不等于硬件急停。

## 输入和执行语义

- 默认 `--source state`，保持原 replay 的“实际反馈回放”语义。
  显式 `--source action` 才回放采集的控制目标，两种数据不混用。
- 71 维顺序：左 EEF9、左手22、右 EEF9、右手22、头 EEF9。
  位置单位米，手部单位弧度；6D 为旋转矩阵前两列依次拼接。
- 当前只支持这类带 `manifest.json`、`offline_replay.npz`、
  `meta/robot.urdf` 的单 episode 验证包，不是任意 LeRobot 目录。
  NPZ 是转换时已逐元素验证等同 Parquet 的数值镜像；运行前核对 NPZ、
  Parquet、URDF、元数据的 SHA256。没有把 PyArrow 安装进 GDK 环境。
- Agibot 公共坐标就是 `base_link`，不需要像 Piper 那样减去底座偏移。
  不是 chunk 相对坐标，不按当前机器人姿态重新对齐轨迹。
- 双臂 6D 恢复为矩阵，再编码成 GDK 协议要求的 xyzw；不引入本地臂 IK。
  双臂 IK 继续由原厂控制器求解。
- 头部使用数据内 URDF 的完整 body/head 链。从 EEF 矩阵解析求
  yaw/roll/pitch（ZXY），不是直接发送保存的头部 qpos。保存的关节数据
  只供躯干/双臂首帧准备和离线对照。仅支持已校验的 Z/X/Y 头部结构。
- 躯干仍只恢复首帧。头部反解以这组固定躯干角为条件，FK 回代要求
  位置误差 ≤1mm、旋转误差 ≤0.1°，不能将明显的躯干运动伪装为头部运动。
- 20Hz 源数据在运动前按原时间轴插值到 100Hz：XYZ/手部线性插值，
  旋转使用 SO(3) 最短弧插值；不改变总时长、不加入额外帧移位。
- 保留原躯干恢复、首帧平滑进入、控制锁、双臂连续性检查、反馈检查、
  100ms 调度超时、Sharpa 生命周期和同帧失败即停止后续发送的行为。
- 新头部模式复用 `AgiBotRobot._send_combined_eef_head`，经 `/gdk/model_predict`
  一条消息发送双臂 EEF 和头部 qpos，以避免原先分开通道的仲裁冲突。
  这是用户批准的发送接口变动：原 EndEffectorPose 的 50ms lifetime 不再用于
  新模式，联合消息沿用现有 10ms trajectory_reference_time。未改原厂仲裁参数。
- 头部在原 10 帧检查节奏下核对实际 qpos，偏差 >10°（0.174533rad）停止；结束时也使用同一阈值检查。
  旧 CLI 未提供 head_path 时完全保留原双臂 EndEffectorPose 发送通道。

## 已验证与未验证

样本 yifu：540 个 20Hz 源帧、2696 个 100Hz 回放点、26.95 秒。
默认 state 的头部 FK 回代最大位置误差约 0.294mm，姿态误差接近浮点精度。
误差包含固定首帧躯干与录制躯干轻微反馈变化、采样插值的差异。

已做数值验证、原 replay 回归和联合消息/失败顺序的离线测试。
尚未启动真机；离线通过不代表现场标定或硬件跟踪已验收。
右腕相机没有翻转，原数据没有改写。

修改前备份：
`/home/descfly/code/agibot-replay71-backup-izWpkD/before.tar.gz`
