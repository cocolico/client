# EEF 短段相对回放

## 新增：双臂完整 EEF + Sharpa 双手实际 qpos 回放（2026-10-15）

本节对应 `scripts/replay_full_eef.py`；下方原来的 `bin/agibot eef-replay`
仍然是单臂短段相对回放，没有改动。新版不是全身轨迹回放：

- 躯干 5 关节、头部 3 关节：仍只恢复到录制首帧；双臂先恢复首帧。
- 双臂：完整的 `base_link` 下绝对末端位置与姿态，按原 TF 时间节奏回放。
- 双手：读取 `record/-sharpa-left-state.pbdat` 和 `-sharpa-right-state.pbdat`
  的 `JointState.position`，每手 22 维、弧度；按关节名称映射到 SDK 顺序。
  **不读取 target 控制目标、不使用 Type、不发送旧夹爪或底盘指令。**
- 双手先从实测位置以平滑插值进入首帧，再与双臂共用一个回放时钟。
  不再写死 36 秒，读取整段双臂 TF；双手必须覆盖该时段。

### 命令（在 real 上执行）

```bash
ssh real
source /home/descfly/code/agibot_inference/env.sh

# 先预检：不初始化 GDK、不连接或使能双手、不发送运动指令
python scripts/replay_full_eef.py --episode "1015拧灯泡_20261015104432"

# 现场准备好以后才执行：会复位躯干、头臂和双手，并开始整段运动
python scripts/replay_full_eef.py --episode "1015拧灯泡_20261015104432" --execute
```

`--episode` 后填写数据文件夹名，默认根目录为
`/home/descfly/code/egoav/pico/data/raw_data/1015拧灯泡`。
也保留原来的完整路径用法，两种方式不能同时传入：

```bash
python scripts/replay_full_eef.py \
  "/home/descfly/code/egoav/pico/data/raw_data/1015拧灯泡/1015拧灯泡_20261015104432"
```

默认双手配置是仓库内 `configs/sharpa_hands.json`，可用 `--hands-config PATH` 指定。
当前配置使用 `/opt/sharpa-wave-sdk`，左手序列号 C8519039C852、右手 CE569038CE55。

### 同步、异常与使用前提

1. 使用消息内的源时间戳对齐，不使用 PBDAT 接收时间戳。按双臂时间点线性插值双手
   实际反馈；反馈间隔超过 100 ms、非有限值、错误关节集合或不递增的时间戳会拒绝。
   对齐后相邻手目标差超过 0.35 rad 也会拒绝；不截断反馈数值，SDK 限位仍有效。
2. 端点允许最多 20 ms 的采样相位差，以端点值保持并在 `edge_hold_ms` 明确报告；
   超过则拒绝，不自动裁剪约 1 秒的缺失段。报告包含频率、最大间隔与端点保持时间。
3. 保留原来的 TF 连续性、首帧到位、控制器故障、碰撞状态、EEF 跟踪和 100 ms
   调度超时检查。手部首帧不到位、SDK 发令异常会退出；手或臂发送失败后不再发后续帧。
4. **实际执行前停止 Pico、手套遥操和其他推理/回放，确认相同场景、周围安全并可急停。**
   回放持有现有 `.all_teleop.lock` 和 `.hands_control.lock`，不停止或抢占已有控制者。
   手工启动、没有使用这些锁的其他控制程序仍需操作者自行退出。
5. Ctrl+C 停止后续发送并关闭本次双手 SDK 连接；不是硬件急停，也不保证取消已下发的
   躯干运动或释放手指保持力。原厂运动服务和采集服务不由该脚本重启或修改。

### 本次离线验证与备份

备份：`/home/descfly/code/replay-sharpa-backup-20261015.o1Qkj5`。
包含当前代码、配置、测试及 Git 元数据；不重复复制未改动的 24 GB 模型权重、数据和日志。
基线 60 项测试通过；新增读取、对齐、完整时长、无硬件 dry-run、控制锁、平滑进入、
双臂/双手同步、首帧不到位、启动超时、异常退出等 17 项测试，
全量 77 项通过（SDK 边界采用模拟硬件）。

该数据目录 51 条全部完成只读预检：23 条通过、28 条拒绝。
拒绝包括 23 条触发原有 TF 连续性检查、4 条手部覆盖缺约 1 秒、1 条手部间隔约 225 ms。
没有放宽检查、补造数据或更改原始数据。
示例 `1015拧灯泡_20261015104432`：48.99 秒、双臂各 4900 帧、双手各 4901 帧，
双手约 100.02 Hz，最大间隔左 39.19 ms / 右 43.77 ms，末尾保持约 5 ms。

**尚未进行双手或机器人真实运动验收，不保证模拟通过等同于实机任务成功。**

---

原始 AgiBot 轨迹的 `record/-tf.pbdat` 记录 `base_link → arm_l_end_link`、`base_link → arm_r_end_link` 的位置和四元数。这是 TF 末端位姿；`-hal-left_ee_data.pbdat`、`-hal-right_ee_data.pbdat` 是旧夹爪反馈，不是 EEF 位姿。当前本体没有旧夹爪，脚本不向 Sharpa 手发送任何命令。

```bash
ssh real
source ~/env_agibot.sh
EP=/home/descfly/code/RoboTwin-AV/envs/realworld/agibot/data/raw_data/guizi/0914柜子_20260914174235
bin/agibot eef-replay "$EP" --side left --seconds 1.25
bin/agibot eef-replay "$EP" --side left --seconds 1.25 --execute
```

默认只解析并验证 TF，不连接机器人。`--execute` 使用 GDK `end_effector_pose_control()`，只控制选定的左臂或右臂；`move_ee_pos()` 是旧夹爪接口，不能用于笛卡尔回放。脚本以约 100 Hz 播放所选片段，将录制首帧之后的 `base_link` 平移和旋转变化叠加到**当前**末端姿态。因此这是相对运动效果验证，不是旧场景中的绝对位姿重现，也不保证任务成功。每次运行都以当时的实机姿态为起点；重复执行会继续累积位移和转角，不能当作幂等命令。

安全边界：时长最多 1.5 秒；相对首帧位移不超过 3 cm、旋转不超过 12°；单帧不超过 3 mm、1.5°；检查录制时间戳、实时 TF 新鲜度、控制器错误和碰撞状态，执行落后 100 ms 即停止发送。完成后读取末端反馈，误差超过 1 cm 或 5° 则报失败。脚本不会复位躯干、头、双臂或手部，也不会混用关节伺服。

2026-09-26 已验证 guizi 上述左臂片段：126 点、1.25 秒，录制片段最大位移 1.33 cm、最大转角 8.16°；真机目标误差 1.6 mm、0.99°。事后读取左末端位置由约 `(0.6243,0.2846,0.9357)` 到 `(0.6343,0.2920,0.9375)` m，右末端保持约 `(0.5998,-0.2499,0.9440)` m。未对右臂、Sharpa 手或完整 36 秒轨迹做实机回放。完整轨迹从当前姿态映射可能产生数十厘米和大角度运动，不能直接解除短段限制。
