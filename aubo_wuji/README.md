# Aubo i7H + WujiHand2 + XFlow 关键点闭环

本目录把现有 `../../code` 回放工程、`../keypoints21_to_wujihand2` 的手模型和远端 XFlow 连接起来。
默认模型 **rb_a12**，固定 **pt-rb-a12-g64-1114 / step_004664.pt**，任务 **flower**。
模型于现有5090服务热加载，deployment_id为 `254a4710063c62ff`；原hh_a12已卸载，启动注册表也已同步更新。
rb_a12原始预测时间为32步覆盖1.067秒、历史偏移16/30秒；客户端仍按操作员配置将输出映射到1.04秒，速度0.3。
不修改原回放源码，不复制 Python 环境，不加载本地模型权重。

## 完整流程

1. 持有原回放的独占锁，连接左右 Aubo、左右 Wuji；只读反馈，不自动上电。
2. 从网络相机SDK或ROS读取当前图、历史图和实时标定，保留曝光采样时间戳。
3. 核对左右手眼与当前相机的原始分辨率、K、D、光学坐标系。
4. 用实测臂关节角做 SDK FK，用实测手关节角做 Wuji URDF FK，得到相机系双手 21 点。
5. 经持久 SSH 隧道和持久 TCP 会话执行 `info → reset → infer`。
6. 核对模型训练目录、部署身份、会话、请求、采样时刻、左右顺序、单位和预测时间轴。
7. 将预测关键点按 Wuji 骨长归一化；联合求手根平移、旋转和每手 20 个关节角。
8. 用当前物理手眼和法兰安装变换得到 Aubo 法兰目标：

   `base_T_flange = base_T_camera @ camera_T_wuji_base @ inverse(flange_T_wuji_base)`

9. 原回放 `diagnose_ik` 标称候选 → SDK 逐帧 IK → 最近合法绕角 → 实际 DH 细化 → SDK FK 复核。
10. 原回放 `runtime_pair` 重定时、100Hz 插值、进场、限位/速度/加速度/末端误差检查；几何碰撞预检已按操作员要求禁用。
11. 再次核对设备、起点、文件哈希、反馈电流；仅 `run --execute` 调用原同步执行器。
12. 执行结束，原执行器确认停止和清理；重新读实测状态，进行下一轮。

本实现为 **retimed_replay：停稳观察 → 预测 → 全程预检 → 减速执行 → 再观察**。
它不是 RTC 异步控制器，也不是严格按预测的绝对时刻跟踪。
hh_a12 的 32 步原始预测时间为约 0.0083～0.2667 秒；通过服务端 info 读取，不写死。
计划保留各预测点的相对时间，再由回放按默认 0.2 倍速及动态约束统一放慢。
保留原回放至少 10 秒进场、1 秒首帧保持、2 秒执行预热；由于包含实测 t=0，进场通常接近原地保持。
因此每轮有明显停顿，尚不能据此声称完成了自主 flower 任务。

## 环境和入口

运行目录：

```bash
cd '/media/descfly/新加卷/ty/replay/client/aubo_wuji'
./run doctor
./python -m unittest discover -s tests -v
```

`./python` 复用 `../../code/python`，包括 Python3.12、ROS Jazzy、Aubo SDK 0.26 和 Wuji SDK 动态库。
仍支持原工程的 `REPLAY_PYTHON`。整体目录移动后，在 `config.json` 修改相对依赖路径；
若回放目录不在默认位置，启动器还可设置 `AUBO_WUJI_REPLAY_ROOT`。
手 FK/IK 只依赖 NumPy/SciPy，不要求安装 MuJoCo。

所有子命令默认只读。`--config` 放在子命令前：

```bash
./run --config /absolute/path/to/config.json doctor
```

## 先查看相机、设备和服务

```bash
./run camera      # 读取网络相机和标定，保存原图/适配图；不连接机器人
./run snapshot    # 只读双臂/双手反馈；不使能、不发送目标
./run info        # 查询模型并创建通信会话；不连接机器人
```

默认直接读取网络相机，不依赖ROS驱动：

- `192.168.1.10:8090`，Orbbec Gemini 435Le，序列号 `CP4W86300030`。
- 当前使用原生640×400 RGB、10FPS；SDK实测K为1280×800标定K的等比例缩放，D不变，启动时核对。
- SDK RGB输出1280×800、10FPS，实测K/D与回放手眼元数据完全一致。
- 使用SDK全局曝光时间，不以发送或接收时间冒充采样时间。
- 模型历史目标仍为4/30秒；按实际时间戳选择最近帧，不按4个10FPS帧计算。
  当前历史约100ms，与133ms目标相差约33ms；配置明确允许最多40ms偏差，并保存实际偏差。
  这是网络10FPS模式的时间量化近似，任务效果仍需验证。
- Python3.12 SDK 2.1.2随 `vendor/camera_runtime` 提供，不修改系统或原回放环境。

需要ROS模式时将 `camera.backend` 改成 `ros`，由原工位驱动发布：

- `/camera/color/image_raw`：原始未校正的 `rgb8` 或 `bgr8` Image；有真实采样时间戳。
- `/camera/color/camera_info`：对应原始图像的 K、D 和分辨率。
- 光学 frame 为 `camera_color_optical_frame`，30 FPS。

ROS_DOMAIN_ID、RMW 实现和网络应与发布者一致。ROS模式不自动启动相机驱动，
不会将 JPEG、处理过的图像或任意 resize 后的图像当作原始数据。
需要约 4/30 秒的历史（其他模型读取 info 中的 offset），缺失时等待，超时即退出。

## 本地相机如何对应远端 cam0

找到并复用了回放配置里的左右手眼、桌面平面和 Wuji 安装变换：

- 左右 Aubo：192.168.1.3 / 192.168.1.2，端口 30004。
- 左右 Wuji：192.168.1.110:50001 / 192.168.1.111:50001，逐只核对序列号。
- 左手眼：`../../code/resources/calibration/reference_c4840512/result.json`。
- 右手眼：`../../code/resources/calibration/reference_d72abc19/result.json`。
- 碰撞桌面：`../../code/resources/calibration/reference_87ac58a1/observation.json`。
- 法兰安装变换：`../../code/config/human_replay.json` 中每侧的 `flange_T_wuji_base`。

本地标定原图为 1280×800；远端训练 cam0 原图为 640×400，且 K、D 不完全相同。
因此默认 `camera.geometry_mode = canonical_cam0` 是明确的**跨相机几何适配**：

1. 对远端原始 cam0 的每个像素，用远端 K/D 求光学射线。
2. 用本地 K/D 把同一条射线投影回本地原始图。
3. 双线性采样形成 640×400、保留远端畸变模型的虚拟 cam0 图；原始数组无损传输。
4. 服务端照常使用自己的 K/D 去畸变和缩放。

适配不旋转坐标轴、不平移光心，3D 关键点仍用本地物理相机光学系；不引入回放的跨桌面任务迁移矩阵。
按存档标定计算，约 99.59% 的远端像素射线被本地画面覆盖；缺失边缘填黑，低于98%拒绝。
此模式改变了服务指南“直接发送原始 cam0”的假设，是本客户端新增的兼容层，
**不是服务端已验收的原生相机路径，真实视觉特征和任务效果尚待验证**。
需要原生路径时设 `strict_raw`，它要求本地原始640×400的 K/D 与远端直接匹配。

启动时实际 CameraInfo 必须匹配回放手眼元数据；不以历史文件代替实时核对。
如果物理相机或安装位置已经变化，需要更新真实手眼，不应靠调宽容差绕过。

## 远端模型连接

默认 `server.transport = ssh`、`ssh_alias = b300`，使用本机现有 SSH 配置。
当前服务位于一个5090节点的两张卡；原B300实例已停止并释放资源。
固定hh_a12/step_008408，任务管理见 `SERVER.md`。
当前自动查询：

```text
/mnt/workspace/users/pengyang/workspace/outputs/aw-hha12-5090-1114-c7d1/endpoint.json
```

`./run info` 已完成真实握手验证。原wanchi任务已停止；新实例的地址文件位于自己的目录，
不再依赖原目录权限。也可把获知的计算节点内网地址填入：

```json
"server": {
  "transport": "ssh",
  "ssh_alias": "b300",
  "host": "GPU_NODE_INTERNAL_IP",
  "port": 18765
}
```

这里只展示要修改的字段，其他配置保留。`host` 是GPU计算节点地址，
不是 B300 公网 SSH 入口地址。手动建好隧道时可用 `transport=tcp`，
host/port 填本机隧道端点。服务端需要已加载 `hh_a12`；客户端不热加载或换权重。

每个客户端生命周期只建立一次SSH隧道和TCP连接；隧道绑定动态本地回环端口。
退出只关闭本次隧道，握手、超时或响应错误直接停止，不自动改连其他模型。
当前 `expected_deployment_id=34ca15a63ef03aa8` 锁定实际checkpoint，同时核对训练目录，
在会话内固定 `deployment_id` 和 `weights`，后续响应变化即拒绝。

服务端已有启动脚本。在**已分配GPU的计算节点环境**中运行：

```bash
bash /mnt/workspace/users/wanchi/workspace/code/XFlow/experiments/ego_pretrain_keypoints21/revisions/r5-posttrain-a12-20261114/training/scripts/deploy/run_server.sh
```

它读取同目录models.json，默认仅自动加载hh_a12；当前注册表指定模型GPU0、共享编码器GPU7，
需要对应可见GPU资源。本次独立注册表已将共享编码器改为GPU1，使用两卡，见SERVER.md。
调度提交和停止是独立管理操作，机器人客户端不自动提交或启动GPU任务。
模型加载完成后，获取GPU节点实际内网host和监听端口18765，填入客户端server.host/port即可。

## 预检和实机闭环

```bash
# 当前状态→真实模型→关键点重定向→完整Aubo/Wuji预检，不发送运动目标
./run run --steps 32

# 显式执行一轮
./run run --steps 32 --execute

# 每轮执行后重新观察，最多100轮；没有自动任务成功判定
./run run --steps 32 --iterations 100 --execute
```

`--steps` 为执行预测的前1～32步，客户端始终请求模型完整32步。
重定向以实测手根、手角度初始化，补入实测 t=0，防止丢失第一预测间隔。
所有关节限位、速度/加速度、末端FK误差与执行检查复用原回放，不因模型输出而放行不可达动作。
几何碰撞预检已按操作员明确要求完全禁用，包括手内自碰、臂/手、左右机器人、桌面和间隙检查。
配置必须显式为 `collision_check_policy=disabled_by_operator`；运行报告保存 `collision_checked=false`、`collision_passed=null`，不会冒充碰撞通过。
每电机绝对电流上限沿用1.5A；不求和、不修改MIT参数和负载/TCP。
单块手根平移距离上限已取消（`max_root_translation_m=null`）；旋转上限0.5rad、归一化指尖拟合误差上限15mm仍保留。
观察到开始执行超过60秒、机械臂起点漂移、模型身份改变等都会拒绝。
手指在观测或观测后变化超过0.005rad改为告警并继续，告警记录侧别、关节序号和最大变化量。
机械臂在观测后或规划后变化超过0.002rad也仅告警继续，不再因这一微小起点漂移阈值退出。
网络相机偶发的过期、超前或非递增曝光时间戳帧会被丢弃并告警；只有在5秒内始终等不到有效新帧才停止。
执行器的大幅命令不连续、跟踪误差、过流、限位和机械臂漂移仍是硬停止条件；碰撞由现场人工监督和急停承担。

默认 `persistent_servo=true`：一次启用双手和Servo，连续执行所有轮次，退出或异常时统一停止。
当前默认 `trajectory_sampling=source_waypoints`：仅发送32个原始预测路点（另含实测起点），不做客户端100Hz插值。
`scheduler_lateness_policy=warn_and_reschedule`：调度迟到超过30ms时仅告警并顺延后续路点，不因该阈值停止，也不连续补发积压指令。
持久Servo段间衔接时，双臂目标差超过0.002rad、双手超过0.03rad均只告警继续，记录到对应轮次的 `splice_warnings.json`。
这项告警策略不覆盖通信故障、反馈过期、过流、执行跟踪异常或控制器拒绝命令等停止条件。
这会延长实际执行时间；控制器拒绝命令、反馈过期、通信异常等仍会停止。
时间轴1.04秒、请求速度0.3，对应路点间隔约108.33ms，servoJoint的执行时间参数与此间隔一致。
路点之间由控制器完成运动；逐路点同步发送双臂和双手目标。样本速度/加速度越界时拒绝执行，不通过增加插值点或自动修改这32点处理。
去掉每轮10秒固定进场、1秒保持、2秒预热；起点差异仍通过按实际距离计算的过渡衔接（最小20ms）。
请求速度为0.5，双臂速度/加速度比例为额定上限的1.0；仍保留控制器额定上限和手部动态边界，超限时自动重定时。
段末持续发送最后目标，在后台等待双臂停稳并采集、推理、规划下一段；准备完毕后在同一Servo会话接续。
当前仍有约2–3秒段间推理规划停顿，未实现运动中的观测预测及无停顿拼接。
新运行的每轮准备时延保存在 `cycle_XXXX/timing.json`；持久会话最终停止结果在原工程执行日志中。

Ctrl+C停止；或另一个终端：

```bash
./stop runs/具体运行目录
```

执行中由原回放处理同步停止、Servo退出和双手禁用。确认细节保存在原工程
`../../code/output/xflow_*_execution.json` 的 `stop_results`、`cleanup_errors` 中。
本客户端和原回放共用设备独占锁；不应同时启动另一个设备命令源。

## 保存数据及离线复现

每次运行写入 `runs/日期_时间_随机号/`：

- `configuration.json`、`server_info.json`、`session.json`。
- 每轮 `observation.npz`：模型图像、实测关键点、初始手根/手角度/臂角度。
- `prediction.npz/.json`：原始模型响应和时间轴。
- `retarget.npz/.json`：Wuji关节角、相机系手根位姿、拟合误差、求解状态。
- `prepared.npz`、标称IK候选、左右SDK IK报告、`pair_checks.json`。
- 只有全部检查通过才有 `runtime.npz/.json`，含100Hz的52维左右臂手联合轨迹。
- `warnings.json`（如有）保存手指静止性告警；`timing.json`保存观测、推理重定向、规划和执行耗时。

内部52维顺序为 **左臂6、左手20、右臂6、右手20**，复用原执行器。
这与原转换包“右手26、左手26”的动作导出文件不同，不能互换。

```bash
./run infer --observation runs/某次运行/cycle_0000/observation.npz --steps 32
```

离线infer只请求服务和求解手动作，不连接机器人，不允许直接执行已保存的模型预测。
录制时间戳保留在文件，通信请求重新分配单调采样标识；它不伪装为当前实机观测。

## 验证范围

`./python -m unittest discover -s tests -v`：独立 Pinocchio 对照FK、解析Jacobian、
关键点拟合、坐标变换、真实TCP封装与持久会话、模型身份和坏响应拦截、
离线CLI端到端、相机光学射线投影、标定不匹配、过期观测与起点漂移。

另有显式只读硬件联调：

```bash
./python tests/live_readonly.py --live-readonly
```

该测试使用当前实测状态、**模拟模型响应和全黑模拟图像**，运行真实控制器IK/FK及真实碰撞网格，
从不调用执行器；结果标为test_only，不能作为真实模型或任务效果验收。
真实GPU验证已完成：三次flower录制观测往返0.794/0.744/0.673秒，完整32步Wuji重定向通过。
记录见 `SERVER.md` 和 `runs/real_model_20261114_171251_d91fd0/validation.json`。
该单窗口预测误差12.49mm、保持姿态基线5.62mm，不能支持任务效果结论。
后续网络相机实时链路已通过完整只读预检：见 `runs/20261114_172932_265f52/`。
已使用真实画面和实测状态通过5090模型、32步重定向、Aubo SDK IK及1,737帧动态碰撞检查。
随后按用户明确授权完成一次4步预测前缀的真机短测试，双臂和双手均确认停止，见VALIDATION.md。
未执行完整32步闭环任务，未验收任务成功率。

## 文件职责

| 文件 | 职责 |
|---|---|
| run.py | CLI、资源生命周期、逐轮闭环和数据保存 |
| network.py | SSH/TCP会话与严格XFlow身份/响应校验 |
| observation.py | 相机时间缓存、标定核对、真实状态与21点组装 |
| camera_geometry.py | 本地原始相机到服务cam0像素几何的显式适配 |
| hand_kinematics.py | Wuji URDF FK/Jacobian、骨长适配、带限位free-root IK |
| planning.py | 复用原SDK IK、100Hz重定时及同步执行器；显式禁用几何碰撞预检 |
| config.json | hh_a12/flower、路径、连接、相机及动作范围配置 |
| check_recorded.py | 真实模型录制数据往返、时间插值误差和离线Wuji重定向验证 |
| vendor/ | 从服务端读取的wire_v4和cam0标定快照 |
