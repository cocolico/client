# AgiBot + Sharpa 本机接口

当前本体：头部与双臂继续通过 GDK 控制，左右末端已换成 Sharpa Wave SE 灵巧手。每只手有 22 个位置关节。仓库中原有 FastWAM、ours、Pi0.5 权重仍按旧的 19 维状态和动作训练，其中含左右单自由度夹爪；**这些权重只用于保存轨迹的离线推理校验，不允许驱动当前本体**。目前没有将 1 维旧夹爪动作映射到 Sharpa 的 22 维手指动作。

```bash
ssh real
source ~/env_agibot.sh
bin/agibot server status
bin/agibot test --task guizi --model pi05
bin/agibot hand schema
bin/agibot hand read --side both
```

Sharpa 独立位置控制使用 `/home/descfly/code/sharpa/.venv`，与 GDK 的 Python 环境隔离。`hand read` 会连接并使能选定手部，但不发送位置目标。单关节测试命令如下；每次先读当前 22 维反馈，再只改变指定关节的目标值，同时把其他 21 个关节设为当前反馈值。命令发送后等待反馈进入容差，否则报错。

```bash
bin/agibot hand move --side left --joint index_MCP_FE --delta-rad 0.03 --execute
```

`move` 必须指定单只手、关节和 `--execute`；单次相对位移上限为 0.1 rad，默认到位容差 0.005 rad。测试代码可加 `--fake` 使用 SDK 模拟手。左右手真机单关节 `+0.08 rad` 位置命令及反馈已通过。运行真机命令前确认手部空载且周边无障碍。具体接口和命令见 [Sharpa 位置控制](docs/SHARPA_POSITION.md)。

旧模型离线验证：

```bash
bin/agibot server restart --task guizi --model fastwam
bin/agibot test --task guizi --model fastwam
bin/agibot replay data/raw_data/guizi/0914柜子_20260914174235
```

`once`、`interactive` 和 `replay --execute` 均已禁用。模型权重、文本特征、日志和轨迹位于 `/data/agibot_inference/` 或其指向的外部路径；Python 环境不纳入仓库。旧 19 维模型的训练格式见 [历史接口](docs/INTERFACE.md)，部署和目录见 [部署文档](docs/DEPLOYMENT.md)。

EEF 短段回放使用原始 `-tf.pbdat` 的 `base_link → arm_l_end_link` 或右臂末端位姿；命令将录制片段的相对变化叠加到当前机器人姿态，不回到旧机器人的绝对首帧。默认仅预检，实机需显式 `--execute`，最多 1.5 秒，不控制 Sharpa 手。见 [EEF 回放说明](docs/EEF_REPLAY.md)。
