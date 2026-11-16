# 历史 19 维模型接口（当前仅离线校验）

FastWAM、ours、Pi0.5 的现有权重使用旧的单自由度夹爪数据训练。当前本体的左右末端均为 Sharpa Wave SE，每只手 22 个位置关节；旧 19 维动作不能发送给当前机器人。`bin/agibot test` 用保存的轨迹进行模型推理；`bin/agibot replay EPISODE` 只解析历史轨迹。实时推理和实机回放入口已禁用。

离线观测：`active_rgb` 为 `uint8 RGB [480,640,3]`；`state` 为有限 `float32 [19]`，单位 rad，来自轨迹的头、双臂和旧夹爪反馈。动作输出为有限 `float32 [32,19]` 绝对角度，20 Hz。19 维顺序如下：

| 索引 | 历史关节 |
|---:|---|
| 0–6 | 左臂 `idx21_arm_l_joint1` … `idx27_arm_l_joint7` |
| 7 | 旧左夹爪 `idx31_gripper_l_inner_joint1` |
| 8–14 | 右臂 `idx61_arm_r_joint1` … `idx67_arm_r_joint7` |
| 15 | 旧右夹爪 `idx71_gripper_r_inner_joint1` |
| 16–18 | 头部 `idx11_head_joint1` … `idx13_head_joint3` |

旧夹爪 `-0.785 rad` 表示张开，`0 rad` 表示闭合；这个定义不适用于 Sharpa 手。躯干的 5 个关节不在模型状态或动作中，历史 replay 文件中仍有其首帧目标。

本机 TCP 服务位于 `127.0.0.1:19000`。消息格式为 4 字节大端长度加 UTF-8 JSON，NumPy 数组编码成包含 `__array__`、`dtype`、`shape`、`data` 的对象。`info` 响应标识任务、模型、训练步数、20 Hz、32 步和 19 维顺序；`infer` 接收一帧 RGB 和 19 维状态，返回 `[32,19]` 动作。客户端在离线测试时核对这些字段。一台服务一次加载一个模型和任务。

当前 Sharpa 的反馈、限位和独立位置控制见 [Sharpa 位置控制](SHARPA_POSITION.md)。训练新的整机模型前，须定义 GDK 头/双臂与左右 Sharpa 手共 44 个手部关节的观测时间戳、名称顺序、动作语义和采样率；不能给旧模型填假夹爪值或直接复用旧复位轨迹。
