# G2 drink 腰部接入（2026-10-22）

## 范围

只在 real 的 G2 client 且 `--task drink` 时启用。其他任务的执行器、固定身体坐标适配及 3B client 不改。原代码备份：`backups/drink-waist-20261022/source-before.tgz`。

- 78 维协议保持不变，0-based 索引 27 为 `idx05_body_joint5` 腰部旋转，单位 rad。drink 的 mask 有 72 个有效维；其余 G2 任务仍为 71 个。
- state[27] 来自实际身体反馈，action[27] 为模型目标。索引 28–31 不开放；前四个身体关节固定为启动归位后的首次观测值。
- task 对应服务器 `g2/drink_head_only`，instruction 与服务端逐字匹配。
- 目前已核对的 drink 服务为 joint_relative / joint / tri；其他 policy 不在本次 drink 实机入口支持范围。

## 每个 chunk

1. 实测双臂、手、头及五个身体关节；用实测身体和头关节 FK 建立本 chunk 首帧相机锚点。
2. 输入 state78 + task instruction + validity + head RGB；三个服务均返回首帧相机系下的 XYZ+6D 目标，不重复合成 relative 模型内部残差。
3. 用这一固定锚点恢复 base_link 目标；每帧模型时间为 50 ms，每帧分为五个 100 Hz 子帧。首次平滑进入沿用至少 3 秒的安排。
4. 每个子帧同时插值 EEF 和腰角，然后以该子帧的**目标身体角**做 FK，把双臂 base_link 目标转换为 arm_base_link 目标；头部姿态求解也用同一目标身体角。头 XYZ 仅诊断。
5. 复用 replay 的 ReactiveControl 消息构造：同一帧发送双臂 ABS_POSE、腰部 ABS_JOINT 和头部 ABS_JOINT；双手通过原 Sharpa 接口发送。双臂 IK 留给原厂，不用实测腰角代替预测腰角，也不把腰角仅当 IK seed。

原 replay 的 fault/collision、双臂/头跟踪检查与监测频率保留；响应有效期仍为 2 秒；计划完成后再检查一次有效期。没有关闭原厂保护。关节范围、头部/身体步长等现有检查仍保留。

## 启动（会执行真机，本次验证没有执行这些命令）

在 real：

```bash
cd /home/descfly/code/agibot_inference
bash run_inference.sh --model joint_relative --task drink --steps 32
# 或将 model 改为 joint / tri
```

沿用原入口的启动归位、环境和 SSH 隧道；分别连接 172.31.68.233:18868 / 18869 / 18870，不是旧的 8868–8870 服务。

## 离线验证

新增 task/mask、实测锚点、逐子帧目标腰角、固定非 yaw 身体关节、消息字段、故障零发送/中途停止及原任务回归。测试里的 GDK 消息由 fake publisher 接收，不发布 DDS。

使用真实 drink 录制数据第 0、27 帧（后一段包含明显转腰），从原始身体反馈取得完整五维身体和三维头关节。核对 camera/base/arm_base 往返；再用三台真实模型服务各推理两次。报告在 `backups/drink-waist-20261022/validation/validation.json`，包含模型时间、32×78 输出、逐帧规划结果及 `robot_io=false`。

正式离线入口例子（不加 --execute）：

```bash
source env.sh
python scripts/run_mixed76_client.py \
  --representation g2-camera78 --model tri --embodiment g2 \
  --task drink --inference remote --steps 32 \
  --observation backups/drink-waist-20261022/validation/recorded_drink_27.npz \
  --output backups/drink-waist-20261022/validation/cli_tri.npz
```

离线通过不等于物理 IK、控制仲裁或真实轨迹跟踪已经验收；本次没有初始化机器人、归位或发送运动命令。全仓历史测试另有外置录制样本缺失及旧测试未传 model 的既有失败，不能宣称全仓全部测试通过。

## 本次最终结果

- 相关回归：119 项通过。正式目录全仓：179 项中 174 项通过，5 项既有错误（4 项缺失外置录制样本，1 项旧测试未提供 model）；与改动前失败集合一致。
- 三个真实服务各两个录制观测，共 6/6 返回有限的 32×78 动作并通过规划；正式 CLI 的三个模型入口也分别通过。
- 最后一轮网络请求加离线规划：joint_relative 0.86–1.10 秒，joint 0.82–0.91 秒，tri 1.17–1.33 秒。仅规划约 62–65 ms。本次样本均小于 2 秒；不保证未来网络时延恒定，超时仍停止发送。
- 录制目标在 camera/base/arm_base 间往返位置误差小于 1e-12 m。此项证明坐标变换一致，不代表真机执行精度。
- robot_wrapper.py、g2_chunk_camera.py、replay_full_eef.py、replay_public_eef.py 的 SHA256 与备份相同。
