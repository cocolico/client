# G2 chunk 首帧相机坐标适配

## 当前状态

已实现数值适配与录制数据离线验证，复用原有 client 的观测/动作转换入口。
后续已接通显式 `--representation g2-camera71` 的71维联网入口，并用模拟server验证；完整启动命令和server协议见 [G2_CAMERA71_PROTOCOL.md](G2_CAMERA71_PROTOCOL.md)。不带该参数的旧命令仍保持旧v4绝对坐标模式。真实远端模型尚待用户启动后联调。

数值适配阶段没有修改网络client；后续联网接入只给该client增加显式71维契约。SSH连接、`scripts/robot_wrapper.py`、Sharpa控制和旧client执行策略均未修改。没有启动机器人、复位或使能双手。

## 新增的数值接口

`scripts/g2_chunk_camera.py` 的 `G2ChunkCamera(urdf, body_target)`：

- `body_target` 必须显式传入本次固定的五个身体关节控制目标，顺序 `idx01_body_joint1` 至 `idx05_body_joint5`；不能默认零，也不能悄悄改用反馈。
- `begin(native61, sample_ns=...)` 复用 replay 的 FK 得到 `base_from_camera`，产生只读 chunk 上下文与相机系 state71。
- `decode(context, actions71)` 用该请求冻结的参考系恢复 base 目标，复用原来的 native61 映射和头部 Z-X-Y 逆解，返回 `(actions61, diagnostics)`。
- 相机暂等同 `head_link3`；XYZ 单位米、qpos 单位弧度、6D 是旋转矩阵前两列拼接。
- 71 维顺序：左 EEF9 / 左手22 / 右 EEF9 / 右手22 / 头 EEF9。这是物理适配层布局，不是未经确认的网络 padding 布局。
- 头部预测 XYZ 仅记录逐帧误差和预测/可实现坐标；头部关节仍受原有限位和旋转残差检查。原 replay 的 XYZ 检查默认保留。

原 `scripts/run_mixed76_client.py` 增加两个数值钩子：

```python
state, context = prepare_g2_observation(obs, sample_ns=stamp, adapter=adapter)
# 这里保留现有 client.infer；接网前按新版模型真实契约明确 wire 映射。
actions61, diagnostics = restore_g2_actions(actions71, context=context, adapter=adapter)
# 之后仍调用原 robot.execute_sharpa61；不另写执行器。
```

省略 adapter 时，两函数严格执行原先的 61→76→61 映射。不要直接将 state71 发给旧 v4 服务。

## 离线验证命令（real）

```bash
cd /home/descfly/code/agibot_inference
PYTHONPATH=scripts:/home/descfly/.cache/agibot/app/gdk/lib \
/home/descfly/miniconda3/envs/gdk-py310/bin/python scripts/run_mixed76_client.py \
  --model pi05 --embodiment g2 --task yifu \
  --coordinate-check /home/descfly/code/egoav/pico/data/lerobot_versions/egoav_new_pilot_20261020/agibot_yifu/offline_replay.npz \
  --output /tmp/g2_coordinate_report.json
```

该模式不连接 server、不创建 GDK 实机对象、没有 `--execute`；模型参数只复用现有 CLI，**不会加载或测试该模型**。它将已有录制动作转换成模拟的相机系输出，再验证恢复结果，不证明模型预测正确或联网已打通。

## 验证与备份

- 540 帧、17 个 32 帧 chunk（最后一个不足32）：逐帧检查双臂位姿、头部角度、44个手指关节。
- 数值报告：`reports/g2_chunk_coordinate_20261021.json`。
- 新增测试：`tests/test_g2_chunk_camera.py`。
- 新增 12 项测试全部通过；全量 123 项中 122 项通过，1 项为下述改动前已有的旧测试错误。
- 修改前备份：`/home/descfly/code/agibot_inference_backups/g2_chunk_adapter_before_20261021.tar.gz`。
- 原有文件只修改 `run_mixed76_client.py` 与 `replay_public_eef.py`；执行器与网络客户端已逐字节对照备份保持不变。
- 全量测试原先已有失败项 `test_mixed76.ContractTests.test_tcp_client_keeps_76d_response`：测试遗漏必填 `model` 参数。本次不改变旧协议/测试以掩盖该失败。
- `git diff --check` 被已有的空 Git object 阻断（`.git/objects/bc/2d1ff1630d804879a0de1d84b1cf8e03bb0151`，文件日期 9 月 24 日）；未擅自修复仓库。本次以备份归档逐文件比较确认改动范围。

联网所需字段、归一化责任、相机字段和chunk时序已在 G2_CAMERA71_PROTOCOL.md 固定。接下来只需按该契约启动真实远端server并填入连接地址，不能用旧76维绝对坐标模型代替新版相对坐标模型。
