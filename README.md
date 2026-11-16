# 机器人推理客户端

本项目将模型预测的双手动作接入机器人控制。默认的 Aubo i7H 双臂与 Wuji Hand 2 客户端读取相机画面和当前关节状态，向远端 XFlow 请求双手各 21 点的预测轨迹，再通过相邻的回放工程计算机械臂和手指的关节目标。

| 使用场景 | 入口 |
| --- | --- |
| 用远端模型预测 Aubo 双臂与 Wuji 双手的下一段动作 | [Aubo/Wuji 客户端](aubo_wuji/README.md) |
| 把已录制的相机坐标系手部关键点离线转换为 Wuji 手根位姿和手指角 | [21 点转换工具](keypoints21_to_wujihand2/README.md) |
| 在 AgiBot＋Sharpa 平台上验证推理或使用对应硬件接口 | [AgiBot 接口](agibot_inference/README.md) |

## Aubo/Wuji 快速检查

将本仓库放在工作目录的 `client/`，把 Aubo/Wuji 回放工程放在同级的 `code/`。客户端默认复用 `code/python`、SDK、运动规划代码和工位标定。默认模型为 `rb_a12`，任务为 `flower`。模型身份、SSH 连接、相机参数及回放路径见 [默认配置](aubo_wuji/config.json)。

从本仓库根目录运行：

```bash
cd aubo_wuji
./run doctor          # 本地核对依赖、模型资源和标定
./run info            # 连接推理服务，核对模型和会话
./run run --steps 32  # 读取相机与机器人状态，完成预测和规划，无运动指令
```

使用其他任务配置时，将 `--config` 放在子命令前，例如 `./run --config config.tidying.json info`。实际运动需要显式指定 `--execute`。默认配置关闭了几何碰撞预检，因此上述规划结果不包含碰撞通过结论。

客户端将每轮观测、模型预测、重定向结果和计划写入配置的 `output_root`（默认 `aubo_wuji/runs/`）。排查模型连接可看[服务说明](aubo_wuji/SERVER.md)，核对已有测试覆盖范围可看[验证记录](aubo_wuji/VALIDATION.md)。

## 本地 Pi0.5

使用单独打包的 `pi05_client/` 时，从该包的顶层 `serve` 启动模型服务，顶层 `run` 启动客户端。该入口读取包内的 `config.json` 和 `code/`。远端 XFlow 客户端使用本仓库的 `aubo_wuji/config.json` 及相邻的回放工程。
