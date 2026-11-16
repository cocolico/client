# 已启动的 rb_a12 / flower 服务

2026-11-14 22:56更新：现有5090任务已热加载rb_a12，设为默认并卸载hh_a12。
固定权重pt-rb-a12-g64-1114/checkpoints/weights/step_004664.pt，deployment_id=254a4710063c62ff。
客户端及远端启动models.json已同步更新，旧注册表备份为models.json.before-rb-a12。
下面hh_a12的测量数据为切换前历史记录，不是rb_a12验证结果。

## 当前部署：5090 池

已按用户要求使用 `--pool 5090 --nodes 1 --gpus-per-node 2`，替代下方已停止的B300实例。

- 任务 `aw-hha12-5090-1114-c7d1`，ID `job-8hgou25s9x`。
- 一个5090节点、两张卡，策略GPU0、共享T5/VLM编码器GPU1。
- 地址文件 `/mnt/workspace/users/pengyang/workspace/outputs/aw-hha12-5090-1114-c7d1/endpoint.json`。
- 当前端点 `10.0.0.142:18765`，客户端自动查询，不固定IP。
- 仍使用hh_a12、pt-hh-a12-g64-1114、step_008408，deployment_id仍为34ca15a63ef03aa8。
- 使用源码支持的 `XFLOW_ATTENTION_BACKEND=sdpa`，没有修改模型权重或原共享环境。
- SSH别名仍叫b300，它只作为开发入口和隧道跳板，推理任务实际属于5090池。

```bash
ssh b300 'submit --pool 5090 --status aw-hha12-5090-1114-c7d1'
ssh b300 'submit --pool 5090 --logs aw-hha12-5090-1114-c7d1 --tail 50'
ssh b300 'scancel --pool 5090 aw-hha12-5090-1114-c7d1'
```

5090录制观测验证通过：`runs/real_model_20261114_172621_7f5ee7/`，往返约1.02秒，
完整32步Wuji重定向通过；单窗口MPJPE12.33mm、保持姿态基线5.62mm，不代表任务效果。

实时网络相机、实测状态到5090模型及Aubo完整只读预检已通过：

- `runs/20261114_172744_b1bd83/`：1,896帧动态碰撞预检通过。
- 优化相机初始化顺序后 `runs/20261114_172932_265f52/`：1,737帧完整预检通过，未出现此前相机初始化导致的手部订阅积压警告。
- 两次都没有发送运动指令。当前服务保持运行，占用上述两张5090。

之后按用户授权完成一次4步预测前缀的真机测试，运行目录 `runs/20261114_173715_b3e901/`。
执行和清理成功，双臂已退出Servo并停稳，双手40个关节均确认禁用；详见VALIDATION.md。

## B300 历史实例（已停止、资源已释放）

2026-11-14 17:12:45（北京时间）模型加载完成。随后按用户要求停止服务，模型容器已终止。
以下连接地址和任务信息为本次运行记录，不代表服务仍在线。

| 项目 | 当前值 |
|---|---|
| SSH入口 | b300，pengyang@116.198.113.8:10003 |
| 调度任务 | aw-hha12-1114-54a9 |
| 实际任务ID | job-v316m7cezp |
| 项目 / 资源 | xflow，1节点、2张B300 |
| GPU计算节点 | 10.0.46.103:18765；重启后以endpoint.json为准 |
| 模型 | hh_a12 |
| 训练目录 | pt-hh-a12-g64-1114 |
| 权重 | checkpoints/weights/step_008408.pt |
| deployment_id | 34ca15a63ef03aa8 |
| 动作时间约定 | humanhand，32步，1/120～32/120秒 |
| 客户端任务 | flower |
| 编码器 / 策略GPU | cuda:1 / cuda:0 |

远端运行目录：

```text
/mnt/workspace/users/pengyang/workspace/outputs/aw-hha12-1114-54a9/
```

其中 `models.json` 是本次独立注册表：只加载hh_a12、明确固定step_008408，
共享编码器从原注册表的GPU7改为GPU1，以使用两卡资源。未修改wanchi的原模型注册表或环境。
`launch.sh` 调用原run_server.sh并将XFLOW_SERVER_DIR设置为该目录。
客户端config.json已改为自动读取这里的endpoint，并固定上述deployment_id。

```bash
# 从本机检查服务
./run info

# 查看调度状态和日志
ssh b300 'submit --status aw-hha12-1114-54a9'
ssh b300 'submit --logs aw-hha12-1114-54a9 --tail 50'

# 不再使用时停止这个服务并释放资源
ssh b300 'scancel aw-hha12-1114-54a9'
```

服务已按用户要求停止。客户端退出只关闭自己创建的SSH隧道，
不会停止远端服务。若要换权重，需要有意更新客户端的deployment_id，不自动放宽身份校验。

## 已完成的真实模型验证

```bash
./python check_recorded.py --observation validation_data/flower_holdout_000006.npz --n 3
```

输入：`unix_wam/flower/000006`留出episode第40帧，历史为第36帧，真实录制RGB与双手关键点。
不是实时相机，不连接机械臂或手，不包含执行授权。

结果保存在 `runs/real_model_20261114_171251_d91fd0/`：

- 同一TCP会话连续3次真实GPU推理成功，往返约0.794、0.744、0.673秒。
- 返回有限float32 `[32,2,21,3]`，会话、模型身份、时间轴和当前关键点全部一致。
- 完整32步Wuji重定向成功；输出见 `wuji_retarget.npz`。
- 单窗口预测相对录制未来关键点的平均误差12.49mm，保持当前姿态基线5.62mm。
  此窗口模型误差高于保持基线；它只验证通路，不能证明任务效果。

重定向的初始手根/手角度由录制关键点拟合，属于离线诊断初始化，
不是当前机器人实测状态，也没有生成可执行的机械臂计划。
此前真实Aubo SDK及全网格预检使用的是模拟策略；两类验证在VALIDATION.md中分别记录。

尚缺：本地实时相机图像。目前USB枚举未发现深度相机，ROS没有图像和CameraInfo话题。
接好相机并启动原工位驱动后先运行 `./run camera`，通过后运行 `./run run --steps 32` 做完整只读预检。
