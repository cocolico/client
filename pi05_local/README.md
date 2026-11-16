# 本地 Pi0.5 humanhand step 8408

独立部署，不修改现有 XFlow 模型服务或机器人默认配置。推理服务只监听127.0.0.1:18805，不连接任何机械臂或灵巧手。

权重来自 `/mnt/workspace/users/wanchi/workspace/experiments/posttrain_baselines_20261114/runs/pi05_posttrain_humanhand/baseline/8408`。
训练源码、OpenPI及tokenizer来自同一远端训练环境。`source_manifest.json`记录来源和原权重SHA256，加载时核对权重摘要。

## 启动推理服务

```bash
cd '/media/descfly/新加卷/ty/replay/client/pi05_local'
./serve
```

首次加载需要等待权重进入GPU。打印READY、生成endpoint.json后可连接。Ctrl+C停止服务；也可在另一个终端运行本目录`.venv/bin/python stop_server.py`，它只向经过命令行校验的本服务PID发送停止信号。

## 模型及协议

- 模型输入：一张RGB图、126维相机坐标状态、训练任务文本；不需要历史图。
- 模型训练顺序：右手21×3，再左手21×3；适配层将客户端的左、右顺序交换后输入，输出再交换回来。
- 严格使用checkpoint里的q01/q99归一化与反归一化，复用训练版本的PIL等比缩放填充224×224、640长度状态文本token，以及相同的缺失腕部相机掩码。
- 输出是相机系、米制的绝对关键点，完整50步、每步1/30秒；不是关节角或位移增量。
- 兼容现有`xflow_kp21_chunk_v1`客户端的info/reset/infer；`model_family=pi05`标识实际模型，服务端未运行XFlow网络。
- 标准`keypoints`字段提供原生预测的前32步，时间为1/30～32/30秒，没有拉伸、抽样或改动空间位置。
- `native_keypoints`保留完整50步，服务记录也保存全部50步和原始126维输出。
- `history_offset_frames=0`，客户端可取当前帧作为history，服务端不使用history字段。
- 任务名支持flower、tidying、lemon、whiteboard，对应训练集完整提示词。

## 接入机器人客户端

独立配置已放在`../aubo_wuji/config.pi05_local.json`，保持默认远端配置不变。
先启动本服务，再在另一个终端使用显式的`--config config.pi05_local.json`。

```bash
cd '/media/descfly/新加卷/ty/replay/client/aubo_wuji'
# 只查询本地模型身份，不连接机器人
./run --config config.pi05_local.json info
# 当前状态/图像到轨迹的只读测试（会连接相机和设备读取反馈）
./run --config config.pi05_local.json run --steps 32
```

本地配置使用原生32/30秒预测时间轴（不套用此前XFlow的人手时间轴修正），当前速度为0.03。
发送方式为 `source_waypoints`：32个原始预测路点加实测起点，不再客户端插值到100Hz。
按原生1/30秒间隔及0.03速度，实际路点间隔约1.111秒，32步约35.56秒。
调度错过一个完整发送周期就顺延后续发送，不连续补发；该处理独立于1000ms告警阈值。
Pi0.5独立客户端配置现将 `execution.maximum_tick_lateness_s` 设置为1.0秒（1000ms）。
沿用 `warn_and_reschedule`：超过该值告警并顺延，不集中补发积压指令；控制器自身的通信超时和命令返回码处理不变。
现有机器人客户端的关节、跟踪、通信及停止逻辑仍由原代码处理；本项目不修改它们。
此前已有的几何碰撞禁用等选项会在独立配置中明确保留，因此推理通路验证不代表任务成功或轨迹已通过碰撞检查。

## 环境

`.venv`基于本机RoboDojo Python3.11创建，继承其中已支持RTX5090的PyTorch2.7.0+cu128；其余Pi0.5依赖安装在独立venv中。
Transformers4.53.2和训练版transformers_replace只覆盖本venv内文件，不修改RoboDojo环境。
JAX使用CPU，PyTorch使用CUDA。requirements.txt列出主要版本；该环境仍依赖本机原有RoboDojo的PyTorch动态库。

## 验证

```bash
../aubo_wuji/python -m unittest test_adapter -v
```

该测试使用真实TCP和现有Client，校验左右手交换、动作形状、32步时间轴、50步保留、模型身份及重复请求拦截，不调用GPU或机器人。
真实GPU验证结果已存于`validation.json`：权重SHA256与远端一致，三次推理约0.12–0.13秒，往返约0.12–0.18秒。

可重复的无硬件推理测试：

```bash
cd '/media/descfly/新加卷/ty/replay/client/pi05_local'
../aubo_wuji/python smoke.py --observation ../aubo_wuji/runs/20261115_145214_489f4a/cycle_0057/observation.npz --n 3
```

当前尚未通过真机轨迹入口的重定向检查：保存观测的预测使左右手根旋转达到约1.42/1.11rad，超过现有0.5rad上限。
正常客户端在这里停止；没有修改限制来强行执行，也没有发送真机指令。应先核对预测与机器人手根姿态/重定向的对应关系。
