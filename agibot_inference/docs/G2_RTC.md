# G2 独立 RTC 实验入口

本轮只新增文件。原版 `run_inference.sh`、Client、执行器、部署配置和模型 Server 不修改。
RTC 指本轮实现的异步预取、时间对齐和交接融合；不包含模型采样器内的动作前缀约束。

## 启动与切回

以下均在 real 的 `/home/descfly/code/agibot_inference` 中执行。
本地模型 Server 使用原来的 `bash run_local_server.sh`，不要重复启动已运行的服务。

RTC 实验版（此命令会启动真机）：

```bash
bash run_inference_rtc.sh --model mid_a1_final_local --task yifu --steps 32
```

task 可以换成 `diguan`、`dengpao`、`drink`。首版限现有 aligned-C0 Client 路径，推荐 `--steps 32`。

切回时先 Ctrl+C，等待 RTC 退出、手部连接清理完毕，再运行原命令：

```bash
bash run_inference.sh --model mid_a1_final_local --task yifu --steps 32
```

不需要复制回旧文件，也不要同时运行两个真机 Client。历史图片、当前 proprio、任务裁剪、
instruction、mask、模型权重与 Server 采样器仍使用原配置。

## 新增的运行文件

- `run_inference_rtc.sh`：独立启动脚本，沿用原参数与环境加载方式。
- `scripts/run_mixed76_client_rtc.py`：复制原入口的初始化/复位/手部使能/清理，执行循环改为 RTC。
- `scripts/g2_rtc.py`：后台单请求推理与规划；唯一主线程100Hz下发；按时间切片和100ms融合。
- `configs/g2_rtc.json`：独立 RTC 参数。

原版222个文件的备份与校验清单位于：
`backups/rtc_isolated_20261111T154558/`，包含 `baseline.tar.gz`、`baseline_sha256.json`。

## 本轮行为

1. 保留启动归位、按任务录制首帧复位双手，以及原执行器的一次性平滑进入首目标。
2. 平滑进入后重新观测并预热；只有初始静止等待会重新设置执行起点。运动中的后续预测不重新计时。
3. 当前动作队列未耗尽时后台请求下一块，同时复用原坐标转换与头部逆解。
4. 每个 chunk 用自己的观测锚点转换到 base 系；按绝对时间跳过过时预测，从对应时间位置接续。
5. 交接100ms内融合新旧在同一时刻的目标，使用端点一、二阶导数为零的权重曲线；
   XYZ、手部/腰部标量线性加权，旋转用最短弧SLERP。不是对6D向量或四元数直接相加。
6. 饮品先融合 EEF 与目标腰角，再调用原头部逆解、`reexpress_arm_paths`；每个100Hz子帧使用其目标腰角。
7. 后台完成全块规划，并预约未来交接点；主循环维持同一个100Hz时钟。
   原有100ms执行迟到上限、头部/腰部限位与步长检查、手部SDK检查和饮品跟踪监控继续使用。
   “推理期间静止”的关节对比只用于初始静止阶段；运动中接续的起点来自同一时刻的旧计划，
   不把旧观测当成新的插值起点。连续观测仍使用原 PairedHistory 的新鲜度检查。
8. 不加 EMA 或额外滤波。队列耗尽、规划错过交接、响应过期或原检查失败即退出；不补发陈旧帧、不无限保持旧目标。

## 参数

`configs/g2_rtc.json`：

| 参数 | 默认 | 作用 |
|---|---:|---|
| action_dt_s | 0.05 | 模型动作20Hz，固定 |
| servo_dt_s | 0.01 | 下发100Hz，固定 |
| blend_s | 0.10 | 换块交接融合时长；取整到100Hz子帧 |
| initial_lead_s | 0.75 | 初始提前请求量；后续根据近期耗时向上调整 |
| planning_reserve_s | 0.15 | 收到预测后，给后台规划留出的未来交接余量 |
| margin_s | 0.10 | 自适应提前量中的时间余量 |

新预测仍以原观测时间为基准。规划余量与推理时间都计入过时帧跳过量；不会重新从预测第0帧执行。
不能保证任意推理延迟下都持续运行；预测窗口不足以覆盖耗时时会明确停止。

## 日志与离线验证

正式运行日志：`logs/rtc/<时间>-<PID>.jsonl`。
事件包括 CONFIG、PREFETCH、NEXT_READY、HANDOFF、SERVO_STATS、STOP/COMPLETE。
记录推理/规划耗时、过时帧跳过量、队列余量、发送间隔及迟到量。

- 11项新增测试通过：`tests/test_g2_rtc.py`，包括新旧启动脚本的独立路由。
- 38项原版回归测试通过：`tests/g2_rtc_baseline_regression.json`。
- 四任务共8次真实模型后台推理/规划通过：`tests/g2_rtc_recorded_20261111/ACCEPTANCE.json`。
- 四任务真实模型、录制输入、模拟执行器连续换块通过：`tests/g2_rtc_loop_20261111/ACCEPTANCE.json`。
  平均调度间隔约10ms；实测最大间隔11.6–18.3ms。未调用机器人SDK执行、未发送运动命令。

上述是离线验收；实际GDK/Sharpa的发送时序、机器人动作与任务效果尚未实机验证。
