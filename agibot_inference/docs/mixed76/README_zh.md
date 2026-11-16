# 六模型通用 mixed76 client

## 已统一的职责

| 层 | 职责 |
|---|---|
| G2/Piper本体适配 | 原生61/23状态转76；执行前76转原生；复用各自现有控制器 |
| 公共client | TCP v4握手、模型身份、reset会话、请求关联、完整动作块及mask验证 |
| 各模型server | 图像预处理、文本、训练统计归一化、模型原生采样、反归一化 |

接口为 `info / reset / infer`，原始头部RGB＋物理76D状态输入，完整32×76动作块输出。20Hz；client决定执行前几步、重新观测。不会用32次单步调用伪造完整预测。

## 实际路径

- G2：`real:/home/descfly/code/agibot_inference/scripts/run_mixed76_client.py`
- G2公共client：同目录 `client_mixed76.py`；复用未修改的 `_ours_protocol.py`、`mixed76_contract.py`、`inference_connection.py`。
- G2配置：`configs/{model}_mixed76_v4.json`。
- G2旧入口备份：`docs/mixed76/v4_backup/`（本地也保存于本目录backup）。
- Piper：`piper:/home/descfly/egoav/src/infer/mixed76_client.py`；公共模块 `mixed76_common/`；配置 `mixed76_configs/`。
- Piper旧 `client.py`、`robot.py`、原HTTP服务和启动脚本保持原样。

模型名固定：fastwam、alpha、joint、tri、coarse_fine、pi05。本体：3b或g2。任务和本体不再隐式绑定。

## 无运动调用

G2：
```bash
cd /home/descfly/code/agibot_inference
~/.local/share/agibot_inference/venvs/inference61/bin/python scripts/run_mixed76_client.py \
  --model pi05 --embodiment g2 --task yifu --observation /tmp/yifu_observation.npz
```

Piper：
```bash
cd /home/descfly/egoav
/home/descfly/.venvs/fastwam-piper-5090/bin/python -m src.infer.mixed76_client \
  --model pi05 --task basket --observation /tmp/mixed76_basket_observation.npz
```

两者均支持 `--remote-config PATH` 指定已验收端点；或者 `--inference local --ip HOST --port PORT` 连接已建隧道。需要真实v4服务，旧v3/WS不自动降级。

`--execute` 才进入原机器人执行路径；`--steps 1..32` 控制执行前缀；默认32。此次未执行机器人动作，未验收闭环控制或任务成功。帧系、单位必须由server报告的训练来源与本体核对，不能仅凭维度相同判定可执行。

## 验证与交接状态

- real与piper各5个unittest测试方法通过；覆盖六模型×两本体12组握手/推理/重置、8类坏响应、错误模型、坏输入、native往返和Piper执行结构。
- 两端实际安装CLI各完成一次记录NPZ→mock TCP v4→32×76结果，无硬件导入/运动。此项不冒充GPU模型验证。
- 四份SERVER文档及PROTOCOL_V4已经通过消息发给四个指定对话，均已开始实施。各server真实权重GPU与网络一致性由对应对话完成。
- 六份配置现已填入实测v4端点；real/piper共12组、24次真实GPU推理通过。详见LIVE_REPORT_zh.md和evidence目录。未执行机器人动作。

协议正文：PROTOCOL_V4.md。分工文档：SERVER_ALPHA.md、SERVER_FASTWAM.md、SERVER_OURS.md、SERVER_PI05.md。
