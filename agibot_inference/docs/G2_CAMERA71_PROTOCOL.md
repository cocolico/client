# G2 71维联网推理：启动与远端协议

## 状态与改动范围

已有 `Mixed76Client` 和 `run_mixed76_client.py` 增加显式的 `--representation g2-camera71` 模式。原 76 维模式默认不变，不另写网络 client，不自动猜测或降级协议。

71 维模式已接入：观测61 → 头部 FK/冻结相机 anchor → state71 → 原 TCP infer → actions71 → 坐标还原/头部旋转逆解 → 原 `execute_sharpa61`。

连接与模型 reset 沿用原流程；只有用户加 `--execute` 才会运行原 G2 归位、获取实机状态、连接双手并执行。归位前通过同一个 reset helper 去掉 `--execute`，只读获取当前平板“G2零点”的身体控制目标，作为本会话固定的五关节 FK 输入；不默认为零或使用旧录制数据。

保持原有每次32帧/20 Hz、100 Hz插值、启动平滑进入、`--steps`、超时和安全检查。读取三路相机的原观测函数没有修改；网络继续只发送原来的 `head_rgb`。

目前验证来自模拟 server 和假设备接收端，**未运行真机，也未验证用户尚未启动的真实模型 server**。

## 启动（real，由用户执行）

先填写 `configs/g2_camera71_remote.json`：

```json
{
  "ssh_host": "b_dev",
  "server_host": "127.0.0.1",
  "server_port": null
}
```

`ssh_host` 是 real 上可用的 SSH 别名；`server_host` 是从该 SSH 主机看到的推理服务地址；`server_port` 必须改成实际 TCP 端口整数。默认 null 是有意保留的，避免误连旧模型。若服务不在 b_dev，修改这三项，不改客户端代码。

```bash
cd /home/descfly/code/agibot_inference
source env.sh
python scripts/run_mixed76_client.py \
  --representation g2-camera71 \
  --model pi05 --embodiment g2 --task yifu \
  --inference remote --remote-config configs/g2_camera71_remote.json \
  --iterations 100000 --steps 32 --execute
```

`--model` 必须和 server 的 model 一致，可用 pi05/fastwam/alpha/joint/tri/coarse_fine。`--task` 可改 yifu/diguan/dengpao；server 必须提供对应任务的 prompt。Ctrl+C 退出，沿用原清理逻辑。默认观测到返回动作的期限仍为2秒，没有为了模拟测试放宽；慢服务会沿用原超时行为。

如端口可从 real 直接访问，不需要 SSH 隧道，则用 `--inference local --ip 远端IP --port 实际端口` 替换 remote 参数。这里 local 表示直连 TCP，不表示模型必须跑在 real。

只验证握手：去掉 `--execute`（也不加 `--observation`），不会读取机器人配置或连接硬件。

## 离线模拟验证（不启动硬件）

```bash
cd /home/descfly/code/agibot_inference
PYTHONPATH=scripts:tests:/home/descfly/.cache/agibot/app/gdk/lib \
/home/descfly/miniconda3/envs/gdk-py310/bin/python -m unittest test_g2_network71 -v
```

测试自己创建/退出 localhost 模拟 server；使用真实 TCP 编解码与现有 client。即使其中测试模拟了 `--execute` 分支，机器人、归位和 SDK 外部接口也全部是假设备，不会初始化真实 GDK 实例。

录制 NPZ 联调输入：`state: float[61]`（原生 base XYZ+xyzw，头qpos，手qpos）、`head_rgb: uint8[H,W,3]`、`body_target: float[5]`（本录制固定的身体控制目标，rad）。使用 `--observation FILE.npz --output result.npz`，不能加 `--execute`。输出包含 `actions71`、还原的 `actions61`、`base_from_camera`、`diagnostic_json`。此模式不访问实机；不能拿已转相对系的 state71 再传入作原生 state。

## 远端 server 必须遵守的协议

沿用 `scripts/_ours_protocol.py`：4字节大端长度＋UTF-8 JSON；numpy 数组为原有 dtype/shape/base64 封装。单连接、同步请求，顺序 info → reset → infer 多次。协议版本5只用于区分新坐标/字段语义，底层传输没有改。

### info

请求 `{"cmd":"info"}`。返回这些字段：

```json
{
  "protocol_version": 5,
  "model": "pi05",
  "representation": "g2_eef71_chunk_first_camera_rot6d",
  "state_dim": 71,
  "action_dim": 71,
  "state_units": "physical",
  "action_units": "physical",
  "action_mode": "chunk_relative",
  "coordinate_frame": "chunk_first_camera_fixed",
  "rotation_order": "first_two_columns",
  "head_frame": "head_link3",
  "position_units": "meter",
  "joint_units": "radian",
  "horizon": 32,
  "dt_s": 0.05,
  "camera_keys": ["head_rgb"],
  "weights_strict": true,
  "checkpoint": "真实权重路径或标识",
  "normalization_sha256": "实际归一化配置的SHA256",
  "deployment_id": "权重与配置的稳定唯一标识",
  "embodiments": ["g2"],
  "prompts": {"yifu": "实际训练指令"}
}
```

`chunk_relative` 指相对于请求观测时相机坐标系的**位姿目标**，不是逐帧增量；不能积分。单个 chunk 的相机参考系固定。每次请求更新一次，不在执行中重新标定。相机暂以 head_link3 为准，没有额外光心外参。

71维半开切片：0:9左臂EEF，9:31左手qpos，31:40右臂EEF，40:62右手qpos，62:71头部EEF。每个EEF为 XYZ + `[R00,R10,R20,R01,R11,R21]`。所有值都是物理量；server负责训练归一化和动作反归一化。客户端不发身体关节、不做归一化。

### reset

请求 `{"cmd":"reset","request_id":"..."}`。

返回 `protocol_version:5`、相同 request_id/deployment_id、`ok:true`、新非空 session_id。这是清空模型会话，不是机器人归位命令。

### infer

请求字段不变：cmd、request_id、session_id、sample_ns、task、embodiment=`g2`、observation={state: float32[71], head_rgb: uint8[H,W,3]}。sample_ns 是客户端单调时钟标识，不要求 server 和 real 时钟同步；必须原样回传。

回复必须有：protocol_version=5、原 request_id/session_id/sample_ns、相同 deployment_id、embodiment=g2、上述 representation、action_units=physical、dt_s=.05、action_mask=[true]×71、actions=float32[32,71]、非负有限 inference_seconds。

保持原执行时序：观测作为插值起点，每个返回目标占一个50ms步长；首个 chunk 之前另有原有平滑进入阶段。server 不应将返回动作误做逐帧相对位移。数组NaN/Inf、错误shape/身份/协议均沿用客户端检查拒绝，不发给执行器。

## 验证记录与恢复

本次备份 `/home/descfly/code/agibot_inference_backups/g2_network71_before_20261021.tar.gz`。

修改原文件只有 `scripts/client_mixed76.py` 和 `scripts/run_mixed76_client.py`；`robot_wrapper.py`、`sharpa_hands.py`、`run_ours_client.py`、`inference_connection.py`、坐标适配与 replay 数学文件均与备份逐字节一致。

新增测试覆盖：两轮同连接收发、动态更新每请求anchor、head XYZ仅诊断、协议/字段/请求错配、离线CLI、只读G2身体目标、模拟原始执行器100Hz官方protobuf和双手输出。报告日志 `reports/g2_network71_suite_20261021.log`。

全量测试仍保留此前已有的 `test_mixed76.ContractTests.test_tcp_client_keeps_76d_response` 失败：过时v3测试漏传现在必需的model；未改产品逻辑去兼容或掩盖该旧测试。
