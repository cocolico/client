# 本机 / b_dev 远端推理

在 real 的 `/home/descfly/code/agibot_inference` 使用现有
`scripts/run_ours_client.py`。默认仍为本机。机器人观测、复位、动作执行、
Sharpa 控制和启动平滑过渡没有改动。

## 本机：原来的命令继续有效

```bash
cd /home/descfly/code/agibot_inference
/home/descfly/.local/share/agibot_inference/venvs/pi05/bin/python \
  scripts/run_ours_client.py --test --task yifu --model pi05
```

可显式加 `--inference local`；`--ip`、`--port` 保持原来的 TCP 端点含义。
离线读取 Parquet 使用上面的现有 pi05 环境；gdk-py310 环境当前没有 pyarrow。
实机仍使用原来的 `source env.sh` 和原来的 client 命令，未修改实机 Python 环境。

## b_dev：连接已配置，模型 server 待用户启动

real 上直接登录：

```bash
ssh b_dev
```

连接目标为 `haoran@116.198.113.21:2323`。专用密钥位于 real 的
`~/.ssh/agibot_b_dev_inference`，私钥没有复制到其他机器。
远端主机指纹经已有已认证 SSH 会话核对，记录在 `~/.ssh/known_hosts_b_dev`。

`configs/inference_b_dev.json`：

```json
{
  "ssh_host": "b_dev",
  "server_host": "127.0.0.1",
  "server_port": null
}
```

等 server 启动后，将 `server_port` 填为它在 b_dev 上的实际 TCP 监听端口。
这里不是 SSH 端口 2323。`server_host` 是从 b_dev 内部访问模型服务的地址。
端口留空时远端模式明确报缺少配置，不猜测端口、不回退到本机。

配置完成后的离线验证命令：

```bash
cd /home/descfly/code/agibot_inference
/home/descfly/.local/share/agibot_inference/venvs/pi05/bin/python \
  scripts/run_ours_client.py --test --task yifu --model pi05 \
  --inference remote --remote-config configs/inference_b_dev.json \
  --iterations 3
```

`--test` 只读取录制数据，不创建机器人或 Sharpa 对象。上述命令在
server_port 尚未填写或模型 server 尚未启动时不能完成推理。

## server 需要兼容的协议

复用本仓库 `server/experiments/agibot/run_pi05_server.py` 的协议：

- 同一个 TCP 连接先请求 `info`，之后连续发送 `infer`。
- 消息为 4 字节大端长度 + UTF-8 JSON；数组按 `_ours_protocol.py`
  使用 dtype、shape 和 Base64 字节传输。
- yifu/Pi0.5：三路 RGB（`head_color`、`left_wrist_camera`、
  `right_wrist_camera`），61 维 state，返回 `(50, 61)` action。
- `info` 的任务、模型、关节顺序、相机名、维度等必须与
  `scripts/agibot_tasks.py::interface_spec` 一致。checkpoint 和归一化配置
  在 server 端加载，client 不拉取 checkpoint。

标准 OpenPI WebSocket/MsgPack server 不与当前 TCP/JSON client 直接兼容。
server 起好后提供监听地址/端口、启动命令、模型任务和 checkpoint 路径，
再做真实模型的录制数据往返验收。

## 生命周期与超时

remote 模式建立独立 SSH master 和动态本地回环转发端口，整个 client 会话
持续复用。正常退出、握手或推理异常时关闭本次转发；不关闭已有 SSH 会话。
SSH 转发建立成功只表明转发就绪，模型是否就绪由原有 client 握手判断。

没有新增启动等待时限、重试次数、自动重连或切换本机的策略。
原有 TCP 连接默认 5 秒、client 响应等待默认 300 秒保持不变。
b_dev 专用 SSH 配置关闭 ServerAlive 心跳超时；其他 SSH 主机配置未改。
如果关闭转发失败，会保留控制路径并报出，便于手工运行
`ssh -S <日志中的控制路径> -O exit b_dev`。

原 client 和 SSH 配置备份位于：
`/home/descfly/code/agibot-bdev-backup-LGmV5K/`。
b_dev 的原授权公钥文件备份为：
`/home/haoran/.ssh/authorized_keys.before-real-bdev-20261018`。

## 本次验证（2026-10-18）

- 离线测试套件 83 项通过，包含原有执行逻辑测试。
- real → b_dev 专用 SSH 登录成功；同一 SSH master 连续完成三次
  对 b_dev 现有 SSH 服务的 TCP 转发验证，退出后端口与控制路径释放。
- 三帧真实 yifu 录制数据经现有 OursClient / TCP 协议完成本地回传验证：
  三路 `(480,640,3)` 图像、61 维 state 逐元素一致，返回 `(50,61)`。
  测试端回传了由 state 构造的动作；这不是模型推理结果。
- 没有启动 b_dev 模型服务，没有加载 checkpoint，没有连接或启动真机。
- b_dev 真实模型往返与延迟尚待 server 启动后验收。
- 仓库已有空 Git object，`git diff --check` 无法完成；本次修改以原文件
  备份和直接 diff 复核，未修复 Git 历史或提交变更。
