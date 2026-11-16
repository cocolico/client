# Sharpa Wave SE 独立位置控制

当前本体左右各一只 Sharpa Wave SE 手。头、双臂仍由 GDK 管理，但本接口只控制手部，不连接 GDK，也不调用旧 19 维模型。每只手的 SDK schema 是 `sharpa_wave_22_v1`，位置单位 rad。`bin/agibot hand schema` 输出 22 个关节名称、顺序和上下界；`read` 的 JSON 也按名称输出当前位置。

| 操作 | 命令 | 对真机的影响 |
|---|---|---|
| 查看 schema | `bin/agibot hand schema` | 不连接硬件 |
| 读左手 | `bin/agibot hand read --side left` | 连接/使能左手并读反馈，不发送位置目标 |
| 读双手 | `bin/agibot hand read --side both` | 依次连接/使能并读取左右手 |
| 模拟移动 | `bin/agibot hand move --side left --joint index_MCP_FE --delta-rad 0.03 --fake` | 只用 SDK 模拟手 |
| 真机单关节移动 | `bin/agibot hand move --side left --joint index_MCP_FE --delta-rad 0.03 --execute` | 向左手发送 22 维 POSITION 目标 |

位置命令先读取当前 22 维反馈，校验 schema、数值和目标关节限位；指定关节相对当前角度改变 `--delta-rad`，其他 21 维保持读取值。单次增量绝对值不能超过 0.1 rad。默认等待 3 秒、到位容差 0.005 rad，可用 `--timeout` 和 `--tolerance` 调整；增量必须大于容差两倍，以免旧反馈误判成功。SDK 的 POSITION 命令可能异步执行，因此脚本发命令后等待新反馈；超时即失败。程序结束时调用 SDK `stop()` 和 `disconnect()`。

每次 `move` 是单手、单关节的一次命令，且必须显式 `--execute`。连接手部本身会切到 POSITION 并使能，因此读取也要在工作空间安全时进行。SDK 环境位于 `/home/descfly/code/sharpa/.venv`，其中 `litchi-hardware==0.1.8`；统一入口先清除 GDK 的 `LD_LIBRARY_PATH` 和 `PYTHONPATH`，再启动该环境的 Python。若自动选择设备有歧义，单手命令可加 `--serial`。

旧模型状态是左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1 + 头部 3，共 19 维；当前手部是左 22 + 右 22，不能用补零或复制夹爪标量代替训练。接入新模型前，需要确定包含头/双臂和双手的观测、动作顺序、绝对/增量语义、采样率与安全限位，然后另行训练和联调。

已验证：SDK 模拟手可读双手并到达左手 `index_MCP_FE +0.03 rad`；真机左右手读取均返回正确 22 维 schema。首次左手 `+0.03 rad` 测试因旧容差等于位移而误报到位，独立复读证明未移动。修正判定后，左手 `+0.08 rad` 从 `0.17642` 到反馈 `0.25154 rad`（目标 `0.25642`，误差 `0.00488`），右手 `+0.08 rad` 从 `0.27880` 到反馈 `0.35398 rad`（目标 `0.35880`，误差 `0.00482`）；两次均在默认 `0.005 rad` 容差内。断开后复读分别为 `0.25154` 和 `0.35388 rad`。这验证了左右手单关节位置命令和反馈，没有验证整手轨迹或新模型闭环。
