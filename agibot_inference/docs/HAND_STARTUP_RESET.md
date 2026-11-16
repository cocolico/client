# 推理启动时 Sharpa 双手复位

2026-10-22：已确认的四条原始录制由 `configs/hand_reset_episodes.json` 固定映射。

| task | raw_data 下的录制目录 |
|---|---|
| drink | chanbing/001_1020铲冰_20261021095912 |
| yifu | yifu/1004晾衣服_20261004114719 |
| diguan | diguan/1003滴管_20261003141221 |
| dengpao | dengpao/episode_000001（原名 1012拧灯泡_20261012103125） |

根路径为 `/home/descfly/code/egoav/pico/data/raw_data`。每侧读取 `record/-sharpa-{left,right}-state.pbdat` 的第一条 `JointState.position`，按关节名重排为 SDK 的 22 维顺序。不是 motor_position、target、模型预测、归零或裁剪后的首帧。左右取各自第一条原始反馈，不做时间插值。

## 启动顺序

1. Server 握手；预先读取并校验两侧手型。缺文件、字段缺失、非法数值或未配置的 task 在任何复位/使能前失败。
2. 原有 G2零点归位保持不变。
3. 连接双手，复用 `replay_feedback.move_hands_to_first`：100 Hz smoothstep，至少 3 秒、峰值不超过 0.3 rad/s，然后保持目标 1 秒。
4. 读取双手实际角度，沿用 replay 的逐关节 0.1 rad（约 5.73°）到位阈值；未到位不进入推理。复位期间保留控制器 fault/collision 检查。
5. 打印 `HAND_RESET_READY` 后，重新获取图像及实际状态，再发送第一次推理请求。

每次启动只复位一次，chunk 间不复位。Ctrl+C、发送失败或到位失败会退出，沿用原入口的手部和连接清理。没有更改动作执行、腰部条件、坐标转换或 IK。

## 入口和日志

当前 `run_inference.sh`（run_mixed76_client）与旧 61 维 Sharpa（run_ours_client）都已接入。旧夹爪流程不变。离线/不带 execute 的路径不会读取复位数据或使能双手。

```bash
cd /home/descfly/code/agibot_inference
bash run_inference.sh --model joint_relative --task drink --steps 32
```

上述命令会启动真机。启动日志应依次出现 G2 归位完成、`HAND_RESET_START`、`HAND_RESET_READY`；发生 `HAND_RESET_FAILED` 时不会开始推理。日志记录来源目录、首消息时间戳/SHA256、左右目标及最大反馈误差。

## 验证与备份

- 新增测试涵盖四任务映射、首条 position 选择/按名排序、缺失/非法源、平滑限速、不到位阻断、两种客户端的启动顺序与清理、离线无复位。
- 使用 fake hardware 与真实录制文件进行离线验证；没有启动真机，实际到位验收待用户运行。
- 备份：`backups/hand-reset-20261022/source-before.tgz`。
- 完整测试的既有错误：test_dynamic_body_replay 的 `test_004_action_replay_preserves_recorded_body_and_head`、`test_004_original_speed_preflight_is_complete`、`test_head_inverse_tracks_body_instead_of_first_frame`、`test_public_base_targets_are_reexpressed_in_moving_arm_base` 缺少外置录制数据；test_mixed76 的 `test_tcp_client_keeps_76d_response` 未传 model 参数。
