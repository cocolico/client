# G2 / real 四任务回放

仅在 real 的 `/home/descfly/code/agibot_inference` 使用，不涉及 real_piper。

```bash
cd /home/descfly/code/agibot_inference
bash replay.sh --task yifu    --episode 0 --dry-run
bash replay.sh --task diguan  --episode 0 --dry-run
bash replay.sh --task dengpao --episode 1 --dry-run
bash replay.sh --task drink   --episode 1 --dry-run
```

以上四条已通过本机离线预检。去掉 `--dry-run` 才执行真机：先恢复录制首帧，再按原时间轴回放头、双臂与 Sharpa 双手。只有 drink 开放腰部旋转；逐帧按该帧腰角转换双臂目标，复用原 `/egoav/teleop/retarget` 同步发送。其他任务使用原头臂联合发送接口，腰部只恢复首帧。没有修改底层 GDK/IK、手 SDK、100Hz 回放时钟、首帧恢复及检查机制。

查看任务和编号：

```bash
bash replay.sh --list
bash replay.sh --task diguan --list
```

`--episode` 接受登记编号、原始录制目录名或登记目录的完整路径。映射固定在 `configs/replay_tasks.json`，不会因目录排序变化重新编号。任务与路径不匹配会拒绝。

## 数据来源与编号

| task | 条目数 | 编号与来源 |
|---|---:|---|
| yifu | 110 | 0 保留旧版71维验证包；1–109 对应本机原始衣服录制，源 dataset 编号加1 |
| diguan | 104 | 0–103，按现有 source_episode_mapping.jsonl 对应原始录制 |
| dengpao | 101 | 0–100，按现有 source_episode_mapping.jsonl 对应原始录制 |
| drink | 98 | 0–97，固定登记本机 raw_data/chanbing 下的98条饮品录制 |

除 yifu 0 外，入口回放的是本机原始录制的完整实际反馈，不是云端裁剪、删帧或处理后的78维训练 Parquet。双臂来自 base_link TF；头部/首帧身体与臂关节来自 HAL motor_position；双手来自 Sharpa JointState.position，均按消息源时间戳对齐。没有把 action 当成 state，也没有使用夹爪或手部 Type。

所有413个登记条目的路径和必要文件已检查；并非413条完整轨迹均已通过数值预检。数据缺口或时间戳异常仍会拒绝，不跳过、不自动补造长缺失段。

当前已发现：dengpao 0 左手反馈缺口229.932ms；drink 0 TF最大间隔约289.974ms，drink 3 TF最大间隔约40.204ms。保留原检查，使用上面通过的样本。真实执行前仍会重新读取并预检所选整条轨迹。

离线回归：36项通过，包括原回放测试、硬件模块禁止导入下的数据准备、任务选择和逐帧 drink 腰角/臂坐标往返检查。执行分派使用模拟执行器验证。没有启动真机。报告：`reports/four_task_replay_20261108/`；旧启动脚本：`backups/four-task-replay-20261108/replay.sh`。
