# Mixed76 六模型推理操作手册

## 本项目部署值（内部使用）

先在本地执行ssh real或ssh piper进入对应主机。以下变量只在对应主机设置，不设置HOME或PATH。

real/G2：
```bash
G2_ROOT=/home/descfly/code/agibot_inference
G2_PYTHON=/home/descfly/.local/share/agibot_inference/venvs/inference61/bin/python
G2_OBS=/tmp/yifu_observation.npz
```
piper：
```bash
PIPER_ROOT=/home/descfly/egoav
PIPER_PYTHON=/home/descfly/.venvs/fastwam-piper-5090/bin/python
PIPER_OBS=/tmp/mixed76_basket_observation.npz
```
/tmp中的文件可能在重启后消失，使用前确认；本地存档在realrobot_mixed_20261017/pi05_server的basket_observation.npz/yifu_observation.npz。

| 模型 | SSH别名 | server_host:port | 权重步数 | 采样快照 |
|---|---|---|---:|---|
| fastwam | b_dev_wc | 127.0.0.1:18849 | 30000 | BF16/eager/20步 |
| alpha | b_dev | 172.31.69.66:8866 | 30000 | compile/10步 |
| joint | b_dev | 172.31.69.82:8861 | 20000 | BF16/eager/20步 |
| tri | b_dev | 172.31.69.94:8861 | 20000 | BF16/eager/20步 |
| coarse_fine | b_dev | 172.31.69.91:8861 | 10000 | BF16/eager/20步 |
| pi05 | b_dev | 172.31.69.80:8858 | 10000 | eager/10步 |

地址仅为2026-10-18实测快照，不是永久服务名。配置读取路径见下文。公共client两端哈希一致，实测结果详见同目录LIVE_REPORT_zh.md；全部原始info、请求时间、动作npy位于evidence/mixed76_live_{real,piper}_20261018_02。
同目录PROTOCOL_V4.md是协议正文；SERVER_ALPHA/FASTWAM/OURS/PI05.md是四方server实施分工。源代码快照位于source/。本文件记录内部路径，不作为脱敏分享包内容。

状态日期：2026-10-18。本文是可移植操作合同，路径和endpoint由实际部署确定。运行前核对安装版help，不把历史快照当实时状态。

## 1. 模型与任务

| 模型 | model |
|---|---|
| FastWAM | fastwam |
| OpenWAM Alpha | alpha |
| Ours Joint | joint |
| Ours Tri（三路DiT） | tri |
| Ours Coarse-fine | coarse_fine |
| Pi0.5 | pi05 |

G2 embodiment=g2：yifu衣服、dengpao灯泡、diguan滴管；Piper embodiment=3b：basket篮子、book放书。任务语义应与现场和记录数据对应。混训服务可声明两本体，但不能任意把一个本体任务部署到另一个本体。任务prompt取自server的训练合同，不在client临时翻译或改写。

## 2. 部署变量和目录

由操作者确认这些变量，不在共享包写死内部地址：
- G2_ROOT：G2仓库根，入口scripts/run_mixed76_client.py。
- PIPER_ROOT：Piper仓库根，入口python -m src.infer.mixed76_client。
- G2_PYTHON / PIPER_PYTHON：安装numpy、scipy和对应硬件依赖的解释器。
- G2_OBS / PIPER_OBS：匹配任务的记录NPZ，键state和head_rgb。
- G2配置位于G2_ROOT/configs；Piper配置位于PIPER_ROOT/src/infer/mixed76_configs。

每模型配置文件名：{model}_mixed76_v4.json。例（替换部署变量后才可用）：
```json
{"model":"alpha","protocol_version":4,"ssh_host":"YOUR_SSH_ALIAS","server_host":"YOUR_SERVER_HOST","server_port":8866}
```
ssh_host是客户端主机上的别名，不是操作者笔记本上的别名。server_host相对于SSH目标主机解释；127.0.0.1并不指客户端本身。常规连接由inference_connection.py建立/关闭本次会话的SSH隧道，使用已核实host key。认证失败先查路由、公钥授权；不复制他人私钥、不关闭主机校验。

## 3. 三种运行模式

| 模式 | 当前是否支持 | 行为 |
|---|---|---|
| 记录NPZ推理 | 是 | 不读硬件、不发动作 |
| 实时相机/状态，只推理 | 尚未实现 | 不提供假命令；需隔离实现 |
| 实时观测并执行 | 代码路径存在、未完成本轮运动验收 | 初始化/归位后读取反馈、推理、执行 |

G2去掉observation但不加execute，只做握手后退出；Piper此组合会报参数错误。两者都不等于实时只读。原Piper旧HTTP client的dry-run参数不能套给新的TCP client。

## 4. 无运动启动命令

变量先按实际部署设置。以下函数只简化命令，不会自行启动。
```bash
cd "$G2_ROOT"
g2infer() { "$G2_PYTHON" scripts/run_mixed76_client.py --embodiment g2 "$@"; }
g2infer --model fastwam --task yifu --observation "$G2_OBS"
g2infer --model alpha --task yifu --observation "$G2_OBS"
g2infer --model joint --task yifu --observation "$G2_OBS"
g2infer --model tri --task yifu --observation "$G2_OBS"
g2infer --model coarse_fine --task yifu --observation "$G2_OBS"
g2infer --model pi05 --task yifu --observation "$G2_OBS"
```
六行是六种选择，不要求连续运行。Piper：
```bash
cd "$PIPER_ROOT"
piperinfer() { "$PIPER_PYTHON" -m src.infer.mixed76_client "$@"; }
piperinfer --model fastwam --task basket --observation "$PIPER_OBS"
piperinfer --model alpha --task basket --observation "$PIPER_OBS"
piperinfer --model joint --task basket --observation "$PIPER_OBS"
piperinfer --model tri --task basket --observation "$PIPER_OBS"
piperinfer --model coarse_fine --task basket --observation "$PIPER_OBS"
piperinfer --model pi05 --task basket --observation "$PIPER_OBS"
```
用--output FILE.npy保存76D动作。--remote-config PATH覆盖默认端点配置。已有本地转发可用--inference local --ip HOST --port PORT；不要同时传remote-config。

切模型：退出当前client，再改model启动，新连接建立新session。切任务：改task并更换对应记录/现场场景，server使用info.prompts中的指令。不需要仅为切client重启server。server重建导致IP变化时重新取证并更新配置。

## 5. 正式执行参数（不是已验收运动指令）

只有用户明确要求运动且第6节检查通过后，才运行以下模式：
```bash
g2infer --model alpha --task yifu --execute --iterations 1 --steps 1 --max-response-age 2
piperinfer --model alpha --task basket --execute --cycles 1 --steps 1 --max-response-age 2
```
G2 iterations、Piper cycles均正整数；steps为1..32，每轮仅执行完整预测的前N帧，再重新观测。32帧、20Hz对应1.6秒预测跨度；循环总时间还包含推理和Piper IK，不是连续20Hz闭环刷新。
G2执行前会startup_reset，Piper会RobotExecutor.start/归位；steps1并不限制归位运动，也不保证预测第一步位移小。不要添加observation和execute同时使用。

## 6. 正式运动前检查

1. 身份：info中的模型、权重步数/路径、deployment_id、normalization_sha256、采样配置对应本次实验。
2. 语义：训练与控制器的左右顺序、TCP定义、基座坐标系、绝对/相对目标、位置/角度/夹爪单位一致；rot6D为前两列，原生四元数xyzw。Piper控制接口位置米、夹爪毫米；G2头部/手指与训练单位逐项核对，不能仅凭physical字段通过。
3. 观测：实时qPOS/EEF/手部反馈有效且新鲜；相机方向、RGB顺序、预处理符合训练；确认head_rgb是哪路相机。Piper现有SourceHub可能仍需多相机在线，网络只传头部不代表硬件采集已改为单相机。
4. 路径：审查输出目标与当前反馈的跳变、限位、IK可达性、碰撞和夹爪范围；检查初始归位全过程。
5. 时延：记录端到端年龄。G2在发送动作前检查，Piper还包括IK；不能把server单独模型时间当完整时限。当前默认2秒，C2F部分请求已超限；需先评估/改进，不能直接放宽掩盖陈旧输入。
6. 现场：控制器无故障、机器人静止/模式正确、独占控制、操作员在场、清空工作区并确认急停有效。Ctrl+C不是硬件急停替代；中断后确认实际停机状态。
7. 逐级验证：记录回放→实时只读（需先实现）→受控短前缀运动→增加轮数，分别留下证据。有限数组不证明动作质量或任务成功。

## 7. 复验和证据

调用打包脚本：
```bash
python scripts/live_matrix.py --client-dir INSTALLED_COMMON_CLIENT_DIR --configs CONFIG_DIR --observation MATCHING_RECORD.npz --embodiment g2 --output NEW_RESULT_DIR
```
此批量脚本固定g2使用yifu、3b使用basket，故MATCHING_RECORD.npz必须分别为衣服/篮子记录；它没有--task，不可用book/diguan/dengpao记录。其他任务使用第4节公共CLI逐模型测试。在Piper运行改embodiment=3b。client目录必须含client_mixed76.py、mixed76_contract.py、_ours_protocol.py、inference_connection.py。脚本依赖numpy/scipy及系统ssh。输出目录必须不存在，防止覆盖证据。
每模型info/reset/infer两次，中间reset；检查形状、finite、mask、请求关联，解码原生动作，保存results.json及动作npy。退出0只表示所有请求通过；仍需将info身份与预期部署逐项比较。脚本不验证闭环控制、不改变配置、不自动修复服务。
2026-10-18历史验收：两主机×六模型=12组，24次真实GPU请求通过；不同本体输入，非统一benchmark。共享client当时SHA256为8a4d7852eb0efb5100e29fbaafb430f3b8dc322bae5949a9088cb993e4cc75a5。重新测试以新证据为准。

## 8. 常见故障

| 现象 | 排查/处理 |
|---|---|
| Could not resolve hostname b_dev | 在运行client的主机查ssh别名，不改模型 |
| Permission denied/host key失败 | 核对账号、公钥授权、可信指纹；不复制私钥或忽略校验 |
| Connection refused/timeout | 查真实server监听、job及podIP，确认没有指向旧端点 |
| protocol/model/checkpoint mismatch | 停止当前请求，核对v4和部署身份，不删检查 |
| stale session/request mismatch | 丢弃动作，reset或新建连接；不重放旧命令 |
| shape/mask/NaN/退化旋转 | 保留请求和输出证据，交模型owner诊断，不补零伪造正常动作 |
| expired | 检查预热、网络、模型、IK及观测年龄，不盲加timeout |
| 正常返回但动作异常 | 查坐标、归一化、单位、左右映射；不要试运动找答案 |

## 9. 维护与共享

本skill是操作知识和无运动验收工具，不包含训练权重、数据、部署凭据或内部IP。外部用户需先部署兼容v4的server和本体client。详见protocol_v4.md；server模型逻辑归模型owner，公共client归协调维护者。文档更新不能顺带重启训练或驱动机器人。
