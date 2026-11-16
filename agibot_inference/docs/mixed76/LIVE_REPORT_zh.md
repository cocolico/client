# 六模型双主机实连验收（2026-10-18）

全部通过：real/G2衣服记录、piper/篮子记录各连接六模型，每组两次推理（中间reset），共12组、24次实际GPU推理。未导入机器人执行模块、未发送运动命令。两主机使用的client_mixed76.py SHA256一致：8a4d7852eb0efb5100e29fbaafb430f3b8dc322bae5949a9088cb993e4cc75a5。

| 模型 | SSH alias / endpoint | real第二次请求秒 | piper第二次请求秒 |
|---|---|---:|---:|
| FastWAM | b_dev_wc / 127.0.0.1:18849 | 0.908 | 0.907 |
| Alpha | b_dev / 172.31.69.66:8866 | 0.391 | 0.470 |
| Joint | b_dev / 172.31.69.82:8861 | 1.456 | 1.366 |
| Tri | b_dev / 172.31.69.94:8861 | 1.761 | 1.857 |
| Coarse-fine | b_dev / 172.31.69.91:8861 | 2.022 | 2.000 |
| Pi0.5 | b_dev / 172.31.69.80:8858 | 0.244 | 0.308 |

时间包含请求编码、网络和模型推理，不含SSH建连；仅两次请求，且两主机并行访问，不作为严谨速度排名。C2F及部分首次请求超过现有执行默认2秒限制，此次未放宽时限、未运行execute。部分帧系/单位仍由server标为未核实，通信通过不代表真机闭环验收。

## 检查
- info模型、v4、76D、32帧、20Hz、physical absolute、严格权重加载字段通过。
- 十二组deployment_id/checkpoint/normalization_sha256与四方交付info逐项一致。
- 所有动作有限，mask正确，inactive slots为0，请求/session/sample标识匹配。
- reset生成新session后第二次推理成功。
- 两端分别完成76→61、76→23转换，旋转解码无退化。
- 证据：evidence/mixed76_live_{real,piper}_20261018_02/results.json及24份动作npy。
- 首轮SSH路由缺失的失败证据保留于对应_01目录，不覆盖。

## 改动
- 六个configs/*_mixed76_v4.json填入真实端点，并同步real/configs和piper/src/infer/mixed76_configs。
- real的SSH配置新增b_dev_wc；piper新增b_dev和b_dev_wc。原SSH配置备份为~/.ssh/config.before_mixed76_v4。
- Piper生成专用mixed76_inference密钥，仅公钥加入B端授权；real复用既有推理公钥。私钥未传输，主机指纹来自本机既有known_hosts。
- 新增只读推理验收驱动source/live_matrix.py和本报告；公共client/model/server/controller/训练配置均未修改。

## 不执行动作的启动
real:
```bash
cd /home/descfly/code/agibot_inference
~/.local/share/agibot_inference/venvs/inference61/bin/python scripts/run_mixed76_client.py --model alpha --embodiment g2 --task yifu --observation /tmp/yifu_observation.npz
```
piper:
```bash
cd /home/descfly/egoav
/home/descfly/.venvs/fastwam-piper-5090/bin/python -m src.infer.mixed76_client --model alpha --task basket --observation /tmp/mixed76_basket_observation.npz
```
将alpha替换为fastwam/joint/tri/coarse_fine/pi05即可。不要添加--execute。
