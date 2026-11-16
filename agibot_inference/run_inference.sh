#!/usr/bin/env bash
# G2 launcher: reuse the existing client and execution path unchanged.
set -e

g2_usage() {
  echo '用法: ./run_inference.sh --model 模型 --task 任务 --steps 帧数'
  echo '示例: ./run_inference.sh --model pi05 --task yifu --steps 32'
  echo 'model: fastwam | alpha (8885) | joint_relative_history (8886) | joint_relative (18868) | joint_relative_c13 (18875) | joint (8887) | tri (18870) | tri_relative (8888) | coarse_fine | pi05 (18872)'
  echo '新增 zw 模型: model_a (8881) | model_b (8882) | model_c (8883) | model_d (8884)；A/B 不支持 drink。'
  echo '新版光学C0: c0_full | c0_lora | c0_lr；先安装对应的真实部署配置。'
  echo '同轮 checkpoint: c0_lr_30k | c0_lr_40k | c0_lr_50k；c0_lr 为本次 Ours pre100K → post30K。'
  echo 'GR00T N1.7: groot (18917)。'
  echo 'Pre+mid+post 30K: mid_a0 | mid_a1 | mid_b1；新篮子/滴管模型 mid_a1_new。'
  echo '本地 mid_a1_final: mid_a1_final_local（先启动 bash run_local_server.sh）。'
  echo '最终版30K: mid_a0_final | mid_a1_final | mid_b1_final。'
  echo '新增30K: mid_a2_final | mid_a1_1w_final | ours_direct_post。'
  echo 'task: diguan | yifu | dengpao | drink（仅 drink 开放腰部旋转，使用已核对的新服务）'
  echo 'steps: 1–32，每次预测后执行的帧数；不是 checkpoint 步数。'
  echo '默认: G2、78维＋validity、远端推理、自动加载环境、执行真机；最多100000轮，Ctrl+C退出。'
  echo 'Joint/Tri/Alpha/Pi0.5/GR00T/FastWAM 连接配置: configs/deployed_MODEL.json；其他模型保持 configs/g2_camera78_remote.json。'
  echo '会沿用原 client 的启动归位、执行及安全检查。'
}
g2_error() { echo "错误: $*" >&2; g2_usage >&2; exit 2; }

g2_model=''
g2_task=''
g2_steps=''
while (( $# )); do
  case "$1" in
    --help|-h) g2_usage; exit 0 ;;
    --model|--task|--steps)
      (( $# >= 2 )) || g2_error "$1 缺少值"
      case "$1" in
        --model) [[ -z "$g2_model" ]] || g2_error '重复 --model'; g2_model="$2" ;;
        --task) [[ -z "$g2_task" ]] || g2_error '重复 --task'; g2_task="$2" ;;
        --steps) [[ -z "$g2_steps" ]] || g2_error '重复 --steps'; g2_steps="$2" ;;
      esac
      shift 2 ;;
    *) g2_error "不支持的参数: $1" ;;
  esac
done
# Profiles select a pinned endpoint; reuse the original c0_lr client/executor.
# 别名只选择对应checkpoint配置，底层客户端和执行逻辑不变。
g2_profile="$g2_model"
case "$g2_profile" in mid_a1_final_local|mid_a2_final|mid_a1_1w_final|ours_direct_post) g2_model=c0_lr ;; esac
case "$g2_profile" in ratio_r75e25|ratio_r50e50|ratio_r25e75|ratio_r0e100|eff_a1_n75|eff_a1_n50|eff_a1_n25|eff_a1_n1|c0_lr_30k|c0_lr_40k|c0_lr_50k|mid_a0|mid_a1|mid_b1|mid_a1_new|mid_a0_final|mid_a1_final|mid_b1_final) g2_model=c0_lr ;; esac
case "$g2_profile" in
  eff_alpha_n75|eff_alpha_n50|eff_alpha_n25|eff_alpha_n1) g2_model=alpha ;;
  eff_fastwam_n75|eff_fastwam_n50|eff_fastwam_n25|eff_fastwam_n1) g2_model=fastwam ;;
esac
case "$g2_model" in fastwam|alpha|joint|joint_relative|joint_relative_c13|joint_relative_history|tri|tri_relative|coarse_fine|pi05|groot|model_a|model_b|model_c|model_d|c0_full|c0_lora|c0_lr) ;; *) g2_error '请指定有效 --model' ;; esac
case "$g2_task" in diguan|yifu|dengpao|drink) ;; *) g2_error '请指定有效的 G2 --task' ;; esac
if [[ "$g2_task" == drink ]]; then
  case "$g2_model" in fastwam|joint_relative|joint_relative_c13|joint_relative_history|joint|tri|tri_relative|alpha|pi05|groot|model_c|model_d|c0_full|c0_lora|c0_lr) ;; *) g2_error 'drink 需要已核对的 Joint / Tri / Alpha / Pi0.5 / GR00T / FastWAM / model_c / model_d 服务' ;; esac
fi
[[ "$g2_steps" =~ ^[0-9]{1,2}$ ]] || g2_error '--steps 必须为 1–32'
g2_steps=$((10#$g2_steps))
(( g2_steps >= 1 && g2_steps <= 32 )) || g2_error '--steps 必须为 1–32'

g2_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
g2_remote_config="$g2_launcher_dir/configs/g2_camera78_remote.json"
case "$g2_model" in fastwam|joint_relative|joint_relative_c13|joint_relative_history|joint|tri|tri_relative|alpha|pi05|groot|model_a|model_b|model_c|model_d|c0_full|c0_lora|c0_lr) g2_remote_config="$g2_launcher_dir/configs/deployed_${g2_model}.json" ;; esac
if [[ "$g2_profile" != "$g2_model" ]]; then
  g2_remote_config="$g2_launcher_dir/configs/deployed_${g2_profile}.json"
fi
case "$g2_profile" in
  ratio_r75e25|ratio_r50e50|ratio_r25e75|ratio_r0e100|eff_a1_n75|eff_a1_n50|eff_a1_n25|eff_a1_n1|eff_alpha_n75|eff_alpha_n50|eff_alpha_n25|eff_alpha_n1|eff_fastwam_n75|eff_fastwam_n50|eff_fastwam_n25|eff_fastwam_n1)
    [[ -f "$g2_remote_config" ]] || g2_error "部署配置尚未安装: ${g2_remote_config}；不会启动机器人" ;;
esac
case "$g2_model" in c0_full|c0_lora|c0_lr)
  [[ -f "$g2_remote_config" ]] || g2_error "部署配置尚未安装: $g2_remote_config；不会启动机器人" ;;
esac
cd "$g2_launcher_dir"
source "$g2_launcher_dir/env.sh" || exit $?
exec python "$g2_launcher_dir/scripts/run_mixed76_client.py" \
  --representation g2-camera78 --model "$g2_model" --embodiment g2 \
  --task "$g2_task" --inference remote \
  --remote-config "$g2_remote_config" \
  --iterations 100000 --steps "$g2_steps" --execute
