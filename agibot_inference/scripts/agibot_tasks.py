"""AgiBot 推理任务、关节顺序与本机模型目录。"""
from pathlib import Path

AGIBOT_ROOT = Path(__file__).resolve().parents[1]
TASK_EPISODES = {
    'clothes': '0912_20260912111854',
    'guizi': '0914柜子_20260914174235',
    'shuiping': '0912水瓶_20260912132254',
    'dengpao': '0914灯泡_20260914104849',
}
FAMILIES = ('ours', 'fastwam', 'pi05')
SHARPA_LEFT = slice(7, 29)
SHARPA_RIGHT = slice(36, 58)
SHARPA61_EEF_HEAD_INDICES = tuple(range(7)) + tuple(range(29, 36)) + tuple(range(58, 61))
SHARPA_JOINT_NAMES = (
    'thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP',
    'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP',
    'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP', 'middle_DIP',
    'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP',
    'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP',
)
SHARPA_CAMERA_KEYS = ('head_color', 'left_wrist_camera', 'right_wrist_camera')
SHARPA_FEATURE_NAMES = tuple(
    name for side in ('left', 'right') for name in
    ([f'{side}_ee_{axis}' for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
     + [f'{side}_hand_{joint}' for joint in SHARPA_JOINT_NAMES])
) + ('idx11_head_joint1', 'idx12_head_joint2', 'idx13_head_joint3')
SHARPA_TRAIN_CONFIG = 'pi05_base_agibot_sharpa61_full'
SHARPA_CHECKPOINT = AGIBOT_ROOT / 'checkpoints/yifu_three_camera/50000'
SHARPA_DATASET = Path('/data/sharpa/pose_studio/data/lerobot/yifu')


def interface_spec(task: str, model: str) -> dict:
    task = canonical_task(task)
    if task == 'yifu' and model != 'pi05':
        raise ValueError('yifu 61D currently requires pi05')
    result = dict(protocol_version=2, task=task, model=model, fps=20,
                  horizon=50 if task == 'yifu' else 32,
                  action_dim=61 if task == 'yifu' else 19,
                  state_order=list(SHARPA_FEATURE_NAMES if task == 'yifu' else MODEL_JOINT_NAMES))
    if task == 'yifu':
        result['camera_keys'] = list(SHARPA_CAMERA_KEYS)
    return result
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
MODEL_JOINT_NAMES = (
    *(f'idx{21+i}_arm_l_joint{i+1}' for i in range(7)),
    'idx31_gripper_l_inner_joint1',
    *(f'idx{61+i}_arm_r_joint{i+1}' for i in range(7)),
    'idx71_gripper_r_inner_joint1',
    'idx11_head_joint1', 'idx12_head_joint2', 'idx13_head_joint3',
)


def canonical_task(name: str) -> str:
    """训练目录名为 shuiping；命令行兼容用户习惯的 shuibei。"""
    name = {'shuibei': 'shuiping', '晾衣服': 'yifu'}.get(name, name)
    if name not in TASK_EPISODES and name != 'yifu':
        raise ValueError(f'Unknown AgiBot task: {name}; choose {tuple(TASK_EPISODES)}')
    return name


def default_episode(task: str) -> Path:
    task = canonical_task(task)
    if task == 'yifu':
        return SHARPA_DATASET / 'data/chunk-000/episode_000000.parquet'
    return AGIBOT_ROOT / 'data/raw_data' / task / TASK_EPISODES[task]


def default_run_dir(task: str, family: str) -> Path:
    if family not in FAMILIES:
        raise ValueError(f'Unknown model family: {family}')
    task = canonical_task(task)
    if task == 'yifu':
        if family != 'pi05':
            raise ValueError('yifu requires pi05')
        return SHARPA_CHECKPOINT
    if family == 'pi05':
        return Path('/data/agibot_inference/models/pi05') / task / '30000'
    return Path('/data/agibot_inference/models') / family / task
