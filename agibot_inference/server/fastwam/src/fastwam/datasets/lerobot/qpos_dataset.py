"""RoboTwin dataset exposing auditable split membership and stable cache keys."""
from .robot_video_dataset import RobotVideoDataset


class QposDataset(RobotVideoDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = [
            {'episode_id': f'{part.root}:{int(episode)}'}
            for part in self.lerobot_dataset.multi_dataset._datasets
            for episode in part.episodes
        ]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['qpos_cache_key'] = f'{int(self.lerobot_dataset.is_training_set)}:{index}'
        return sample
