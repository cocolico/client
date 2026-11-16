"""Recorded LeRobot observations only: no robot, DDS, or hand SDK imports."""
import json
from pathlib import Path

import numpy as np
import cv2

from agibot_tasks import SHARPA_CAMERA_KEYS, SHARPA_FEATURE_NAMES


def observation_from_row(row: dict, dataset_root: Path | None = None) -> dict:
    state = np.asarray(row['observation.state'], dtype=np.float32)
    if state.shape != (61,) or not np.isfinite(state).all():
        raise ValueError('Recorded observation.state must be finite [61]')
    images = {}
    for key in SHARPA_CAMERA_KEYS:
        entry = row['observation.images.' + key]
        if entry.get('bytes'):
            encoded = np.frombuffer(entry['bytes'], dtype=np.uint8)
        elif dataset_root is not None and entry.get('path'):
            encoded = np.frombuffer((dataset_root / entry['path']).read_bytes(), dtype=np.uint8)
        else:
            raise ValueError(f'Missing image data: {key}')
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode {key}')
        images[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return {'state': state, 'images': images}


def recorded_observation61(episode: Path, frame: int = 0) -> dict:
    import pyarrow.parquet as parquet
    episode = Path(episode).resolve()
    dataset_root = next((p for p in episode.parents if (p / 'meta/info.json').is_file()), None)
    if dataset_root is None:
        raise ValueError('LeRobot meta/info.json required to verify 61D ordering')
    info = json.loads((dataset_root / 'meta/info.json').read_text())
    for key in ('observation.state', 'action'):
        feature = info['features'][key]
        names = feature['names']
        if isinstance(names, dict):
            names = names.get('joints', names.get('motors'))
        if feature['shape'] != [61] or list(names) != list(SHARPA_FEATURE_NAMES):
            raise ValueError(f'{key} metadata differs from checkpoint 61D ordering')
    columns = ['observation.state'] + ['observation.images.' + k for k in SHARPA_CAMERA_KEYS]
    reader = parquet.ParquetFile(episode)
    if frame < 0 or frame >= reader.metadata.num_rows:
        raise IndexError(f'frame {frame} outside episode of {reader.metadata.num_rows} rows')
    # Read only the row group containing the requested frame.
    offset = frame
    for index in range(reader.num_row_groups):
        rows = reader.metadata.row_group(index).num_rows
        if offset < rows:
            row = reader.read_row_group(index, columns=columns).slice(offset, 1).to_pylist()[0]
            return observation_from_row(row, dataset_root)
        offset -= rows
