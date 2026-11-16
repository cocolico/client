"""Pinned per-task training coordinates; native78 wire/control geometry is unchanged."""
import copy
import numpy as np

CONTRACT = dict(schema='task_training_frame_v1', default='optical_chunk_camera',
    tasks={'g2/diguan': 'optical_base_legacy_labels'},
    requires_source_from_chunk_camera_tasks=['g2/diguan'])


def configure(client, contract):
    """Verify the same explicit contract for every embodiment, including Piper."""
    if contract is not None and contract != CONTRACT:
        raise ValueError('Unsupported task training frame contract')
    if client.info.get('task_training_frame') != contract:
        raise ValueError('Client/server task training frame contract mismatch')
    client.task_training_frame = copy.deepcopy(contract)


def prepare_request(request, contract, history):
    """Attach only the exact paired request sample's physical camera anchor."""
    if contract is None or request.get('cmd') != 'infer':
        return request
    if contract != CONTRACT:
        raise ValueError('Unsupported task training frame contract')
    if request.get('task') not in contract['requires_source_from_chunk_camera_tasks']:
        return request
    anchor = np.asarray(history['source_from_chunk_camera'], dtype=np.float64)
    if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
        raise ValueError('source_from_chunk_camera must be a finite rigid 4x4 anchor')
    rotation = anchor[:3, :3]
    if (not np.allclose(anchor[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4, rtol=0)):
        raise ValueError('source_from_chunk_camera must be a rigid SO(3) anchor')
    return dict(request, source_from_chunk_camera=anchor)
