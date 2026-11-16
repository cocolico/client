"""Explicit EEF23 to absolute-joint20 initialization, never training-state resume."""
import torch


def load_qpos_transfer(model, path: str) -> dict:
    if (model.full_action_dim, model.arm_action_dim, model.cam_action_dim) != (20, 14, 6):
        raise ValueError('Q-pose transfer requires 20/14/6 dimensions')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    source = payload['mot']
    target = model.mot.state_dict()
    reset = {f'mixtures.{expert}.{layer}.{field}'
             for expert in ('action', 'arm_action')
             for layer in ('action_encoder', 'head') for field in ('weight', 'bias')}
    if set(source) != set(target) or not reset <= set(source):
        raise ValueError('Unexpected checkpoint keys; refusing partial backbone transfer')
    if source['mixtures.action.action_encoder.weight'].shape[-1] != 7 or source['mixtures.arm_action.action_encoder.weight'].shape[-1] != 16:
        raise ValueError('Expected camera7/arm16 EEF pretrained interfaces')
    if tuple(payload['proprio_encoder']['weight'].shape) != (4096, 23):
        raise ValueError('Expected 23D pretrained proprio encoder')
    initialized = {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        for expert in ('action', 'arm_action'):
            for layer in ('action_encoder', 'head'):
                prefix = f'mixtures.{expert}.{layer}'
                shape = target[prefix + '.weight'].shape
                linear = torch.nn.Linear(shape[1], shape[0])
                for field, value in linear.state_dict().items():
                    initialized[prefix + '.' + field] = value
        proprio = torch.nn.Linear(20, 4096)
    merged = {}
    for key, tensor in target.items():
        if key in reset:
            merged[key] = initialized[key]
        else:
            if source[key].shape != tensor.shape:
                raise ValueError(f'Unexpected backbone mismatch: {key}')
            merged[key] = source[key]
    model.mot.load_state_dict(merged, strict=True)
    model.proprio_encoder.load_state_dict(proprio.state_dict(), strict=True)
    return {'source': path, 'source_step': payload.get('step'),
            'loaded_mot_keys': len(merged) - len(reset),
            'reset_mot_keys': sorted(reset), 'reset_proprio': True,
            'optimizer_loaded': False, 'global_step': 0}
