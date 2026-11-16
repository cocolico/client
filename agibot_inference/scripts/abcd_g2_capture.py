"""Read paired G2 feedback for the four history models; no hardware startup."""
from functools import wraps
import time


class SerializedSDK:
    """Serialize feedback/native-hand calls without locking an action chunk."""
    _methods = frozenset(('read_qpos', 'send_qpos', 'get_joint_states',
                          'get_motion_control_status', 'get_tf_from_base_link'))

    def __init__(self, target, lock):
        self._target, self._lock = target, lock
        self._closed = False

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if name not in self._methods or not callable(value):
            return value
        @wraps(value)
        def serialized(*args, **kwargs):
            with self._lock:
                if self._closed:
                    raise RuntimeError('SDK adapter is closed')
                return value(*args, **kwargs)
        return serialized

    def close(self):
        # A timed-out producer may later resume after a blocked camera call.
        # Close cannot overlap native hand reads/sends; resumed reads then fail
        # before entering a disconnected native SDK handle.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            return self._target.close()


def g2_history_reader(robot, hands, adapter, *, drink=False, body_names=(), to78=None):
    """One actual feedback assembly supplies RGB, state, anchor, and execution start."""
    if not drink and to78 is None:
        from egoav78_contract import g2_to78
        to78 = g2_to78

    def read():
        observation = robot.get_observation61(hands)
        stamp = time.monotonic_ns()
        robot.capture_progress = dict(stage='coordinate_conversion', started_ns=stamp)
        if drink:
            context = adapter.begin(observation['state'],
                [observation['joints'][name] for name in body_names], sample_ns=stamp)
            state = context.state78
        else:
            context = adapter.begin(observation['state'], sample_ns=stamp)
            state = to78(context.state71)
        robot.capture_progress = dict(stage='observation_assembled', started_ns=time.monotonic_ns())
        image_stamps = dict(robot.last_image_timestamps)
        return dict(sample_ns=stamp, source_id=image_stamps['head_color'],
            state=state, head_rgb=observation['images']['head_color'],
            T_source_camera=context.base_from_camera,
            observation=observation, context=context, image_timestamps_ns=image_stamps)
    read.diagnostics = lambda: dict(getattr(robot, 'capture_progress', {}))
    return read
