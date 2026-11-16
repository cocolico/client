"""Bounded paired RGB/proprio capture and causal history selection.

The supplied reader owns all hardware access. Every packet must express its
78-D state in its own camera frame and provide that camera's physical anchor
``T_source_camera`` (source_from_camera, e.g. world_from_camera). This module
never reads a robot, predicts proprioception, or normalizes model inputs.
"""
from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
import threading
import time

import numpy as np


def _immutable(array, dtype):
    """Use immutable bytes backing, so consumers cannot re-enable writing."""
    value = np.asarray(array, dtype=dtype)
    return np.frombuffer(value.tobytes(order='C'), dtype=dtype).reshape(value.shape)


class PairedObservationStale(RuntimeError):
    """Passive snapshot of a freshness failure; no hardware reads."""
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.execution = {}
        super().__init__('paired observation is stale: ' + str(diagnostic))


class PairedHistory:
    """Capture at 10 Hz independently of inference and execution waits.

    ``push`` is also a hardware-free ingestion API. Source IDs are increasing
    integer image sequences; the latest repeated ID returns False and never
    refreshes timestamps. All other malformed packets or
    reader failures fail closed. Extra packet values (e.g. execution context)
    pass through by identity; required arrays are detached and immutable.
    """

    def __init__(self, read_packet, period=.1, max_age_s=.3):
        if not callable(read_packet):
            raise TypeError('read_packet must be callable')
        if not np.isfinite(period) or period <= 0:
            raise ValueError('period must be finite and positive')
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self._reader = read_packet
        self._period = float(period)
        self._max_age_ns = int(max_age_s * 1e9)
        self._packets = deque(maxlen=12)
        self._last_source_id = None
        self._first_sample_ns = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._failure = None
        self._started = False

    def push(self, packet):
        """Validate and detach one paired observation; reject duplicate images."""
        if not isinstance(packet, Mapping):
            raise TypeError('packet must be a mapping')
        source_id = packet['source_id']
        if isinstance(source_id, bool) or not isinstance(source_id, (int, np.integer)):
            raise TypeError('source_id must be an integer image sequence')
        source_id = int(source_id)
        with self._condition:
            if source_id == self._last_source_id:
                return False
            if self._last_source_id is not None and source_id < self._last_source_id:
                raise ValueError('source_id image sequence must be strictly increasing')
        stamp = packet['sample_ns']
        if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)):
            raise TypeError('sample_ns must be an integer monotonic timestamp')
        stamp = int(stamp)
        if stamp <= 0 or stamp > time.monotonic_ns():
            raise ValueError('sample_ns must be positive and not in the future')
        state = _immutable(packet['state'], np.float32)
        if state.shape != (78,) or not np.isfinite(state).all():
            raise ValueError('state must be finite float32[78] in its own camera frame')
        for start in (0,9,18):
            columns = state[start+3:start+9].reshape(2,3).T
            if not np.allclose(columns.T @ columns, np.eye(2), atol=1e-3, rtol=0):
                raise ValueError('state pose rotation columns must represent SO(3)')
        if not np.allclose(state[18:27], [0,0,0,1,0,0,0,1,0], atol=1e-5, rtol=0):
            raise ValueError('input head pose must be identity in its own camera frame')
        rgb = np.asarray(packet['head_rgb'])
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or not 32 <= rgb.shape[0] <= 480 or not 32 <= rgb.shape[1] <= 640):
            raise ValueError('head_rgb must be uint8 HWC RGB, at least 32x32 and at most 640x480')
        rgb = _immutable(rgb, np.uint8)
        anchor = _immutable(packet['T_source_camera'], np.float64)
        if anchor.shape != (4,4) or not np.isfinite(anchor).all():
            raise ValueError('T_source_camera must be a finite rigid 4x4 anchor')
        rotation = anchor[:3,:3]
        if (not np.allclose(anchor[3], [0,0,0,1], atol=1e-6, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4, rtol=0)):
            raise ValueError('T_source_camera rotation must belong to SO(3)')
        stored = dict(packet)
        stored.update(sample_ns=stamp, source_id=source_id, state=state, head_rgb=rgb,
                      T_source_camera=anchor)
        stored = MappingProxyType(stored)
        with self._condition:
            # Ingestion may race when called directly while producer is running.
            if source_id == self._last_source_id:
                return False
            if self._last_source_id is not None and source_id < self._last_source_id:
                raise ValueError('source_id image sequence must be strictly increasing')
            if self._packets and stamp <= self._packets[-1]['sample_ns']:
                raise ValueError('sample_ns must be strictly monotonic')
            if self._first_sample_ns is None:
                self._first_sample_ns = stamp
            self._last_source_id = source_id
            self._packets.append(stored)
            self._condition.notify_all()
        return True

    def _check_locked(self):
        if self._failure is not None:
            raise RuntimeError('paired capture failed') from self._failure
        if not self._packets:
            raise RuntimeError('paired capture has no fresh observation')
        self._require_fresh(self._packets[-1])

    def _require_fresh(self, packet):
        now = time.monotonic_ns()
        age_ns = now - packet['sample_ns']
        if age_ns > self._max_age_ns:
            detail = dict(age_s=age_ns / 1e9, max_age_s=self._max_age_ns / 1e9,
                          sample_ns=packet['sample_ns'], source_id=packet['source_id'],
                          detected_monotonic_ns=now)
            snapshot = getattr(self._reader, 'diagnostics', None)
            if callable(snapshot):
                try:
                    detail['capture'] = snapshot()
                except Exception as error:
                    detail['capture_diagnostic_error'] = repr(error)
            raise PairedObservationStale(detail)

    def check(self):
        """Execution monitor: check producer health and its newest observation."""
        with self._condition:
            self._check_locked()

    def snapshot(self):
        """Return the exact newest stored packet without a second hardware read."""
        with self._condition:
            self._check_locked()
            return self._packets[-1]

    def prepare(self, sample_ns):
        """Select paired frames for the exact stored current observation.

        Targets are -0.8,-0.6,-0.4,-0.2,0 seconds. Past targets allow 0.1 s
        nearest-frame error and cannot reuse an image. Only targets preceding
        the very first acquisition may repeat that earliest frame as padding.
        All pose triplets become expressed in the *current* camera anchor.
        """
        with self._condition:
            self._check_locked()
            packets = list(self._packets)
            first_ns = self._first_sample_ns
        current = next((p for p in packets if p['sample_ns'] == sample_ns), None)
        if current is None:
            raise KeyError('requested current observation is not in paired history')
        self._require_fresh(current)
        current_ns = current['sample_ns']
        past = [p for p in packets if p['sample_ns'] < current_ns]
        selected, padded, errors = [], [], []
        for offset_ns in (-800_000_000,-600_000_000,-400_000_000,-200_000_000):
            target = current_ns + offset_ns
            is_pad = target < first_ns
            if is_pad:
                earliest = packets[0]
                if earliest['sample_ns'] != first_ns:
                    raise RuntimeError('initial padding observation was evicted')
                chosen = earliest
            else:
                if not past:
                    raise RuntimeError('missing historical observation')
                chosen = min(past, key=lambda p: (abs(p['sample_ns']-target), p['sample_ns']))
                if abs(chosen['sample_ns'] - target) > 100_000_000:
                    raise RuntimeError('historical observation exceeds 0.1s timestamp tolerance')
                if any(not pad and p['source_id'] == chosen['source_id']
                       for p,pad in zip(selected,padded)):
                    raise RuntimeError('historical observations must be distinct')
            selected.append(chosen)
            padded.append(is_pad)
            errors.append((chosen['sample_ns'] - target) / 1e9)
        if current_ns - first_ns >= 1_000_000_000 and any(padded):
            raise RuntimeError('four distinct historical observations required after warmup')
        selected.append(current); padded.append(False); errors.append(0.)
        states = []
        current_anchor = current['T_source_camera']
        # Rigid inverse, avoiding numerical drift from general matrix inversion.
        inverse = np.eye(4)
        inverse[:3,:3] = current_anchor[:3,:3].T
        inverse[:3,3] = -current_anchor[:3,:3].T @ current_anchor[:3,3]
        for chosen in selected:
            state = np.array(chosen['state'], copy=True)
            if chosen is not current:
                relative = inverse @ chosen['T_source_camera']
                for start in (0,9,18):
                    state[start:start+3] = relative[:3,:3] @ state[start:start+3] + relative[:3,3]
                    # Rotation6D is [first column, second column], not rows.
                    columns = state[start+3:start+9].reshape(2,3).T
                    state[start+3:start+9] = (relative[:3,:3] @ columns).T.reshape(6)
            states.append(state)
        return dict(state=current['state'], head_rgb=current['head_rgb'],
                    source_from_chunk_camera=current['T_source_camera'],
                    state_history=np.stack(states),
                    head_rgb_history=np.stack([p['head_rgb'] for p in selected]),
                    history_is_pad=np.array(padded, dtype=bool),
                    history_offsets=[-16,-12,-8,-4,0],
                    valid_history_frames=sum(not p for p in padded[:4]),
                    selected_sample_ns=[p['sample_ns'] for p in selected],
                    timestamp_errors_s=errors, padding=padded)

    def _produce(self):
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                self.push(self._reader())
                deadline = max(deadline + self._period, time.monotonic())
                self._stop.wait(max(0., deadline - time.monotonic()))
        except Exception as error:
            with self._condition:
                self._failure = error
                self._condition.notify_all()
            self._stop.set()

    def __enter__(self):
        if self._started:
            raise RuntimeError('paired capture context cannot be started twice')
        self._started = True
        self._thread = threading.Thread(target=self._produce, name='paired-history', daemon=True)
        self._thread.start()
        try:
            deadline = time.monotonic() + 2.
            with self._condition:
                while not self._packets and self._failure is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError('paired capture did not produce a first frame within 2s')
                    self._condition.wait(remaining)
                self._check_locked()
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        """Stop and join the producer; a blocked reader is an explicit failure."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.)
            if self._thread.is_alive():
                raise RuntimeError('paired capture thread failed to stop within 5s')

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
