"""Direct Sharpa native SDK adapter. No litchi_hardware dependency.

Construction/import is inert. Only explicit connect() loads the SDK, discovers,
connects and enables POSITION mode. Do not call it for read-only inspection.
"""
from __future__ import annotations

import ctypes
import importlib
import json
import logging
import math
from pathlib import Path
import sys
import time

LOG = logging.getLogger(__name__)


class SDKError(RuntimeError):
    def __init__(self, operation, error):
        self.code = int(error.code)
        super().__init__(f'{operation}: SDK code={self.code}: {error.message}')


class HandSendError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('SHARPA_SEND_FAILED ' + json.dumps(diagnostic, ensure_ascii=False))


class HandReadError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('SHARPA_READ_FAILED ' + json.dumps(diagnostic, ensure_ascii=False, allow_nan=False))


def _check(operation, error):
    if int(error.code) != 0:
        raise SDKError(operation, error)


def _positions(values, side):
    values = tuple(float(x) for x in values)
    if len(values) != 22 or not all(math.isfinite(x) for x in values):
        raise ValueError(f'{side} hand requires 22 finite qpos values in radians')
    return values


def _load_sdk(root):
    root = Path(root).resolve()
    existing = sys.modules.get('sharpa')
    expected = root / 'python/sharpa'
    if existing is not None and Path(existing.__file__).resolve().parent != expected:
        raise ImportError('Another Sharpa SDK is already loaded; start a clean process')
    ctypes.CDLL(str(root / 'lib/libsharpa-wave-sdk.so'), mode=ctypes.RTLD_GLOBAL)
    path = str(root / 'python')
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module('sharpa')


class SharpaHands:
    def __init__(self, config, *, sdk=None):
        self.config = config
        self._sdk = sdk
        self._manager = None
        self._hands = {}
        self._ready = False
        self.last_send = {}
        self._first_feedback_logged = set()
        self._started_at = {}

    @classmethod
    def from_config(cls, path, *, sdk=None):
        return cls(json.loads(Path(path).read_text()), sdk=sdk)

    def _initialize(self, wave, side):
        cfg = self.config[side]
        desired = {'speed_coeff': cfg['speed_coefficient'],
                   'current_coeff': cfg['current_coefficient'],
                   'control_source': self._sdk.ControlSource.SDK,
                   'control_mode': self._sdk.ControlMode.POSITION,
                   'enable_state': True}
        # Startup only. Avoid redundant writes rejected by some firmware;
        # the send loop never polls settings or waits for physical arrival.
        for name, target in desired.items():
            error, current = getattr(wave, 'get_' + name)()
            _check('get_' + name, error)
            matches = (math.isclose(float(current), target, abs_tol=1e-4, rel_tol=0.)
                       if name.endswith('_coeff') else current == target)
            if not matches:
                _check('set_' + name, getattr(wave, 'set_' + name)(target))
                if name == 'enable_state':
                    time.sleep(.1)  # SDK documents enable may take about 70 ms.
        if not wave.start():
            raise RuntimeError(f'{side}: Sharpa start() returned false')
        self._started_at[side] = time.monotonic()

    def connect(self):
        if self._ready:
            return
        if self._hands:
            raise RuntimeError('Previous Sharpa connection needs close() before reconnect')
        if self._sdk is None:
            self._sdk = _load_sdk(self.config['sdk_root'])
        try:
            self._manager = self._sdk.SharpaWaveManager.get_instance()
            for side in ('left', 'right'):
                serial = self.config[side]['serial']
                if self._manager.is_connected(serial):
                    raise RuntimeError(f'{side}: SDK hand already connected by another owner: {serial}')
            deadline = time.monotonic() + self.config['discovery_timeout_s']
            devices = {}
            while time.monotonic() < deadline:
                devices = {d.sn: d for d in self._manager.get_all_devices()
                           if d.device_type == self._sdk.DeviceType.HAND}
                if all(self.config[side]['serial'] in devices for side in ('left', 'right')):
                    break
                time.sleep(.1)
            for side, hand_side in (('left', self._sdk.HandSide.LEFT), ('right', self._sdk.HandSide.RIGHT)):
                serial = self.config[side]['serial']
                if serial not in devices or devices[serial].hand_side != hand_side:
                    raise RuntimeError(f'{side}: configured Sharpa serial/side not discovered: {serial}')
            for side in ('left', 'right'):
                cfg = self.config[side]
                wave = self._manager.connect(cfg['serial'], not cfg['enable_tactile'])
                self._hands[side] = wave
                self._initialize(wave, side)
            self._ready = True
        except BaseException:
            try:
                self.close()
            except Exception:
                LOG.exception('Sharpa cleanup after failed connection')
            raise

    def require_connected(self):
        if not self._ready:
            raise RuntimeError('Both Sharpa hands must be explicitly connected')

    def read_qpos(self):
        self.require_connected()
        # Native State.angles is actual joint feedback in SDK order, not the
        # last command. No artificial timestamp or extra freshness threshold.
        result = []
        for side in ('left', 'right'):
            started = time.perf_counter()
            state = self._hands[side].get_states()
            raw = tuple(float(x) for x in state.angles)
            info = dict(side=side, serial=self.config[side]['serial'], count=len(raw),
                        nonfinite_indices=[i for i, x in enumerate(raw) if not math.isfinite(x)],
                        qpos=[x if math.isfinite(x) else str(x) for x in raw],
                        timestamp=getattr(state, 'timestamp', None),
                        sequence=getattr(state, 'sequence', None),
                        host_recv_time_us=getattr(state, 'host_recv_time_us', None),
                        since_start_ms=(time.monotonic()-self._started_at[side])*1000,
                        read_ms=(time.perf_counter()-started)*1000)
            try:
                result.append(_positions(raw, side))
            except ValueError as exc:
                error = HandReadError(info)
                LOG.error('%s', error)
                raise error from exc
            if side not in self._first_feedback_logged:
                LOG.info('SHARPA_FIRST_FEEDBACK %s', json.dumps(info, allow_nan=False))
                self._first_feedback_logged.add(side)
        return tuple(result)

    def send_qpos(self, left, right, *, frame_index=None, subframe_index=None):
        self.require_connected()
        targets = {'left': _positions(left, 'left'), 'right': _positions(right, 'right')}
        self.last_send = {}
        for side in ('left', 'right'):
            started = time.perf_counter()
            info = dict(side=side, operation='set_joint_position',
                        frame_index=frame_index, subframe_index=subframe_index)
            try:
                error = self._hands[side].set_joint_position(list(targets[side]), self.config[side]['interpolate'])
                _check('set_joint_position', error)
            except Exception as exc:
                info.update(elapsed_ms=(time.perf_counter()-started)*1000, error=str(exc),
                            sdk_error_code=getattr(exc, 'code', None), target_qpos=list(targets[side]))
                self.last_send[side] = info
                error = HandSendError(info)
                LOG.error('%s', error)
                raise error from exc
            info.update(elapsed_ms=(time.perf_counter()-started)*1000, error=None, sdk_error_code=0)
            self.last_send[side] = info
        return self.last_send

    def close(self):
        self._ready = False
        errors = []
        for side, wave in list(self._hands.items()):
            try:
                # Native stop() stops the receive lifecycle; unlike HAL stop(),
                # this does not explicitly issue set_enable_state(False).
                if not wave.stop():
                    errors.append(f'{side}: stop returned false')
            except Exception as exc:
                errors.append(f'{side}: {exc}')
            try:
                self._manager.disconnect(self.config[side]['serial'])
            except Exception as exc:
                errors.append(f'{side}: disconnect: {exc}')
            else:
                self._hands.pop(side)
        if not self._hands:
            self._manager = None
        if errors:
            raise RuntimeError('Sharpa cleanup failed: ' + '; '.join(errors))
