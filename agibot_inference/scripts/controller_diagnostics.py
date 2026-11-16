"""Passive fault snapshots and atomic JSON records; never issue robot commands."""
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import uuid

import numpy as np


def _plain(value):
    """Detach supported SDK/numeric values into strict JSON-compatible data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    for names in (('position', 'orientation'), ('translation', 'rotation'),
                  ('linear', 'angular'), ('force', 'torque'),
                  ('x', 'y', 'z', 'w'), ('x', 'y', 'z')):
        if all(hasattr(value, name) for name in names):
            return {name: _plain(getattr(value, name)) for name in names}
    return repr(value)


class ControllerFault(RuntimeError):
    """Same fatal controller check, with the original status preserved."""
    def __init__(self, status, robot):
        super().__init__(f'Controller fault/collision: {status}')
        self.execution = {}
        self.diagnostic = dict(
            detected_at_utc=datetime.now(timezone.utc).isoformat(),
            detected_monotonic_ns=time.monotonic_ns(),
            status={'raw_repr': repr(status)},
            feedback_note='Latest SDK feedback read after fault detection; its source '
                          'timestamp is preserved. This is not the IK solver candidate.')
        for name in ('error_code', 'error_msg', 'mode', 'control_mode',
                     'collision_pairs_1', 'collision_pairs_2', 'frame_names',
                     'frame_poses', 'twists', 'wrenches', 'arm_disturbance_force'):
            try:
                self.diagnostic['status'][name] = _plain(getattr(status, name, None))
            except Exception as exc:
                self.diagnostic['status'][name] = {'read_error': repr(exc)}
        self.diagnostic['joint_feedback_read_started_monotonic_ns'] = time.monotonic_ns()
        try:
            self.diagnostic['joint_feedback'] = _plain(robot.get_joint_states())
        except Exception as exc:
            self.diagnostic['joint_feedback_error'] = repr(exc)
        self.diagnostic['joint_feedback_read_finished_monotonic_ns'] = time.monotonic_ns()


def _emit(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        # A broken stderr must not replace the controller failure either.
        pass


def record_controller_fault(exc, *, context, directory):
    """Record a detected controller fault, without swallowing the caller's error.

    Paired observation expiry is also recorded; other failures are a no-op. Recording failures fall back to stderr;
    the caller always re-raises its original exception and performs cleanup.
    """
    try:
        return _record_controller_fault(exc, context=context, directory=directory)
    except Exception as error:
        # Even an unexpected encoder/snapshot failure must preserve the fault.
        _emit('CONTROLLER_FAULT_RECORD_FAILED ' + type(error).__name__)
        try:
            _emit(repr(error))
            _emit('CONTROLLER_FAULT_RAW ' + repr(exc))
        except Exception:
            pass
        return None


def _record_controller_fault(exc, *, context, directory):
    chain, seen, fault = [], set(), None
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({'type': type(current).__name__, 'message': str(current)})
        if fault is None and isinstance(current, ControllerFault):
            fault = current
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    if fault is None:
        from paired_history import PairedObservationStale
        current = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, PairedObservationStale):
                fault = current
                break
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        if fault is None:
            return None
    payload = dict(schema_version=1, event=('controller_fault' if isinstance(fault, ControllerFault) else 'paired_observation_stale'),
                   diagnostic=fault.diagnostic, execution=fault.execution,
                   exception_chain=chain)
    try:
        payload['context'] = _plain(context)
        payload['execution'] = _plain(fault.execution)
        payload['traceback'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    except Exception as error:
        payload['context'] = {'serialization_error': repr(error)}
        payload['execution'] = repr(fault.execution)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    temporary = None
    try:
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + uuid.uuid4().hex
        destination = directory / ('controller_fault_' + stem + '.json')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.' + stem, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except Exception as error:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        _emit('CONTROLLER_FAULT_RECORD_FAILED ' + repr(error))
        _emit(encoded)
        return None
    _emit('CONTROLLER_FAULT_RECORDED ' + str(destination))
    return destination
