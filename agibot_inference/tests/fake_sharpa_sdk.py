"""Native SDK boundary double. Never imports a device library or opens sockets."""
from types import SimpleNamespace as NS


def error(code=0, message=''):
    return NS(code=code, message=message)


class Wave:
    def __init__(self):
        self.values = [0.]*22
        self.commands = []
        self.settings = {'speed_coeff': .3, 'current_coeff': .6, 'control_source': 4,
                         'control_mode': 13, 'enable_state': True}
        self.gets = []; self.sets = []; self.starts = 0; self.stops = 0
        self.send_error = error()
    def __getattr__(self, name):
        if name.startswith('get_') and name[4:] in self.settings:
            def get():
                self.gets.append(name)
                return error(), self.settings[name[4:]]
            return get
        if name.startswith('set_') and name[4:] in self.settings:
            def set_value(value):
                self.sets.append((name, value)); self.settings[name[4:]] = value
                return error()
            return set_value
        raise AttributeError(name)
    def start(self): self.starts += 1; return True
    def stop(self): self.stops += 1; return True
    def get_states(self):
        return NS(angles=list(self.values), velocities=[0.]*22, torques=[0.]*22,
                  timestamp=1234, sequence=2, host_recv_time_us=5678)
    def set_joint_position(self, values, interpolate):
        self.commands.append((list(values), interpolate))
        if self.send_error.code == 0: self.values = list(values)
        return self.send_error


class SDK:
    def __init__(self):
        self.HandSide = NS(LEFT=0, RIGHT=1)
        self.DeviceType = NS(HAND=1)
        self.ControlSource = NS(SDK=4)
        self.ControlMode = NS(POSITION=13)
        self.devices = [NS(sn='C8519039C852', hand_side=0, device_type=1),
                        NS(sn='CE569038CE55', hand_side=1, device_type=1)]
        self.waves = {d.sn: Wave() for d in self.devices}
        self.connections = []; self.disconnections = []; self.instances = 0
        self.connected = set()
        self.fail_serial = None
        self.SharpaWaveManager = NS(get_instance=self.get_instance)
    def get_instance(self): self.instances += 1; return self
    def get_all_devices(self): return self.devices
    def connect(self, serial, skip_tactile):
        if serial == self.fail_serial: raise RuntimeError('right unavailable')
        self.connections.append((serial, skip_tactile))
        self.connected.add(serial)
        return self.waves[serial]
    def is_connected(self, serial): return serial in self.connected
    def disconnect(self, serial):
        self.disconnections.append(serial)
        self.connected.discard(serial)
