from sswgmm_motor.endpoint import Handler as OldHandler
from sswgmm_motor_v2.motor_monitor import MotorMonitorV2


class Handler(OldHandler):

    def __init__(self):
        super().__init__()
        self.motor_monitor = MotorMonitorV2()
