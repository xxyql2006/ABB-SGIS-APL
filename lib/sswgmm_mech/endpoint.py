from base_algorithm import AlgorithmResult, BaseHandler, SignalIndicator
from base_algorithm.base_output import BaseOutput
from sswgmm_mech.mech_monitor import MechMonitor
from sswgmm_mech.templates import Input


class Handler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.mech_monitor = MechMonitor()

    def entry(self):
        """
        function to handle http request
        Receives inputs and calibration data from Performance model,
        then call mech_monitor.py to calculate SSWG mechanical health index.
        Return the health index and messages
        """
        # actual data
        inputs = Input(self.get_inputs())
        outputs = BaseOutput(inputs)
        try:
            algorithm_result = self.mech_monitor.run(
                inputs.angle, inputs.current, inputs.mech_type, inputs.sub_category, inputs.rate_voltage,
                inputs.base_values, inputs.time_step)
        except Exception as e:
            algorithm_result = AlgorithmResult(
                SignalIndicator.algorithm_error.value, SignalIndicator.algorithm_error.value, 1
            )
            algorithm_result.save_error(e)
        outputs.save_result(algorithm_result)
        return outputs
