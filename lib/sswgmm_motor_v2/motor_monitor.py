from base_algorithm.base_output import AlgorithmResult
from sswgmm_motor.motor_monitor import MotorMonitor
from utils import execution_timer
import numpy as np
import gc
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, Union
from math_tools.flood_fill_2d import FloodFill2D
from math_tools.curve_processing_tools import get_nearest_point
from math_tools.stats_tools import percentile_mean
from sswgmm_mech.tools import check_library


class MotorMonitorV2(MotorMonitor):
    def __init__(self):
        super().__init__()
        self.flood_fill = FloodFill2D(50)

    def reset(self):
        gc.collect()
        self.reg = DecisionTreeRegressor(criterion='mae', max_depth=3, max_leaf_nodes=4)
        self.warning_message = []

    def convert_to_steps(self, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert (len(current.shape) == 2) & (current.shape[0] > 1), 'current must have shape (n_ samples, 1)'

        self.reset()
        x = np.arange(0, len(current), 1).reshape(-1, 1)
        weight = max(current) - current + min(current) + 1e-6
        current = current.reshape(-1, 1)
        self.reg.fit(x, current * weight)
        steps = self.reg.predict(x) / weight.reshape(-1)
        return x, steps

    def __input_checker(self, steps: np.ndarray) -> bool:
        thresholds = sorted(np.floor(self.get_reg_thresholds()).astype(int))
        step_0 = np.median(steps[: thresholds[0]])
        step_1 = np.median(steps[thresholds[0]: thresholds[1]])
        step_2 = np.median(steps[thresholds[1]: thresholds[2]])
        step_3 = np.median(steps[thresholds[2]:])
        valid_input = (step_2 > step_1) & (step_1 > step_0) & (step_2 > step_3)
        if not valid_input:
            raise ArithmeticError(
                'The measured current is not a valid current; it does not match the expected curve shape '
                '(up, up, low). Steps are: {}.'.format([step_0, step_1, step_2, step_3])
            )
        return valid_input

    def normalize(self, x: np.ndarray, current: np.ndarray) -> np.ndarray:
        assert len(x) == len(current), 'x and y shape mismatch.'

        # normalize by 0 level
        thresholds = self.get_reg_thresholds()
        baseline = np.mean(current[np.bitwise_or(x < min(thresholds), x > max(thresholds))])
        normalized_current = current - baseline
        # flip sign if peak current is negative
        if np.mean(normalized_current) < 0:
            normalized_current *= -1
        return normalized_current

    def get_reg_thresholds(self):
        assert getattr(self.reg, 'tree_', False), 'DecisionTreeRegressor had not been fitted yet.'

        non_leaf_ix = self.reg.tree_.children_left != -1
        thresholds = self.reg.tree_.threshold[non_leaf_ix]
        thresholds_count = len(thresholds)
        assert thresholds_count == 3, (
            'thresholds must have length == 3, got {} elements in thresholds.'.format(thresholds_count)
        )
        return thresholds

    def find_operation_interval(self, x: np.ndarray, curve: np.ndarray):
        assert (len(x.shape) == 2) & (x.shape[-1] == 1)
        assert (len(curve.shape) == 2) & (curve.shape[-1] == 1)
        assert x.shape == curve.shape

        thresholds = self.get_reg_thresholds()
        # for what is t0 and t2, refer to APL description on motor monitoring
        t0_estimate = np.array([np.array(min(thresholds) + 2), -100]).reshape(1, 2)
        t2_estimate = np.array([max(thresholds) - 2, -100]).reshape(1, 2)
        xy_array = np.concatenate([x, curve], 1)
        t0 = get_nearest_point(t0_estimate, xy_array)
        t2 = get_nearest_point(t2_estimate, xy_array)
        return t0[0], t2[0]

    def validate_curve(self):
        return

    def calc_current_rms(self, current: np.ndarray) -> float:
        thresholds = self.get_reg_thresholds()
        sorted_threshold = sorted(thresholds.astype(int))
        working_current = current[sorted_threshold[-2]: sorted_threshold[-1]]
        current_rms = percentile_mean(working_current, 95)
        return current_rms

    @execution_timer
    def run(self, motor_current: np.ndarray, mech_type: str, sub_category: str, rate_voltage: str, base_value: dict,
            time_step: Union[int, float]) -> AlgorithmResult:
        motor_current = motor_current.reshape(-1, 1)
        x, steps = self.convert_to_steps(motor_current)
        self.__input_checker(steps)
        normalized_motor_current = self.normalize(x, motor_current)
        t0, t2 = self.find_operation_interval(x, normalized_motor_current)
        charging_time = (t2 - t0) * time_step
        charging_current = self.calc_current_rms(normalized_motor_current)
        parameters = {'charging_time': charging_time, 'charging_current': charging_current}
        si, hs, hi, outputs_collection = check_library(
            parameters, self.library, base_value, mech_type, sub_category, rate_voltage
        )
        diagnosis = AlgorithmResult(si, hi, hs, extra_outputs=outputs_collection)
        return diagnosis
