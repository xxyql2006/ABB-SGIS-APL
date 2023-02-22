# from base_algorithm import SignalIndicator
import numpy as np
from scipy import signal
import json
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, Dict, Any
from pathlib import Path
import os


with open(os.path.join(Path(__file__).parent, 'config.json'), 'r') as fh:
    config = json.load(fh)['SignalProcessing']


def step_function(curve, reg: DecisionTreeRegressor):
    """
    Use a decision tree regressor to fit an impulse function to the curve
    @param curve: a curve, sensor reading. np array of shape (-1)
    @param reg: the decision tree regressor
    @return:
    steps: index where the step occurs
    pred: the impulse function
    is_step_function: whether the curve can be approximated with a step function
    """
    x = np.arange(0, len(curve), 1).reshape(-1, 1)
    reg.fit(x, curve.reshape(-1, 1))
    pred = reg.predict(x)
    is_step_function = True
    if len(np.unique(pred)) != 3:
        is_step_function = False
    ix = np.where(pred != pred[0])[0][0]
    steps = [pred[0], pred[ix], pred[-1]]
    upward_step = (steps[1] > steps[0]) & (steps[1] > steps[2])
    downward_step = (steps[1] < steps[0]) & (steps[1] < steps[2])
    if not (upward_step or downward_step):
        is_step_function = False
    return steps, pred, is_step_function


def curve_smoothing(curve, _config):
    """
    smooth the curve using Savgol filter
    @param curve: sensor reading, np array of shape (-1)
    @param _config: the config dict that stores smoothing parameters
    @return:
    tmp: smoothed curve
    """
    window = _config['SignalProcessing']['savgol_window']
    p_order = _config['SignalProcessing']['savgol_order']
    curve_smooth = signal.savgol_filter(curve, window, p_order)  # heavy smoothing
    if (np.abs(curve - curve_smooth) >= (curve.max() - curve.min()) / 4).any():
        tmp = curve_smooth
    else:
        tmp = curve
    return tmp


def check_library(diagnosis: dict, library: dict, base_values: dict,
                  mech_type: str, sub_category: str, rate_voltage: str) -> Tuple[int, int, int, dict]:
    outputs_collection: Dict[str, Dict[str, Any]] = {}
    try:
        target_library = library[mech_type][sub_category]
    except KeyError:
        raise ValueError('invalid input types: {} - {}'.format(mech_type, sub_category))

    voltage_specific_properties = [
        'open_current', 'close_current', 'charging_time', 'charging_current'
    ]
    for k, v in diagnosis.items():
        try:
            if k in voltage_specific_properties:
                limit = ParaLimit(base_values[k], target_library[k][rate_voltage]['span'],
                                  target_library[k][rate_voltage]['hard_range'],
                                  target_library[k][rate_voltage]['alarm_percent'],
                                  target_library[k][rate_voltage]['warn_percent'])
            else:
                limit = ParaLimit(base_values[k], target_library[k]['span'], target_library[k]['hard_range'],
                                  target_library[k]['alarm_percent'], target_library[k]['warn_percent'])
        except KeyError as e:
            raise ValueError('input mismatch on {}'.format(k))
        status = limit.chk_status(v)
        warning, alarm = (0, 0)
        if status == SignalIndicator.warning.value:
            warning = 1
        elif status == SignalIndicator.alarm.value:
            alarm = 1
        outputs_collection[k] = {
            'Value': round(v, 2),
            'upper_warning_limit': round(limit.warning.upper, 4),
            'lower_warning_limit': round(limit.warning.lower, 4),
            'upper_alarm_limit': round(limit.alarm.upper, 4),
            'lower_alarm_limit': round(limit.alarm.lower, 4),
            'warning': warning,
            'alarm': alarm,
        }

    deduction = 0
    deduction_lut = library['deduction']
    for k, v in outputs_collection.items():
        if k in deduction_lut.keys():
            if v['alarm'] == 1:
                deduction = 100  # alarm
                # deduction = 0  # for debug only
                break
            elif v['warning'] == 1:
                deduction += deduction_lut[k]

    si, hs, hi = calculate_score(deduction)
    return si, hs, hi, outputs_collection


def parse_params(inputs):
    mech_type = inputs['mech_type']
    sub_category = inputs['sub_category']
    base_values = json.loads(str(inputs['base_values']))
    time_step = float(inputs['time_step'])
    return mech_type, sub_category, base_values, time_step


def calculate_score(deduction):
    score = np.clip(100 - deduction, 0, 100)
    if score == 100:
        si = SignalIndicator.normal.value
        hs = score
    elif score >= 60:
        si = SignalIndicator.warning.value
        hs = 75
    else:
        si = SignalIndicator.alarm.value
        hs = 59
    hi = si
    return int(si), int(hs), int(hi)


class Interval(object):
    """Basic class of [lower, upper] limit range
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.range = [self.lower, self.upper]

    def __contains__(self, item):
        """Alow the usage of "in" expression.
            e.g.
            >>> x = Interval(0, 10)
            >>> print(2 in x)
            True
        """
        return self.lower <= item <= self.upper


class ParaLimit(object):
    """
    Class of a single parameter thresholds.
    """

    def __init__(self, base=0, span=1, hard_range=(float('-inf'), float('inf')), alarm_percent=(50, 50),
                 warn_percent=(10, 10)):
        """
        Init the warning and alarm upper/lower limits of a parameter.
        Compute the warning/alarm limits based on input parameters. By default, the warning
        range is upper and lower 10% of the normal range.
        @param base: base line value of a parameter, normally this is the value of a new mech.
        @param span: the value of normal range (from upper alarm to low alarm limit)
        @param hard_range: a pair of values which a parameter should never exceed
        @param alarm_percent: percentage of span of alarm limits above and below the base
        """
        assert hard_range[1] - hard_range[0] >= span, 'hard range must be larger than span!'
        # initial alarm range
        self.alarm = Interval(
            base - alarm_percent[0] / 100 * span,
            base + alarm_percent[1] / 100 * span)

        hard_range = Interval(hard_range[0], hard_range[1])

        overlap = self.overlap(self.alarm.range, hard_range.range)
        if overlap == 0:
            self.alarm = Interval(0, 0)
        else:
            lower_bound = max(self.alarm.lower, hard_range.lower)
            self.alarm = Interval(lower_bound, lower_bound + overlap)

        # set the warning range, 10% of normal range
        self.warning = Interval(
            self.alarm.lower + warn_percent[0] / 100 * overlap,
            self.alarm.upper - warn_percent[1] / 100 * overlap)

    def chk_status(self, para) -> int:
        """Check the parameter status.
        Return:
        -------
            0: normal
            1: warning
            2: alarm
        """
        if para in self.warning:  # normal
            return SignalIndicator.normal.value
        elif para in self.alarm:  # warning
            return SignalIndicator.warning.value
        return SignalIndicator.alarm.value  # alarm

    @staticmethod
    def overlap(alarm, hard_range):
        return max(0, min(alarm[1], hard_range[1]) - max(alarm[0], hard_range[0]))
