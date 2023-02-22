# from base_algorithm.base_output import AlgorithmResult, OutputMessage
import numpy as np
from scipy import signal
from validator import angle_prescreen, current_prescreen
from sklearn.tree import DecisionTreeRegressor
import json
import gc
from utils import execution_timer
from pathlib import Path
import os


class MechMonitor(object):
    def __init__(self,
                 config_path=os.path.join(Path(__file__).parent, 'config.json'),
                 library_path=os.path.join(Path(__file__).parent, 'para_limits.json')):
        with open(config_path, 'r') as fh:
            self.configuration = json.load(fh)
        with open(library_path, 'r') as fh:
            self.library = json.load(fh)
        self.time_step = 1  # in micro-second
        self.warning_message = []
        self.reg = DecisionTreeRegressor(max_depth=3, max_leaf_nodes=3)

    def reset(self):
        gc.collect()
        self.warning_message = []
        self.reg = DecisionTreeRegressor(max_depth=3, max_leaf_nodes=3)

    def single_op_type(self, head, tail):
        """
        Given an angle curve, determine its operation type
        @param head: avg value of the head of the curve
        @param tail: avg value of the tail of the curve
        @return:
        op_type: operation type, either 'C' or 'O'
        """
        # TODO: replace hard-coded head and tail with dynamic head and tail
        diff = np.mean(head) - np.mean(tail)
        abs_diff = np.abs(diff)

        if abs_diff <= 10:
            op_type = 'U'  # unidentified op_type
        elif diff < 0:
            op_type = 'C'
        else:
            op_type = 'O'
        return op_type

    def split_curve(self, curve, splits):
        """
        An obsolete method; split a curve into multiple curves if necessary (for multiple ops)
        @param curve: raw angle sensor data, np array of shape (-1,)
        @param splits: split points determined by validator
        @return:
        curves: curves splits from the raw data
        """
        n = len(splits)  # number of splits
        if n == 1:
            curves = [curve]
        else:
            curves = []
            splits = (splits[1:] + splits[:-1]) / 2  # mid-points
            splits = [0] + splits.astype(np.int).tolist() + [len(curve)]
            for i in range(n):
                lower = np.max([0, splits[i] - 100])  # push backward 100 pts
                upper = np.min([splits[i + 1] + 100, len(curve)])  # push forward 100 pts
                curves.append(curve[lower: upper])
        return curves

    def angle_start_pt(self, curve, op_type, low, high):
        """
        Find where the first angle bend is; semi-obsolete method due to document
        updates
        @param curve: input sensor reading, np array of shape (-1)
        @param op_type: operation type, 'O' for open, 'C' for close or 'U' for unknown
        @param low: lower value of the curve
        @param high: higher of the curve
        @return:
        start_pt: the first angle bend's index
        """
        x = len(curve)
        # normalize and rescale the curve
        # this op shifts tail away from anchor
        _curve = (curve - low) / (high - low) * x
        if op_type == 'C':
            y = -1000
        elif op_type == 'O':
            y = 100 + x
        else:
            raise ValueError('Incorrect op type {}.'.format(op_type))
        anchor = np.array([x, y]).reshape(1, 2)
        _curve = np.concatenate([np.arange(0, len(curve), 1).reshape(-1, 1), _curve.reshape(-1, 1)], axis=1)
        diff = np.linalg.norm(anchor - _curve, axis=1)
        start_pt = np.argmin(diff)
        return start_pt

    def travel(self, head, tail):
        """
        Find the total travel in degree
        @param head: avg value of the head of the curve
        @param tail: avg value of the tail of the curve
        @return:
        angle_open: angle at open
        angle_close: angle at close
        travel: the total travel distance in degree
        """
        # TODO: replace hard-coded 100 points with dynamic point selection
        angle_open, angle_close = np.sort([head, tail])
        travel = angle_close - angle_open
        return angle_open, angle_close, travel

    def overshoot_close(self, curve, start_pt, angle_close):
        """
        calculate the mechanism's closing overshoot
        @param curve: input angle sensor reading, np array of shape (-1)
        @param start_pt: where the first angle bend is, integer
        @param angle_close: the angle close degree
        @return: (_overshoot, _overshoot_ix)
        _overshoot: the amount of overshoot
        _overshoot_ix: where the overshoot occurs on the curve
        """
        after_start = curve[start_pt:]
        _overshoot = np.clip(after_start.max() - angle_close, 0, np.inf)
        overshoot_ix = np.argmax(after_start) + start_pt
        return _overshoot, overshoot_ix

    def rebound_overshoot_open(self, curve, start_pt, angle_open):
        """
        For an opening operation, calculate its rebound and overshoot
        @param curve: input angle sensor reading, np array of shape (-1)
        @param start_pt: where the first angle bend is, integer
        @param angle_open: angle_close: the angle open degree
        @return:
        _rebound: the amount of rebound
        rebound_ix: where the rebound occurs on the curve
        _overshoot: the amount of overshoot
        overshoot_ix: where the overshoot occurs on the curve
        """
        after_start = curve[start_pt:]
        cum_min = np.minimum.accumulate(after_start)
        rev_cum_max = np.maximum.accumulate(after_start[::-1])[::-1]
        overshoot_ix = np.argmax(rev_cum_max - cum_min) + start_pt
        # overshoot is defined to be positive, refer to document
        _overshoot = np.clip(angle_open - curve[overshoot_ix], 0, np.inf)
        # calculate rebound
        # find all bumps after start_pt
        partitions = after_start - cum_min
        sessions_split = np.where(partitions == 0)[0].tolist() + [len(curve) - start_pt]
        sessions_split = np.array(sessions_split).astype(np.int)
        integrals = []
        # integrate all areas under bumps
        for i in range(len(sessions_split) - 1):
            ix_0 = sessions_split[i]
            ix_1 = sessions_split[i + 1]
            integrals.append(partitions[ix_0: ix_1].sum())
        # find the biggest bump, assuming all other small bumps are caused by noise
        max_integral_ix = np.argmax(integrals)
        # get rebound index
        rebound_start, rebound_end = sessions_split[[max_integral_ix, max_integral_ix + 1]]
        rebound_ix = np.argmax(after_start[rebound_start: rebound_end]) + start_pt + rebound_start
        _rebound = np.clip(curve[rebound_ix] - angle_open, 0, np.inf)
        return _rebound, rebound_ix, _overshoot, overshoot_ix

    def find_intersection(self, curve, value):
        return np.argmin(np.abs(curve - value))

    def avg_speed(self, curve, mech_type, sub_category,
                  op_type, travel, angle_close):
        """
        calculate the average speed of the operation
        @param curve: input angle sensor reading, np array of shape (-1)
        @param mech_type: mechanical type, a string, i.e. 'SafeRing 12kV'
        @param sub_category: sub-category of a mechanical type, a string, i.e. 'C' or 'V25'
        @param op_type: operation type, 'O', 'C' or 'U'
        @param travel: the total amount of travel
        @param angle_close: the angle close degree
        @return:
        speed: the average speed of the operation
        """
        config = self.configuration[mech_type][sub_category]
        # special case for VI: total travel needs to be trimmed
        if mech_type in self.configuration['VI'].keys():
            if sub_category in self.configuration['VI'][mech_type]:
                travel = travel - float(config[op_type + 'B'])
        # break degree
        self.break_degree = angle_close - float(config[op_type + 'B'])  # degree
        pt_0 = self.break_degree - config[op_type + '0'] * travel  # degree
        pt_0_ix = self.find_intersection(curve, pt_0)  # time in index
        pt_1 = self.break_degree - config[op_type + '1'] * travel  # degree
        self.break_degree_ix = pt_0_ix  # time in index
        pt_1_ix = self.find_intersection(curve, pt_1)  # time in index
        t = (pt_1_ix - pt_0_ix) * self.time_step  # time in milli-second
        speed = np.abs((pt_1 - pt_0) / t)
        return speed

    def current_features(self, curve):
        """
        Given a current curve, calculate its features
        @param curve: input current sensor reading, np array of shape (-1), in unit mA
        @return:
        start: where the current starts to rise, refer to doc session 3.3.7
        plateau_start: index of where the plateau session starts
        valley: index of where the valley is at; obsolete output
        plateau_end: index of where the plateau session ends
        plateau_rms: the mean value of the plateau session, converted to A
        """
        tmp = curve_smoothing(curve, self.configuration)
        steps, pred, is_step_function = step_function(tmp, self.reg)

        if not is_step_function:
            self.warning_message.append(
                'Poor current signal quality: not a step function.'
            )

        # if the current is upside down, flip it
        if (steps[1] < steps[0]) & is_step_function:
            base = steps[0]
            tmp = 2 * base - tmp
            curve = 2 * base - curve

        steps_ix = [np.where(pred == steps[x])[0][-1] for x in [0, 1]]
        # start point
        search_end = steps_ix[0]
        anchor = np.array([search_end + 500, tmp[search_end] - 5000]).reshape(1, 2)
        _curve = np.concatenate(
            [np.arange(0, search_end, 1).reshape(-1, 1),
             curve[:search_end].reshape(-1, 1)], axis=1)
        start = np.argmin(np.linalg.norm(_curve - anchor, axis=1))

        # valley
        search_start = steps_ix[0]
        search_end = steps_ix[1]
        cum_max = np.maximum.accumulate(curve[search_start: search_end])
        flats, cts = np.unique(cum_max, return_counts=True)  # find cum_max plateaus
        flats = flats[np.where(cts >= 3)[0]]  # get rid of small plateau
        flats_ix = [np.where(cum_max == x)[0] for x in flats]  # plateau indices
        # find valley
        valleys = [cum_max[x] - curve[x + search_start] for x in flats_ix]
        valley = 0
        for n, i in enumerate(valleys):
            if i.max() > 0.1:  # unit in Amp
                ix = flats_ix[n]
                valley = search_start + np.argmax(cum_max[ix] - curve[ix + search_start]) + ix[0]
                break

        # plateau start
        if valley != 0:
            search_start = valley
        else:
            search_start = start
        cm = np.cumsum(tmp - steps[0])
        cm = cm / cm.max() * tmp.max()
        mod = tmp - cm
        plateau_start = np.argmax(mod[search_start: search_end]) + search_start
        # plateau end
        search_start = len(curve) - 1 - steps_ix[1]
        search_end = len(curve) - 1 - plateau_start
        curve_rev = tmp[::-1]  # reverse the curve
        cm = np.cumsum(curve_rev - steps[-1])
        cm = cm / cm.max() * curve_rev.max()
        mod_rev = curve_rev - cm
        plateau_end = len(curve) - 1 - np.argmax(mod_rev[search_start: search_end]) - search_start  # plateau end point
        plateau_rms = np.sqrt(np.mean(np.square(curve[plateau_start: plateau_end] - steps[0])))  # in A

        # order of points: start, valley, plateau_start, plateau_end
        if (valley == 0 or valley <= start or
                start >= plateau_start or plateau_end <= plateau_start):
            self.warning_message.append(
                'Poor current quality: signal shape does not meet expectation.'
            )
        return start, plateau_start, valley, plateau_end, plateau_rms

    def op_time(self, start):
        """
        calculate the operation time
        @param start: time at where the current first start to rise; refer to doc session 3.3.8
        @return:
        _op_time: the total operation time
        """
        _op_time = (self.break_degree_ix - start) * self.time_step
        _op_time = np.clip(_op_time, 0, np.inf)
        if _op_time <= 0:
            message = ('operation time is less than zero, '
                       'start = {}, break = {}'.format(start, self.break_degree))
            self.warning_message.append(message)
        return _op_time

    def inherent_op_time(self, valley):
        # obsolete method, ignore
        _op_time = (self.break_ix - valley) * self.time_step
        if _op_time <= 0:
            message = ('inherent operation time is less than zero, '
                       'valley = {}, break = {}'.format(valley, self.break_ix))
            self.warning_message.append(message)
        return np.clip(_op_time, 0, np.inf)

    def filtered_curve(self, curve):
        config = self.configuration['SignalProcessing']
        b, a = signal.butter(N=config['lp_N'], Wn=config['lp_Wn'],
                             btype='lowpass', output='ba',
                             fs=1 / self.time_step * 1e3)
        _filtered_curve = signal.filtfilt(b, a, curve)
        return _filtered_curve

    @execution_timer
    def run(self, angle, current, mech_type, sub_category, rate_voltage: str,
            base_values, time_step) -> AlgorithmResult:
        self.time_step = time_step
        """
        Accepts inputs and calculate SSWG mechanical health information
        @param angle: angle sensor reading, np array of shape (-1,)
        @param current: current sensor reading, np array of shape (-1,)
        @param mech_type: mechanical type, a string, i.e. 'SafeRing 12kV'
        @param rate_voltage: the sswg's rate voltage 
        @param sub_category: sub-category of a mechanical type, a string, i.e. 'C' or 'V25'
        @param base_values: calibration value passed in, dictionary
        @return:
        diagnosis: the result. If mech_type == 'C', it returns SI, HI, HS, travel, average speed and message.
        If mech_type == 'F' or 'V', it returns SI, HI, HS, travel, average speed, overshoot, rebound, operation time,
        coil current and message.
        """
        self.reset()
        diagnosis = {}
        parameters = {}

        # pre-screen angle sensor input
        inlier_validate_angle, curve_validate, _ = angle_prescreen(angle)
        if inlier_validate_angle:
            self.warning_message.append(
                'Poor angle sensor signal: too few inliers.'
            )
        if curve_validate:
            self.warning_message.append(
                'Poor angle sensor signal: too noisy.'
            )

        # filtered travel curve. Does not need this now since MRC cannot display
        # processed data from performance model
        # filtered_curve = self.filtered_curve(angle)
        steps, _, _ = step_function(angle, self.reg)
        head = np.mean(angle[:50])
        tail = np.mean(angle[-50:])
        # check operation type, open or close
        op_type = self.single_op_type(head, tail)
        low, high = sorted([head, tail])
        start_pt = self.angle_start_pt(angle, op_type, low, high)
        # travel
        angle_open, angle_close, travel = self.travel(head, tail)
        # speed
        avg_speed = self.avg_speed(angle, mech_type, sub_category, op_type, travel, angle_close)
        # overshoot or close current
        if op_type == 'C':
            overshoot, _ = self.overshoot_close(angle, start_pt, angle_close)
            current_name = 'close_current'
            speed_name = 'close_spd'
            overshoot_name = 'close_overshoot'
            time_name = 'close_time'
        # rebound or open current
        else:
            parameters['rebound'], _, overshoot, _ = self.rebound_overshoot_open(angle, start_pt, angle_open)
            current_name = 'open_current'
            speed_name = 'open_spd'
            overshoot_name = 'open_overshoot'
            time_name = 'open_time'

        if sub_category != 'C':
            # pre-screen current sensor input
            inlier_validate, noise_validate = current_prescreen(current)
            if inlier_validate:
                self.warning_message.append(
                    'Poor current signal quality: too few inliers.'
                )
            if noise_validate:
                self.warning_message.append(
                    'Poor current signal quality: too noisy.'
                )
            # coil current
            start_pt_c, _, _, _, parameters[current_name] = self.current_features(current)
            # operation time
            parameters[time_name] = self.op_time(start_pt_c)
        # check database
        parameters['travel'] = travel
        parameters[speed_name] = avg_speed
        parameters[overshoot_name] = overshoot
        si, hs, hi, outputs_collection = check_library(
            parameters, self.library, base_values, mech_type, sub_category, rate_voltage
        )
        if op_type == 'C':
            outputs_collection['operation'] = 'close'
        else:
            outputs_collection['operation'] = 'open'

        # diagnosis['filtered_curve'] = filtered_curve
        messages = [OutputMessage(x) for x in self.warning_message]
        diagnosis = AlgorithmResult(si, hi, hs, messages, outputs_collection)
        return diagnosis
