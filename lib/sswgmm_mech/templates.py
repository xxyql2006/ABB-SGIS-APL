from base_algorithm.base_input import BaseInput
import numpy as np
import json
import os
from pathlib import Path


mech_type_mapping = {
    '0': 'SafeRing 12kV',
    '1': 'SafeRing 24kV',
    '2': 'SafeAir 12kV',
    '3': 'SafeAir 24kV'
}
sub_category_mapping = {
    '0': 'C',
    '1': 'F',
    '2': 'V',
    '3': 'V25'
}
rate_voltage_mapping = {
    '0': '220',
    '1': '220',
    '2': '110',
    '3': '110',
    '4': '60',
    '5': '48',
    '6': '30',
    '7': '24'
}


def sanity_check(mech_type: str, sub_category: str):
    # input sanity check
    with open(os.path.join(Path(__file__).parent, 'para_limits.json'), 'r') as fh:
        param_limits = json.load(fh)
    with open(os.path.join(Path(__file__).parent, 'config.json'), 'r') as fh:
        # equipment related numerical characteristics
        config = json.load(fh)
    excludes = ['DocString', 'deduction', 'deduction_zero', 'fault', 'VI', 'SignalProcessing']
    # mech_type - sub_category pair from param_limits
    param_limits_pairs = {k: list(v.keys()) for (k, v) in param_limits.items() if k not in excludes}
    # mech_type - sub_category pair from config
    config_pairs = {k: list(v.keys()) for (k, v) in config.items() if k not in excludes}
    try:
        first_lvl = param_limits_pairs[mech_type]
        if sub_category not in first_lvl:
            raise ValueError
    except (KeyError, ValueError) as e:
        raise ValueError('parameter limit config not available for {} - {}. Available configs are: {}'.format(
            mech_type, sub_category, param_limits_pairs.__str__()
        ))
    try:
        first_lvl = config_pairs[mech_type]
        if sub_category not in first_lvl:
            raise ValueError
    except (KeyError, ValueError) as e:
        raise ValueError('mech and sub_category: {} - {} combination not supported.'
                         ' Available combinations are: {}'.format(mech_type, sub_category, config_pairs.__str__()))
    return


class Input(BaseInput):
    def __init__(self, json_dict: dict):
        super(Input, self).__init__(json_dict)
        json_dict: dict = json_dict['Parameter']
        self.angle = np.array(json.loads(json_dict['angle']['Value']))
        try:
            self.mech_type = mech_type_mapping[json_dict['mech_type']['Value']]
            self.sub_category = sub_category_mapping[json_dict['sub_category']['Value']]
        except KeyError:
            raise ValueError('Invalid inputs: {} - {}. Available choices are: {}, {}'.format(
                json_dict['mech_type']['Value'], json_dict['sub_category']['Value'],
                mech_type_mapping.__str__(), sub_category_mapping.__str__()
            ))
        if self.sub_category == 'C':
            self.current = np.array([0, 0, 0])  # dummy array, won't be used
        else:
            self.current = np.array(json.loads(json_dict['current']['Value']))
        self.base_values = json.loads(json_dict['base_values']['Value'])
        # unit is in micro-second
        self.time_step = float(json_dict['time_step']['Value'])
        # convert to milli-second
        self.time_step /= 1e3
        self.rate_voltage = rate_voltage_mapping[json_dict['rate_voltage']['Value']]
        sanity_check(self.mech_type, self.sub_category)
