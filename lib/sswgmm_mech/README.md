Secondary switchgear mechanical diagnosis algorithm  
Original Author: Eric Wang @ eric-zhixiang.wang@cn.abb.com  
Code Review: Wei Zheng @ wei-wei.zheng@cn.abb.com  
Performance model integration: Wei Zheng @ wei-wei.zheng@cn.abb.com; Andy Xia @ andy-yuanxiang.xia@cn.abb.com

# 1.1 Overview
The algorithms will these SSWGs, `SafeRing C`, `SafeRing F`, `SafeRing V`, `SafeRing V20\V25`,  `SafeAir C` and `SafeAir V`.  
As of 07/13/2020, SafeRing F and V use the same algorithm, but different parameters. SafeRing C and Safe Air C have their unique algorithms. There will be different algorithms, although they share some common functions.  

# 1.1 Code files 
The code will be broken down into the following modules:  
`./old_config`: original code and config from *Eric*  
`./sswgmm_mech/mech_monitor.py`: mechanical monitoring algorithm main body  
`./sswgmm_mech/endpoint.py`: API exposed to performance model  
`./sswgmm_mech/validator.py`: code for validating data quality  
`./sswgmm_mech/config.json`: algorithm configs  
`./sswgmm_mech/para_limits.json`: algorithm parameters  
`./sswgmm_mech/utils.py`: utility tools for scoring  
`./sswgmm_mech_validator.py`: input data validator  
`./sswgmm_motor/motor_monitor.py`: motor monitoring algorithm main body  
`./sswgmm_motor/endpoint.py`: API exposed to performance model  
`./data`: test data for C, F, V and motor  
`./config.py`: web service entry configurator  
`./main.py`: web service main body  
`./test*`: test scripts


# 2. Deployment
Deploy using Azure Container Instance or docker in VM.

# 3. Communication format

## mechanical monitoring API 

from MRC backend to PM

```json
{
    "TravelCurveData": "1,2,3,4,...",
    "CoilCurrentData": "1,2,3,4,...",
    "MECType": 0,
    "UnitType"0,
    "RatedVoltage": 12000,
    "Frequency": 10000,
    "base_travel": 1.0,
    "base_open_spd": 1.0,
    "base_close_spd": 1.0,
    "base_rebound": 1.0,
    "base_open_overshoot": 1.0,
    "base_close_overshoot": 1.0
}
```



from PM to python web service

```json
{
    "angle": "1,2,3,4,...",
    "current": "1,2,3,4,...",
    "mech_type": "some_string",
    "sub_category": "some_string",
    "base_values": {
        "travel": 1.0,
        "open_spd": 1.0,
        "close_spd": 1.0,
        "rebound": 0,
        "open_overshoot": 1.0,
        "close_overshoot": 0
    }
    "time_step": 0.2
}
```

Mapping from `MRC to PM` to `PM to python web service`

| `MRC to PM`          | `PM to python web service` | conversion                  |
| -------------------- | -------------------------- | --------------------------- |
| TravelCurveData      | angle                      | equal                       |
| CoilCurrentData      | current                    | current unit is mA          |
| MECType              | mech_type                  | mapping dict                |
| UnitType             | sub_category               | mapping dict                |
| RatedVoltage         | None                       | kV                          |
| Frequency            | time_step                  | time_step is in millisecond |
| base_travel          | travel                     | equal                       |
| base_open_spd        | open_spd                   | equal                       |
| base_close_spd       | close_spd                  | equal                       |
| base_rebound         | rebound                    | equal                       |
| base_open_overshoot  | open_overshoot             | equal                       |
| base_close_overshoot | close_overshoot            | equal                       |



## motor monitoring API 

### from MRC backend to PM

```json
{
    "MECType": 0,
    "UnitType": 0,
    "RatedVoltage": 12000,
    "MotorCurrentData": "1,2,3,4,...",
    "base_charging_time": 1.0,
    "base_charging_current": 1.0,
    "Frequency": 10000
}
```



### from PM to python web service

```json
{
    "mech_type": "some_string",
    "sub_category": "some_string"
    "motor_current": "1,2,3,4,...",
    "charging_time": 1.0,
    "charging_current": 1.0,
    "time_step": 0.02
}
```

Mapping from `MRC to PM` to `PM to python web service`

| `MRC to PM`           | `PM to python web service` | conversion             |
| --------------------- | -------------------------- | ---------------------- |
| MotorCurrentData      | motor_current              | current unit is mA     |
| base_charging_time    | charging_time              | time in second         |
| base_charging_current | charging_current           | current unit is mA     |
| Frequency             | time_step                  | time_step is in second |
| MECType               | mech_type                  | mapping dict           |
| UnitType              | sub_category               | mapping dict           |
| RatedVoltage          | None                       | kV                     |

### Mapping dict

```python
# MECType: mech_type
{
    0: "SafeRing"
}
# mech_type is the combination of MECType and RatedVoltage, e.g. if MECType = 0, RatedVoltage = 12000, then mech_type is
mech_type = "SafeRing 12kV"
# MECType = 0, RatedVoltage = 24000, then mech_type is
mech_type = "SafeRing 24kV"
```



```python
# UnitType: sub_category
{
	0: "C",
	1: "F",
	2: "V",
	3: "V20/V25",
}
```

## Output format

### mechanical monitoring output returned from PM

```python
{
    "SignalIndicator": 1,
    "HealthIndex": 12,
    "HealthScore": 0,
    "message": "some warning message about the input signal quality",
    "travel": 1,
    # some of the following key-value pairs will be missing, depending on the operation type
    "open_spd": 2,
    "close_spd": 2,
    "open_time": 3,
    "close_time": 3,
    "coil_current": 4,
    "rebound": 7,
    "open_overshoot": 8,
    "close_overshoot": 9
}
```

## motor monitor output returned from PM

```python
{
    "SignalIndicator": 1,
    "HealthIndex": 12,
    "HealthScore": 0,
    "charging_time": 1,
    "chargint_current": 1
}
```

### Error Code

The following dictionary shows error with which parameters:

```json
{
    "travel": 1,
    "open_spd": 2,
    "close_spd": 2,
    "open_time": 3,
    "close_time": 3,
    "coil_current": 4,
    "charging_current": 5,
    "charging_time": 6,
    "rebound": 7,
    "open_overshoot": 8,
    "close_overshoot": 8
}
```

For example, if `"HealthIndex" = 123` and the operation is *open*, it means `travel`, `open_spd` and `open_time` have warning or alarm.



