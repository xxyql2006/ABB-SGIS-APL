from typing import OrderedDict
import matplotlib.gridspec as gridspec
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from datetime import datetime
from sswgmm_mech.tools import curve_smoothing, step_function
from sklearn.tree import DecisionTreeRegressor
sns.set(color_codes=True)



class MechOper(object): 
	"""Class of a single mechanical operation.
	   This operation has been identified as open or close and represented in file name
	   This data file has its data label on the 0th column
    """       
	def __init__(self,data_path):
		# read data with file path
		self.data = pd.read_csv(data_path,header=5)
		self.time_step = 400 # in micro-second
		# self.head = 0
		# self.tail = 0
	
	def find_head_tail(angle):
		"""
		@angle: angle sensor reading
		@param head: avg value of the head of the curve
		@param tail: avg value of the tail of the curve
		"""
		head = np.mean(angle[:50])
		tail = np.mean(angle[-50:])
		return head,tail

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

		angle_open, angle_close = np.sort([head, tail]) / 100
		travel = angle_close - angle_open
		return travel, angle_open, angle_close	
	
	def avg_speed(self, curve, op_type, travel, angle_close):
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

		
		# break degree
		# self.break_degree = angle_close - float(config[op_type + 'B'])  # degree
		if op_type == 'C':
			self.break_degree = angle_close - 40  # degree
			pt_0 = self.break_degree - float(0 * travel)  # degree
			pt_1 = self.break_degree - float(0.2 * travel)  # degree
		elif op_type == 'O':
			self.break_degree = angle_close - 34  # degree
			pt_0 = self.break_degree - float(0 * travel)  # degree
			pt_1 = self.break_degree - float(0.3 * travel)  # degree
		
		# print('pt',pt_0,pt_1)
		pt_0_ix = self.find_intersection(curve, pt_0)  # time in index
		pt_1_ix = self.find_intersection(curve, pt_1)  # time in index
		# print('pt_ix',pt_0_ix,pt_1_ix)
		t = (pt_1_ix - pt_0_ix) * self.time_step / 1000  # time in milli-second
		# print('t',t)
		speed = np.abs((pt_1 - pt_0) / t)
		return speed

	# def find_intersection(curve, value):
	# 	return np.argmin(np.abs(curve - value))

	def box_plot(label_list,data_df,color_list):
		# for i in range(len(data_df)):
		f = plt.boxplot(data_df, 
					labels=label_list,
					# color=color_list,
					patch_artist=True)
		plt.tick_params(labelsize=7)
		for box, c in zip(f['boxes'],color_list):
			box.set(color=c,linewidth=2)
			box.set(facecolor=c)
		# plt.legend()

class DataClean():
	def __init__(self) -> None:
		self.data_narray = np.empty(shape=(0,0))


	def plot_travel_mconfig(path, **kwargs):
		"""method that read mechanical data recorded by mconfig;

		Args:
			path          - path of data file

		Return:
			data_narray   - data in numpy array 

		Notes:

		"""
		title = kwargs.get('title','curves')
		plot_color = kwargs.get('plot_color','g')
		# read data
		with open(path, "r", encoding='utf-8') as f: 
			data = f.read()
		wave_list = data.split("WaveID:") # every wave starts with "WaveID"
		# print(wave_list)
		plt.figure(dpi=200)
		plt.title(title)
		plt.xlabel('sampling interval 0.4ms')
		plt.ylabel('current in amper')
		for num, wave_str in enumerate(wave_list):
			if wave_str == '':
				pass
			else:
				start_idx = wave_str.find('Waveform data:') + len('Waveform data:')
				end_idx = wave_str.find('\n')
				wave_list = wave_str[start_idx:end_idx].split(',')
				wave_narray = np.array(wave_list).astype('float64')
				plt.plot(wave_narray, c=plot_color)
				# form the array for all curves
				if num == 1:
					wave_list_narray = wave_narray.reshape(-1,1)
				elif num > 1:
					wave_list_narray = np.concatenate((wave_list_narray, 
													wave_narray.reshape(-1,1)), 
													axis=1)
	
	def filtered_curve(curve, **kwargs):
		lp_N = kwargs.get('lp_N', 8)
		lp_Wn = kwargs.get('lp_Wn', 200)
		time_step = 0.4
		b, a = signal.butter(N=lp_N, 
								Wn=lp_Wn,
								btype='lowpass', 
								output='ba',
								fs=1 / time_step * 1e3)
		_filtered_curve = signal.filtfilt(b, a, curve)
		return _filtered_curve

	def curve_check_thresholds(curve_ndarray, upp_thr, low_thr, **kwargs):
		"""method that check if the curve is invalid;

		Args:
			curve_ndarray	-	input curve data

		Return:
			invalid   		- 	True or False 

		Notes:

		"""
		ratio_thr = kwargs.get('ratio_thr', 0.33)	
		num_invalid = len(curve_ndarray[(curve_ndarray > upp_thr) |	(curve_ndarray < low_thr)])
		# print(num_invalid)
		ratio = num_invalid / len(curve_ndarray)
		if ratio <= ratio_thr:
			valid = True
		else:
			valid = False
			# print('ratio is', ratio)
		return valid

	def mconfig_to_csv(dir_raw, dir_washed):
		"""method that re-orgnize the waveform files of 
			Mconfig into single .csv for each waveform;

		Args:
			dir_raw	-	folder of mconfig waveform files
			dir_washed - folder of washed .csv files

		Return:
			 

		Notes:

		"""
		travel_path_list = []
		wave_class_dict = {'角度(合闸).txt':'travel_close', '角度(分闸).txt':'travel_open', 
                   '合闸电流(合闸).txt':'current_close', '分闸电流(分闸).txt':'current_open', 
                   '储能电流(储能).txt':'current_motor'}
		for cur_dir, dirs, files in os.walk(dir_raw):
			print(cur_dir)
			for wave_class in wave_class_dict.keys():
				if wave_class in files:            
					wave_path = os.path.join(cur_dir, wave_class)
					# print(travel_path)
					travel_path_list.append(wave_path)
					with open(wave_path, "r", encoding='utf-8') as f: 
						data = f.read()
					wave_list = data.split("WaveID:")
					# since the 1st element is blank, wave data splitting from 2nd element
					for wave_str in wave_list[1:]:
						# extract timestamp to be name of waveform file
						time_stamp_start_idx = wave_str.find('Waveform time:') + len('Waveform time:')
						time_stamp_end_idx = wave_str.find(';Sampling frequency(ms)')
						time_stamp_str = wave_str[time_stamp_start_idx : time_stamp_end_idx]
						
						# change format of time stamp into 'XX_XX_XX'
						time_stamp_num = time_stamp_str.replace(' ', '_')
						time_stamp_num = time_stamp_num.replace(':', '_')
						time_stamp_num = time_stamp_num.replace('-', '_')
						
						# extract waveform data
						wave_data_start_idx = wave_str.find('Waveform data:') + len('Waveform data:')
						wave_data_str = wave_str[wave_data_start_idx:]
						wave_data_df = pd.DataFrame({'data':wave_data_str.split(',')})
						
						# export data into .csv file
						wave_file_name = time_stamp_num + '_' + wave_class_dict[wave_class]
						wave_data_df.to_csv(dir_washed + '\\' + wave_file_name + '.csv')
							
							
					else:
						pass
			else:
				pass
		print(travel_path_list)


class MechOperMconfig(object): 
	# """Class of a single mechanical operation recorded by Mconfig 1.5.0"""
	   
	def __init__(self, cur_dir, file, config_path):
		with open(config_path, 'r') as fh:
			self.configuration = json.load(fh)
		self.angle_df = pd.read_csv(os.path.join(cur_dir, file), header=0)
		self.angle_arr = np.array(self.angle_df['data'])
		self.break_deg = 0
		self.time_step = 0.4 # mili second
		self.file_name = file
		# self.coil_current_arr = np.array()
		if 'travel_open' in file:
			self.oper_type = 'O'
		elif 'travel_close' in file:
			self.oper_type = 'C'
		else:
			self.oper_type = 'U'
	
	def cal_travel(self, head, tail):
		"""
		Find the total travel in degree
		@head: avg value of the head of the curve
		@tail: avg value of the tail of the curve
		@return:
		travel: the total travel in degree
		"""
		# head = np.mean(self.angle[:50])
		# tail = np.mean(self.angle[-50:])
		angle_open, angle_close = np.sort([head, tail])
		travel = angle_close - angle_open
		return travel, angle_open, angle_close
	
	@staticmethod
	def find_intersection(curve, value):
		return np.argmin(np.abs(curve - value))
	
	def find_1st_pole(curve):
		diff = np.diff(curve)
		pole_idx = 0
		for idx in range(len(diff) - 1):
			if diff[idx] == 0:
				pole_idx = idx
				break
			elif diff[idx + 1] == 0:
				pole_idx = idx + 1
				break
			elif (diff[idx] * diff[idx + 1]) < 0:	
				pole_idx = idx + 1
				break
			else:
				pass
		return pole_idx



	def cal_speed(self, travel, angle_close, **kwargs):
		"""
        calculate the average speed of the operation
        @param angle_curve: input angle sensor reading
        @param op_type: operation type, 'O', 'C' or 'U'
        @param travel: travel in degree
        @param angle_close: the angle close degree
        @return:
        speed: the average speed of the operation
        """
		time_step = kwargs.get('time_step', 0.4) # mili second
		open_break_angle = kwargs.get('open_break_angle', 11) # degree
		close_break_angle = kwargs.get('close_break_angle', 11) #degree
		open_t1_tb_ratio = kwargs.get('open_t1_tb_per', 0.15) # ratio
		open_t2_tb_ratio = kwargs.get('open_t2_t1_ratio', 0.75) # ratio
		close_t2_tb_ratio = kwargs.get('open_t1_tb_per', 0.15) # ratio 
 
		if self.oper_type == 'O':
			self.break_deg = angle_close - open_break_angle
			pt_0 = self.break_deg - open_t1_tb_ratio * travel
			pt_1 = self.break_deg - open_t2_tb_ratio * travel

		elif self.oper_type == 'C':
			self.break_deg = angle_close - close_break_angle		
			pt_0 = self.break_deg - close_t2_tb_ratio * travel
			pt_1 = self.break_deg

		pt_0_ix = MechOperMconfig.find_intersection(self.angle_arr, pt_0)
		pt_1_ix = MechOperMconfig.find_intersection(self.angle_arr, pt_1)	
		t = (pt_1_ix - pt_0_ix) * time_step
		speed = np.abs((pt_1 - pt_0) / t)
		return speed

	# def angle_start_pt(curve, op_type, low, high):
	# 	"""
	# 	Find where the first angle bend is; semi-obsolete method due to document
	# 	updates
	# 	@param curve: input sensor reading, np array of shape (-1)
	# 	@param op_type: operation type, 'O' for open, 'C' for close or 'U' for unknown
	# 	@param low: lower value of the curve
	# 	@param high: higher of the curve
	# 	@return:
	# 	start_pt: the first angle bend's index
	# 	"""
	# 	x = len(curve)
	# 	# normalize and rescale the curve
	# 	# this op shifts tail away from anchor
	# 	_curve = (curve - low) / (high - low) * x
	# 	if op_type == 'C':
	# 		y = -1000
	# 	elif op_type == 'O':
	# 		y = 100 + x
	# 	else:
	# 		raise ValueError('Incorrect op type {}.'.format(op_type))
	# 	anchor = np.array([x, y]).reshape(1, 2)
	# 	_curve = np.concatenate([np.arange(0, len(curve), 1).reshape(-1, 1), _curve.reshape(-1, 1)], axis=1)
	# 	diff = np.linalg.norm(anchor - _curve, axis=1)
	# 	start_pt = np.argmin(diff)
	# 	return start_pt

	# def cal_overshoot_close(self, angle_close):
	# 	"""method that calculate overshoot for open operation; 

	# 	Args:
	# 		angle_open	-	the angle open degree

	# 	Return:
	# 		overshoot	-	overshoot in degree	
	# 		rebound		-	rebound in degree
			
	# 	Notes:

	# 	"""
	# 	break_deg_idx = MechOperMconfig.find_intersection(self.angle_arr, self.break_deg)
	# 	curve_after_break = self.angle_arr[break_deg_idx:].copy()
	# 	overshoot_idx = MechOperMconfig.find_1st_pole(curve_after_break) + break_deg_idx
	# 	pole_1st_deg = curve_after_break[MechOperMconfig.find_1st_pole(curve_after_break)]
	# 	over_shoot = pole_1st_deg - angle_close
	# 	return over_shoot, overshoot_idx

	def cal_overshoot_close(self, angle_close):
		"""
		calculate the mechanism's closing overshoot
		@param curve: input angle sensor reading, np array of shape (-1)
		@param start_pt: where the first angle bend is, integer
		@param angle_close: the angle close degree
		@return: (_overshoot, _overshoot_ix)
		_overshoot: the amount of overshoot
		_overshoot_ix: where the overshoot occurs on the curve
		"""
		start_pt = MechOperMconfig.find_intersection(self.angle_arr,
													 self.break_deg)
		curve = self.angle_arr.copy()
		after_start = curve[start_pt:]
		_overshoot = np.clip(after_start.max() - angle_close, 0, np.inf)
		overshoot_ix = np.argmax(after_start) + start_pt
		return _overshoot, overshoot_ix

	def cal_rebounce_overshoot_open(self, angle_open):
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
		start_pt = MechOperMconfig.find_intersection(self.angle_arr,
													 self.break_deg)
		curve = self.angle_arr.copy()
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

	def plot_all_csv(dir_data:str, curve_type:str):
		"""method that plot all csv curves under dir_data folder; 

		Args:
			dir_data         -	directory of data file

		Return:
			
		Notes:

		"""
		count_curve = 0
		for cur_dir, dirs, files in os.walk(dir_data):
			
			plt.figure(dpi=200)
			for file in files:

				if curve_type in file:
					count_curve += 1
					curve_df = pd.read_csv(os.path.join(cur_dir, file), header=0)
					plt.plot(curve_df['data'],
							c='g',
							linewidth=0.5)

		plt.title('{} curves, total number: {}'.format(curve_type, count_curve))
		print('number of {} curves: {}'.format(curve_type, count_curve))    


	def plot_every_2k(dir_data:str, curve_type:str, **kwargs):
		"""method that plot curves of every 2k cycles; 
		   only apply to washed data curve originally recorded by Mconfig

		Args:
			dir_data         -	directory of data file
			curve_type		 -	string name of curve type 

		Return:
			
		Notes:

		"""
		fig_2k,ax_2k = plt.subplots()
		fig_4k,ax_4k = plt.subplots()
		fig_6k,ax_6k = plt.subplots()
		fig_8k,ax_8k = plt.subplots()
		fig_10k,ax_10k = plt.subplots()
		date_time_2k = kwargs.get('date_time_2k', datetime(1900, 1, 1))
		date_time_4k = kwargs.get('date_time_4k', datetime(1900, 1, 1))
		date_time_6k = kwargs.get('date_time_6k', datetime(1900, 1, 1))
		date_time_8k = kwargs.get('date_time_8k', datetime(1900, 1, 1))
		date_time_10k = kwargs.get('date_time_10k', datetime(1900, 1, 1))		
		files = os.listdir(dir_data)
		
		# plot open travel curve   
		for file in files:
			if curve_type in file:
				curve_df = pd.read_csv(os.path.join(dir_data, file), header=0)
				file_sep_list = file.split('_')
				time_stamp_str = ''
				for i in range(6):
					time_stamp_str = time_stamp_str + file_sep_list[i] + '_'
				# print('time_stamp_str', time_stamp_str)
				time_stamp_dt = datetime.strptime(time_stamp_str, '%Y_%m_%d_%H_%M_%S_')
				# plot every 2000 cycles
				if ((time_stamp_dt - date_time_2k).days <= 0):
					ax_2k.plot(curve_df['data'],
								c='k',
								linewidth=0.5)
				elif (((time_stamp_dt - date_time_2k).days > 0) 
					& ((time_stamp_dt - date_time_4k).days <= 0)):
					ax_4k.plot(curve_df['data'],
							c='k',
							linewidth=0.5)
				elif (((time_stamp_dt - date_time_4k).days > 0) 
					& ((time_stamp_dt - date_time_6k).days <= 0)):
					ax_6k.plot(curve_df['data'],
							c='k',
							linewidth=0.5) 
				elif (((time_stamp_dt - date_time_6k).days > 0) 
					& ((time_stamp_dt - date_time_8k).days <= 0)):
					ax_8k.plot(curve_df['data'],
							c='k',
							linewidth=0.5) 
				elif (((time_stamp_dt - date_time_8k).days > 0) 
					& ((time_stamp_dt - date_time_10k).days <= 0)):
					ax_10k.plot(curve_df['data'],
								c='k',
								linewidth=0.5)  
		
		ax_2k.set_title('{} curves 0-2k cycles'.format(curve_type))
		ax_4k.set_title('{} curves 2-4k cycles'.format(curve_type))
		ax_6k.set_title('{} curves 4-6k cycles'.format(curve_type))
		ax_8k.set_title('{} curves 6-8k cycles'.format(curve_type))
		ax_10k.set_title('{} curves 8-10k cycles'.format(curve_type))
		# print('the number of valid curves is ', len(valid_close_file_list))
		# print('the number of invalid curves is ', len(invalid_close_file_list))

	def para_dist_plot(data, **kwargs):
		title = kwargs.get('title', 'parameter')
		xlabel = kwargs.get('xlabel', 'test number')
		ylabel = kwargs.get('ylabel', 'parameter value')

		fig = plt.figure(title)
		sns.set(color_codes=True)

		gs2 = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

		axs21 = plt.subplot(gs2[0])
		plt.plot(data, '.g', alpha=0.5)
		plt.plot([-100, 10000], [np.nanmax(data), np.nanmax(data)], '--r')
		plt.plot([-100, 10000], [np.nanmean(data), np.nanmean(data)], '--k')
		plt.plot([-100, 10000], [np.nanmin(data), np.nanmin(data)], '--b')
		plt.text(100, np.nanmax(data) + 0.1, '{:.2f}'.format(np.nanmax(data)), color='r', weight='bold')
		plt.text(100, np.nanmean(data) + 0.1, '{:.2f}'.format(np.nanmean(data)), color='k', weight='bold')
		plt.text(100, np.nanmin(data) + 0.1, '{:.2f}'.format(np.nanmin(data)), color='b', weight='bold')
		axs21.set(title=title, xlabel=xlabel, ylabel=ylabel)

		axs22 = plt.subplot(gs2[1])
		# pddata = pd.Series(data)
		# try:
		sns.distplot(data, bins=50, kde=True, ax=axs22, vertical=True, fit=stats.norm)
		# sns.displot(data, bins=50, kde=True)
		# except Exception:
		# 	print(title, 'calculation error.')
		# 	return
		# axs22.set(ylim=ylim)

		data_std_err = np.nanstd(data, ddof=0)
		data_start = np.nanmean(data[0:20])
		data_end = np.nanmean(data[-20:])
		data_max = np.nanmax(data)
		data_mean = np.nanmean(data)
		data_median = np.nanmedian(data)
		data_min = np.nanmin(data)
		data_span = data_max - data_min
		data_alm_upper = round((data_max - data_start) / data_span * 100, 0)
		data_alm_lower = round((data_start - data_min) / data_span * 100, 0)
		print('{:<18}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}'\
			   .format(title, 
					   data_std_err, 
					   data_start, 
					   data_end, 
					   data_max, 
					   data_mean, 
					   data_median, 
					   data_min, 
					   data_span, 
					   data_alm_upper, 
					   data_alm_lower))
		# plt.show()
		# fig.savefig('./%s/%s/%s-%s%s' % ('_figs', METstCode, METstCode, title, '.png'), dpi=600)


	def current_features(self, curve):
		"""
		created by Wei.Zheng, different from MDC4-M

		Given a current curve, calculate its features
		@param curve: input current sensor reading, np array of shape (-1), in unit mA
		@return:
		start: where the current starts to rise, refer to doc session 3.3.7
		plateau_start: index of where the plateau session starts
		valley: index of where the valley is at; obsolete output
		plateau_end: index of where the plateau session ends
		plateau_rms: the mean value of the plateau session, converted to A
		"""
		
		reg = DecisionTreeRegressor(max_depth=3, max_leaf_nodes=3)
		tmp = curve_smoothing(curve, self.configuration)
		# plt.figure(dpi=300)
		# plt.subplot(2,1,1)
		# plt.plot(curve)
		# plt.subplot(2,1,2)
		# plt.plot(tmp)
		# tmp = curve
		# try:
		steps, pred, is_step_function = step_function(tmp, reg)
		if not is_step_function:
			self.warning_message.append(
				'Poor current signal quality: not a step function.'
			)
		# except:
		# 	print(self.file_name)
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

	def cal_op_time(self, start):
		"""
		calculate the operation time
		@param start: time at where the current first start to rise; refer to doc session 3.3.8
		@return:
		_op_time: the total operation time
		"""
		self.break_deg_ix = self.find_intersection(self.angle_arr, self.break_deg)
		_op_time = (self.break_deg_ix - start) * self.time_step
		_op_time = np.clip(_op_time, 0, np.inf)
		if _op_time <= 0:
			message = ('operation time is less than zero, '
						'start = {}, break = {}'.format(start, self.break_degree))
			self.warning_message.append(message)
		return _op_time