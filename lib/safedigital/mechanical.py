from typing import OrderedDict
import matplotlib.gridspec as gridspec
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from datetime import datetime
sns.set(color_codes=True)



class MechOper(object): 
	"""Class of a single mechanical operation.
	   This operation has been identified as open or close and represented in file name
	   This data file has its data label on the 0th column
    """       
	def __init__(self,data_path):
		
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

	def find_intersection(self, curve, value):
		return np.argmin(np.abs(curve - value))

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











	# def str_to_datetime(str):
	# 	str_sep_list = str.split('_')
	# 	year_str = str_sep_list[0]
	# 	month_str = str_sep_list[1]
	# 	day_str = str_sep_list[2]
	# 	hour_str = str_sep_list[3]
	# 	min = str_sep_list[4]
	# 	sec = str_sep_list[5]
	# 	date_time = date_time.strp

class MechOperMconfig(): 
	# """Class of a single mechanical operation recorded by Mconfig 1.5.0"""
	   
	# def __init__(self):
	# 	self.angle = self.travel_curve_df['data'].values
	# self.head = np.mean(self.angle[:50])
	# self.tail = np.mean(self.angle[-50:])
	
	def cal_travel(head, tail):
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

	# def cal_travel_speed(self, angle_curve, op_type, travel, angle_close,**kwargs):
	# 	"""
    #     calculate the average speed of the operation
    #     @param angle_curve: input angle sensor reading
    #     @param op_type: operation type, 'O', 'C' or 'U'
    #     @param travel: the total amount of travel
    #     @param angle_close: the angle close degree
    #     @return:
    #     speed: the average speed of the operation
    #     """
	# 	break_deg_dif = kwargs.get('break_deg_dif', )
	# 	break_deg = angle_close - break_deg_dif  # degree
	# 	pt_0 = self.break_degree - config[op_type + '0'] * travel  # degree
	# 	pt_0_ix = self.find_intersection(curve, pt_0)  # time in index
	# 	pt_1 = self.break_degree - config[op_type + '1'] * travel  # degree
	# 	self.break_degree_ix = pt_0_ix  # time in index
	# 	pt_1_ix = self.find_intersection(curve, pt_1)  # time in index
	# 	t = (pt_1_ix - pt_0_ix) * self.time_step  # time in milli-second
	# 	speed = np.abs((pt_1 - pt_0) / t)
	# 	return speed

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


