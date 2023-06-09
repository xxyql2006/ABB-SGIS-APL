from typing import OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
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
