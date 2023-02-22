import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from safedigital import temperature as TR
from datetime import datetime
from sklearn import linear_model
import json
from scipy import interpolate 
# import math

class GasPresExperiment_CN(object):
	
	def data_clean(cur_dir, test_date, test_id, **kwargs):
		sample_time = kwargs.get('sample_time', '1min')
		test_folder = test_date + test_id  # test folder
		folder_data_raw = cur_dir + '\\' + test_folder + '\\' + '0_Data original'  # raw data folder
		# folder_data_clean = cur_dir + '\\' + test_folder + '\\' + '1_Data formatted'  # cleaned data folder
		file_name_sen = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TP.csv'  # sensor data file name
		file_name_coup1 = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TC1.csv'  # thermal couples data file name
		path_sensor = folder_data_raw + '\\' + file_name_sen  # file location of sensor data
		path_coup1 = folder_data_raw + '\\' + file_name_coup1  # file location of coupler data
		path_config = cur_dir + '\\' + test_folder + '\\' + '1_Data formatted' + '\\' + test_date + '_config.json'
		# path_data_clearn = folder_data_clean + '\\' + '%s_data_clean_1min.csv' % (test_date)
		raw_sen = TR.DataClean.read_logger_data_DTR_PD(path_sensor)  # read MDC4-M data
		raw_coup1 = TR.DataClean.read_couple_datetime(path_coup1)  # read thermal couples data
		sync_df = TR.DataClean.synch_logger_couple_1min(raw_sen, raw_coup1)  # synchronize two data sets
		# print(sync_df.index)
		# print(sync_df.columns)
		# rename columns based on .json config file
		with open(path_config, 'r') as f:
			config = json.load(f)
		sync_df = sync_df.rename(columns=config)
		print(sync_df.columns)
		return sync_df
		# sync_df.to_csv(path_data_clearn)
	
	def interp_data_nan(data_df, col_name):
        # find the index of zero points
		data_raw = data_df.copy()
		data_raw.index = range(len(data_raw))
		nan_idx = []
		for i in range(len(data_raw)):
			if np.isnan(data_raw.loc[i, col_name]):
				nan_idx.append(i)

		# drop nan of original series
		raw_drop = data_raw[col_name].drop(nan_idx)

		# drop corresponding index of nan element
		idx_drop = data_raw.index.drop(nan_idx)

		# do interpolation
		f_drop = interpolate.interp1d(idx_drop,
									 raw_drop,
									 kind='linear',
									 fill_value='extrapolate')
		for k in nan_idx:
			data_raw.loc[k, col_name] = np.around(f_drop(k), 2)
		data_raw.index = data_df.index
		return data_raw
	
	def read_logger_data(path):
            """method that read the data sampled by TR,GP,DTR,PD data logger,
            Args:
                path                - full path of the data file
            Return:
                raw_data_sliced     - raw data cutted from starting index to ending index
            Notes:
            """
            raw_data = pd.read_csv(path, header=0)
            raw_data = raw_data.dropna()

            # convert string data to float
            raw_data.iloc[:, 1 :].astype(float)
            
            # convert str to datetime
            raw_data.index = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y-%m-%d %H:%M:%S') for i in range(len(raw_data))]
    
            return raw_data
			
class GasPresExperiment_IN(object):

	def __init__(self, file_path):
		"""Init the data of gas pressure experiment.

		Args:
		-------
			data_df 	 - 	raw data of dataframe.
			config_file  - 	dir of config file specifying the name of channels, json file.
		Return:
		-------
			NA
		Notes:
		-------
			NA
		"""
		# load clean data and config file
		data_df = pd.read_csv(file_path,header=2)
		data_df['date_time'] = [datetime.strptime(data_df.loc[i, 'Unnamed: 0'] + '_' + data_df.loc[i, 'Unnamed: 1'],
							'%m/%d/%Y_%I:%M:%S %p') for i in range(len(data_df))]
		self.data = data_df
		self.length = len(self.data)




	# plot regression result
	def plot_t_tank_compare(t_tank_true,t_tank_est,**prop):
		
		x_tick_idx = prop.get('x_tick_idx',[])
		x_tick_str = prop.get('x_tick_str',[])
		title = prop.get('title','t tank comparison')		
		sigma = round((np.mean((t_tank_est - t_tank_true) ** 2)) ** 0.5,1)
		max_abs = round(max(np.abs(t_tank_est - t_tank_true)),1)
		plt.figure(dpi=200)
		ax1 = plt.gca()
		ax1.plot(t_tank_est, 
				 color ='g', 
				 linewidth=0.5, 
				 label='t tank estimated with sigma = %.2f,max_abs = %.2f' % (sigma,max_abs))
		ax1.plot(t_tank_true,
				 color ='k',
				 label='t tank true',
				 linewidth=0.5)
		ax1.set_xlabel('time')
		ax1.set_ylabel('temperature')
		plt.xticks(x_tick_idx,
				   x_tick_str,
				   fontsize=5,
				   rotation=60)

		ax2 = ax1.twinx()
		ax2.plot(t_tank_est - t_tank_true,
				 color ='r',
				 label='error',
				 linewidth=0.5)   
		ax2.set_ylabel('error between est and true')
		ax2.legend(loc="lower right",
		 		   fontsize=6)     
		ax1.legend(loc="lower left",
				   fontsize=6)
			
		plt.title(title)
		plt.grid()

		return sigma,max_abs

	# plot comparison of different fitting approaches
	def plot_p20_compare(p_mano,p20_tank_true,p20_tank_est,**prop):
		x_tick_idx = prop.get('x_tick_idx',[])
		x_tick_str = prop.get('x_tick_str',[])
		sigma = round((np.mean((p20_tank_est - p20_tank_true) ** 2)) ** 0.5,3)
		max_abs = round(max(np.abs(p20_tank_est - p20_tank_true)),3)
		title = prop.get('title','P20 comparison')
		plt.figure(dpi=200, figsize=(10,3))
		ax1 = plt.gca()
		ax1.plot(p_mano,
				 color='k',
				 linewidth=0.5,
				 label='P20 uncompensated'
				)

		ax1.plot(p20_tank_true,
				 color ='g',
				 linewidth=0.5,
				 label='P20 compensated by true tank temp'
				)

		ax1.plot(p20_tank_est, 
				 linewidth=0.5, 
				 label='P20 compensated by est tank temp', 
				 color='b'
				)
		plt.xticks(x_tick_idx,
				   x_tick_str,
				   fontsize=5,
				   rotation=60)
		ax2 = ax1.twinx()
		ax2.plot(p20_tank_est - p20_tank_true, 
				 label='error with sigma={} max_abs={}'.format(sigma,max_abs),
				 color='r', 
				 linewidth=0.5  
				)			
		plt.xlabel('time')
		plt.ylabel('p20 in bar')
		plt.title(title)
		plt.grid()
		ax1.legend(loc="upper right",fontsize=6)
		ax2.legend(loc="upper left",fontsize=6)
		return sigma,max_abs


	@staticmethod
	def down_sample(data,ratio):
		index = np.arange(len(data))
		data_out = data[index%ratio == 0]
		return data_out

	# calculate standard error
	@staticmethod
	def cal_std_err(est,true):
		sigma = round((np.mean((est - true) ** 2)) ** 0.5,1)
		max_abs = round(max(np.abs(est - true)),1)
		return sigma,max_abs

	# calculate polynomial coefficients
	@staticmethod
	def cal_polyfit_coef(order, y, x):
		coef = np.polyfit(x, y, order)
		return coef

	# calculate least square regression coefficients
	@staticmethod
	def cal_lsr_reg_coef(y, x):
		reg = linear_model.LinearRegression()
		reg.fit(x, y)
		coef = np.append(reg.coef_, reg.intercept_, axis=None)

		return coef
	
	# @staticmethod
	# def date_time_to_x_tick(date_time):
	# 	datetime.strftime(date_time)

