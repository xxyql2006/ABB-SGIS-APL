import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime
from sklearn import linear_model
# import math



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
		plt.figure(dpi=200)
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

