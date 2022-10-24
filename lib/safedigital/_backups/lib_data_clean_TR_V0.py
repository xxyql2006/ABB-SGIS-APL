# -*- coding: utf-8 -*-
"""
Created on Mar 5th 2021
Rev 0 Data cleansing was seperated from main programme

@author: CNBOFAN1
"""
#%%

from datetime import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
 # Format YYMMDD 

class DataClean:

	def read_sensor_data(path):
		#读取输入路径的传感器数据文件
		raw_data = pd.read_csv(path,header = None) #将读取文件传入数据帧
		raw_data.iloc[:,2:].astype(float) # 第2列以后全部强制转换为浮点数
		raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i,0]+':'+raw_data.iloc[i,1],
		'%m/%d/%Y:%I:%M:%S %p') for i in range(len(raw_data))] #粘合第0列和第1列行程datetime类型数据存入到新的一列
		
		# 寻找试验开始时间，条件为智能传感器温度至少有一个值非0
		for i in range(len(raw_data)):
			
			if (raw_data.iloc[i,2] != 0 and raw_data.iloc[i,3] != 0 and raw_data.iloc[i,4] 
				!= 0 and raw_data.iloc[i,5] != 0 and raw_data.iloc[i,6] != 0 and raw_data.iloc[i,7] != 0
				and raw_data.iloc[i,8] != 0 and raw_data.iloc[i,9] != 0 and raw_data.iloc[i,10] != 0):
				t0 = i
				print('sensor data start from %s' % (raw_data.loc[i,'datetime'].strftime("%H:%M:%S")))
				break
			elif i == (len(raw_data) - 1):
				print('sensor data was all "0"')
			else:
				continue
		 
		# 寻找试验结束时间，条件为智能传感器温度全为0 
		for i in range(t0,len(raw_data)):
			if (raw_data.iloc[i,2] == 0 and raw_data.iloc[i,3] == 0 and raw_data.iloc[i,4] 
				== 0 and raw_data.iloc[i,5] == 0 and raw_data.iloc[i,6] == 0 and raw_data.iloc[i,7] == 0
				and raw_data.iloc[i,8] == 0 and raw_data.iloc[i,9] == 0 and raw_data.iloc[i,10] == 0):
				tn = i
				print('sensor data ended at %s' % (raw_data.loc[i,'datetime'].strftime("%H:%M:%S")))
				break
			elif i == (len(raw_data)-1):
				tn = i + 1
				print('sensor data ended at %s' % (raw_data.loc[i,'datetime'].strftime("%H:%M:%S")))
			else:
				continue        
		
		# 检查采样间隔是否为10秒，不是的话降采样到10秒
		time_interval = (raw_data.iloc[1,-1] - raw_data.iloc[0,-1]).seconds
		if time_interval != 10:
			multiple = 10/time_interval
			raw_data = raw_data[raw_data.index%multiple==0]
		else:
			pass

		# 截取试验开始到结束的有效数据 
		return raw_data.iloc[t0:tn,:]

        
	#====================================================================================
	# 读取热电偶数据
	#====================================================================================    

	def read_coupler_data(path):

		raw_data = pd.read_csv(path,header = 25,na_values=['          ','     -OVER'])		
		raw_data.iloc[:,4:].astype(float)
		raw_data = raw_data.fillna(0)
		raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i,1] + ':' + raw_data.iloc[i,2],
		'%Y/%m/%d:%H:%M:%S') for i in range(len(raw_data.iloc[:,2]))]
		
		# 检查采样间隔，不是十秒的话降采样至10秒
		time_interval = (raw_data.iloc[1,-1] - raw_data.iloc[0,-1]).seconds
		if time_interval != 10:
			multiple = 10/time_interval
			raw_data = raw_data[raw_data.index%multiple==0]
		else:
			pass

		return raw_data

           
	#====================================================================================
	# 找到三组数据重叠的起止时间，并且同步三组数据的时间轴
	#====================================================================================    

	def synch_data_group(*data):
		
		start_time = max([data[i].iloc[0,-1] for i in range(len(data))]) # 找到所有组数据的共同开始时间
		data_list = []
		print('sensor & couplers common start time = ',start_time)
		j = 0
		for element in data:
			j += 1
			element['ind'] = [round((element.iloc[i,-1] - start_time).seconds/10) for i in range(len(element))]			
			element = element.loc[(element['ind'] >= 0) & (element['ind'] <= 4320),:]     

			# 解决由于round函数造成的出现重复索引的问题
			for k in range(len(element)-1):
				if element.iloc[k,-1] == element.iloc[k+1,-1]:
					element.iloc[k+1,-1] = element.iloc[k,9] + 1
					k = k + 1
				else:
					pass			
			# 检查数据是否存在间断索引
			if (element.iloc[-1,-1] - element.iloc[0,-1]) != (len(element)-1):
				print('%dth group of data has discontinued points' % j)
				print('head,tail and length are ',element.iloc[0,-1],element.iloc[-1,-1],len(element))
				full_index = pd.Series(range(element.iloc[0,-1],element.iloc[-1,-1]+1))
				miss_index = full_index.loc[~full_index.isin(list(element['ind']))]        
				element_full = pd.DataFrame(data=0,columns=range(element.shape[1]),index=range(len(full_index)))

				# 填入旧数据
				for i in range(len(element)):
					element_full.iloc[element.iloc[i,-1]-element.iloc[0,-1],:] = element.iloc[i,:]
				
				# 填入缺失索引
				for i in list(miss_index):
					element_full.iloc[i-element.iloc[0,-1],-1] = i 
				data_list.append(element_full)	
			else:
				print('%dth group of data has no discontinued points' % j)
				data_list.append(element)
				pass
			# 重新
		# 找到三组数据重叠数据点的起止索引
		start_ind = max([i.iloc[0,-1] for i in data_list])

		end_ind = min([i.iloc[-1,-1] for i in data_list])


		
		# 用共同起止索引截取三组数据重叠的数据点
		for j,ele in enumerate(data_list): 
			data_list[j] = ele.loc[ele.iloc[:,-1] <= end_ind,:]

			data_list[j] = data_list[j].loc[data_list[j].iloc[:,-1] >= start_ind,:]

			data_list[j].index = range(len(data_list[j]))


		return data_list

	# 寻找热平衡点 
	def find_balance_index(data,bal_col_list):
		
		data_sliced = data.iloc[:,bal_col_list]
		data_sliced_dif = pd.DataFrame(index=range(10),columns=range(len(bal_col_list)))

		for i in range(1440,len(data)):

			data_sliced_dif = (data_sliced.iloc[i,:] - data_sliced.iloc[i-360,:]).abs()
  
			if (data_sliced_dif <= 1.0).all(axis=None):
				t0 = i
				# print('Temp balanced @ %s' %(str(data.iloc[i,3]))) 
				break
			elif  i == len(data) - 1:
				print('Temp not balanced')
				break
		# bal_hr =int(self.t0 / 360)
		# self.index_seq = [i * 360 for i in range(bal_hr)]
		# self.index_seq.append(self.t0) 
		# self.str_seq = [(self.start_time + timedelta(hours = i)).strftime('%H:%M') for i in range(bal_hr)]
		# self.str_seq.append(data.iloc[self.t0,3].strftime('%H:%M'))
		# print('t0 = ',self.t0)
		return t0

	# 对错误数据标记的“0”点进行插值，补全数据
	def data_interpol(data,ind_col,data_cols):

		time_index = data.iloc[:,ind_col]
		time_index.index = range(len(time_index))
		for i in data_cols:
			print('column number',i)
			zero_index = []
			for j in range(len(data)):
				if  data.iloc[j,i] == 0:
					zero_index.append(j)
			# print(zero_index)                
			raw = data.iloc[:,i]
			raw.index = range(len(raw))
			raw_drop = raw.drop(zero_index)
			time_index_drop = time_index.drop(zero_index)
			# f_raw = interpolate.interp1d(index_drop,raw_drop,kind='linear',fill_value='extrapolate')
			f_raw = interpolate.interp1d(time_index_drop,raw_drop,kind='linear',fill_value='extrapolate')
			for k in zero_index:
				data.iloc[k,i] = np.around(f_raw(k),1)
		return data
	
	#====================================================================================
	# 数据滤波器，包含硬边界滤波以及差分限幅滤波
	#====================================================================================
	def data_filter(data,data_cols,low_lim_list,high_lim_list,lim_diff):

		for i in range(len(data_cols)):
			col = data_cols[i]
			diff = np.diff(data.iloc[:,col].values,prepend=data.iloc[0,col])
			for j in range(len(data)):
				if  (data.iloc[j,col] > high_lim_list[i]) or (data.iloc[j,col] < low_lim_list[i]):
					data.iloc[j,col] = 0

				else:
					pass
			for k in range(len(data)-1):
				if 	((diff[k] > lim_diff[i]) or (diff[k] < -lim_diff[i])) & ((diff[k+1] > lim_diff[i]) or (diff[k+1] < -lim_diff[i])):
					data.iloc[k,col] = 0
       
		return data

	@staticmethod
	def down_sample(data,ratio):
		index = np.arange(len(data))
		data_out = data[index%ratio == 0]
		return data_out





      
    

# %%
