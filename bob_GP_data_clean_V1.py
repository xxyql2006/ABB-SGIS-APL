"""
Created on May 7th 2021
Rev 0 	Data cleansing was seperated from main programme
Rev 1   Added 'P20 from Manometer' column as cleansing data 


@author: CNBOFAN1
"""
#%%
import lib_all_data_clean_V0 as dc
import pandas as pd
import os




#====================================================================================
# 初始化
#====================================================================================

all_date_list = ['210312','210315','210316','210317','210318','210319','210618','210621','210622','210625'] # 全部数据集\


#=================================================================
# 							数据清洗 
#=================================================================
for test_num,test_date in enumerate(all_date_list):
	folder_data = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\1_Temperature rise\11_Data\40kV TR test\Data_'+test_date
	folder_data_clean = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\1_Temperature rise\11_Data\40kV TR test' + '\\' + 'data_cleansed'
	if os.path.exists(folder_data_clean):
		pass
	else:
		os.mkdir(folder_data_clean)
	path_data_clean = folder_data_clean + '\\' + '%s_data_clean_all_channel.csv' % (test_date)   
	file0 = '20' + test_date[0:2] + '-' + test_date[2:4] + '-' + test_date[4:6] + '_TP.csv' #'2021-XX-XX_TP.csv'
	file1 = '20' + test_date[0:2] + '-' + test_date[2:4] + '-' + test_date[4:6] + '_TC1.csv' #'2021-XX-XX_TC1.csv'
	file2 = '20' + test_date[0:2] + '-' + test_date[2:4] + '-' + test_date[4:6] + '_TC2.csv' #'2021-XX-XX_TC2.csv'

	FileLocSensor = folder_data + '\\' + file0 # file location of sensor data
	FileLocCoupler1 = folder_data + '\\' + file1 # file location of coupler data
	FileLocCoupler2= folder_data + '\\' + file2

	print('\ntest date : ',test_date)
	raw_sen = dc.DataClean.read_sensor_data(FileLocSensor)
	raw_coup1 = dc.DataClean.read_coupler_data(FileLocCoupler1)
	raw_coup2 = dc.DataClean.read_coupler_data(FileLocCoupler2)
	syn_sen,syn_coup1,syn_coup2 = dc.DataClean.synch_data_group(raw_sen,raw_coup1,raw_coup2)

	# syn_sen = dc.DataClean.down_sample(syn_sen,30) # 降采样至5分钟
	# syn_coup1 = dc.DataClean.down_sample(syn_coup1,30) # 降采样至5分钟
	# syn_coup2 = dc.DataClean.down_sample(syn_coup2,30) # 降采样至5分钟
	chn_num_tc2 = syn_coup2.shape[1] - 4 # 热电偶2的列数
	filt_sen = dc.DataClean.data_filter(syn_sen,[2,3,4,5,6,7,13,14,15],[0,0,0,0,0,0,0,100,100],
	[100,100,100,100,100,100,100,200,200],[1,1,1,1,1,1,1,3,3])
	filt_coup1 = dc.DataClean.data_filter(syn_coup1,list(range(4,64)),[0]*60,[100]*60,[1]*60)
	filt_coup2 = dc.DataClean.data_filter(syn_coup2,list(range(4,chn_num_tc2+4)),[0]*chn_num_tc2,
	[100]*chn_num_tc2,[1]*chn_num_tc2)
	inter_sen = dc.DataClean.data_interpol(filt_sen,9,[2,3,4,5,6,7,13,14,15]) 
	
	if test_date in ['210312','210315','210316']:
		inter_coup1 = dc.DataClean.data_interpol(filt_coup1,0,(list(range(4,52))+[63]))
		inter_coup2 = dc.DataClean.data_interpol(filt_coup2,0,list(range(4,chn_num_tc2+4)))
	elif test_date in ['210618','210621','210622','210625']:
		inter_coup1 = dc.DataClean.data_interpol(filt_coup1,0,(list(range(4,58))+[63]))
		inter_coup2 = dc.DataClean.data_interpol(filt_coup2,0,list(range(4,chn_num_tc2))+[chn_num_tc2+3])
	elif test_date in ['210317','210318','210319']:	
		inter_coup1 = dc.DataClean.data_interpol(filt_coup1,0,(list(range(4,52))+[63]))
		inter_coup2 = dc.DataClean.data_interpol(filt_coup2,0,list(range(4,chn_num_tc2-16)))
	if test_date in ['210618','210621','210622','210625']:
		for i in range(4,63):
			inter_coup1.iloc[:,i] = inter_coup1.iloc[:,i].values + inter_coup1.iloc[:,63].values
	else:
		pass
	col_sen = ['datetime','t_sen1','t_sen2','t_sen3','t_sen4','t_sen5','t_sen6','t_amb_sen','hum_amb_sen',
	't_mano','p_mano','p20_mano']
	col_data_sen = [inter_sen.iloc[:,i].values for i in [8,2,3,4,5,6,7,11,12,13,14,15]]
	col_tc1 = ['ch%d' % i for i in range(1,61)]
	col_data_tc1 = [inter_coup1.iloc[:,i].values for i in list(range(4,64))]
	col_tc2 = ['ch%d' % i for i in range(61,61+chn_num_tc2)]
	col_data_tc2 = [inter_coup2.iloc[:,i].values for i in list(range(4,4+chn_num_tc2))]
	col_all = col_sen + col_tc1 + col_tc2
	col_data_all = col_data_sen + col_data_tc1 + col_data_tc2

	data_clean_df = pd.DataFrame(dict(zip(col_all,col_data_all)))
	data_clean_df.to_csv(path_data_clean)
#%%