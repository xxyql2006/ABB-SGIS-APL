"""
@作者: Bob
版本控制
版本号		日期		 	描述		
Rev 0 		2021.10.28	   初版

说明：数据清洗只包含以下步骤
1.读取数据
2.同步时间戳
3.黏合所有数据
4.调整不同采样间隔
"""
#%%
import lib.lib_data_clean_TR_V0 as dc # TR清洗数据库函数
import pandas as pd
#=================================================================
# 初始化变量
#=================================================================

test_date = '20211019' # 全部数据集\
sample_time_list = [1] # 降采样系数列表
sample_str_list = ['10s'] # 采样时间列表
sample_time_dict = dict(zip(sample_str_list,sample_time_list)) # 建立采样时间和降采样系数的对应关系的字典
device_name = '_TR_SafeRing 40.5kV cable skin TR test'

#=================================================================
# 							数据清洗 
#=================================================================
	
# 初始化数据
test_date_folder = test_date + device_name
folder_data = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\06_Test and Analytics' + '\\' + test_date_folder + '\\' + '0_Data original'
folder_data_clean = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\06_Test and Analytics' + '\\' + test_date_folder + '\\' + '1_Data formatted'


# 不同日期的数据文件名称  
file0 = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TP.csv' #'2021-XX-XX_TP.csv'
file1 = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TC1.csv' #'2021-XX-XX_TC1.csv'

# 不同组数据的完整路径
path_sensor = folder_data + '\\' + file0 # file location of sensor data
path_coup1 = folder_data + '\\' + file1 # file location of coupler data

print('\ntest date : ',test_date)
raw_sen = dc.DataClean.read_sensor_data(path_sensor) # 读取数据 
raw_coup1 = dc.DataClean.read_coupler_data(path_coup1) # 读取数据
syn_sen,syn_coup1 = dc.DataClean.synch_data_group(raw_sen,raw_coup1) # 同步三组数据时间戳

chn_num_tc1 = syn_coup1.shape[1] - 6 # 热电偶1的列数：前四列和后两列不是数据

# 生成不同采样间隔的清洗数据
for sample_ind,sample_str in enumerate(sample_str_list):
	path_data_clean = folder_data_clean + '\\' + '%s_data_clean_%s.csv' % (test_date,sample_str) # 清洗完文件位置
	syn_sen = dc.DataClean.down_sample(syn_sen,sample_time_dict[sample_str]) # 降采样
	syn_coup1 = dc.DataClean.down_sample(syn_coup1,sample_time_dict[sample_str]) # 降采样
	# syn_coup2 = dc.DataClean.down_sample(syn_coup2,sample_time_dict[sample_str]) # 降采样
	
	# 对温度传感器数据进行硬边界滤波和差值限幅滤波
	# filt_sen = dc.DataClean.data_filter(syn_sen,list(range(2,11)),[0]*9,[100]*9,[1]*9)
	
	# 对热电偶数据进行硬边界滤波和差值限幅滤波  
	
	# filt_coup1 = dc.DataClean.data_filter(syn_coup1,list(range(4,chn_num_tc1+4)),[0]*chn_num_tc1,
	# [100]*chn_num_tc1,[1]*chn_num_tc1)		
	# filt_coup2 = dc.DataClean.data_filter(syn_coup2,list(range(4,chn_num_tc2+4)),[0]*chn_num_tc2,
	# [100]*chn_num_tc2,[1]*chn_num_tc2)

	# 对所有组数据进行插值
	# inter_sen = dc.DataClean.data_interpol(filt_sen,-1,[2,3,4,5,6,7,13,14,15]) 
	# inter_coup1 = dc.DataClean.data_interpol(filt_coup1,-1,list(range(4,chn_num_tc1+4)))
	# inter_coup2 = dc.DataClean.data_interpol(filt_coup2,-1,list(range(4,chn_num_tc2+4))) 
	
	# 对所有环温求平均
	temp_amb_avg = syn_coup1.iloc[:,63].values # 环境温度为三个油瓶的平均值

	for i in range(4,chn_num_tc1+3):
			syn_coup1.iloc[:,i] = syn_coup1.iloc[:,i].values + syn_coup1.iloc[:,63].values
	# for i in range(4,chn_num_tc1+3):
	# 		syn_coup2.iloc[:,i] = syn_coup2.iloc[:,i].values + syn_coup2.iloc[:,chn_num_tc2+3].values
	col_sen = ['t_sen1','t_sen2','t_sen3','t_sen4','t_sen5','t_sen6','t_sen7','t_sen8','t_sen9',
	't_amb_sen','hum_amb_sen','t_mano','p_mano','p20_mano','P20_avg_mano','data_time_sen','ind_sen']
	col_data_sen = [syn_sen.iloc[:,i].values for i in list(range(2,19))]
	col_tc1 = ['ch%d' % i for i in range(1,61)] + ['data_time_coup1','ind_coup1']
	col_data_tc1 = [syn_coup1.iloc[:,i].values for i in list(range(4,66))]
	# col_tc2 = ['ch%d' % i for i in range(61,61+chn_num_tc2)]
	# col_data_tc2 = [syn_coup2.iloc[:,i].values for i in list(range(4,4+chn_num_tc2))]
	col_all = col_sen + col_tc1
	col_data_all = col_data_sen + col_data_tc1
	data_clean_df = pd.DataFrame(dict(zip(col_all,col_data_all)))
	data_clean_df.to_csv(path_data_clean)
#%%