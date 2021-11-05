"""
@作者: Bob
版本控制
版本号		日期		 	描述		
Rev 0 		2021.11.03	   初版

说明：数据清洗只包含以下步骤
1.读取数据
2.同步时间戳
3.黏合所有数据
4.调整不同采样间隔
"""
# %%
from safedigital import temperature as TR  # 从safedigital包中导入temperature库
import pandas as pd
import os

# =================================================================
# 初始化变量
# =================================================================
os.chdir('../')
cur_dir = os.getcwd()
test_date = '20211104' # 试验日期
sample_time_list = [1]  # 降采样系数列表
sample_str_list = ['10s']  # 采样时间列表
sample_time_dict = dict(zip(sample_str_list, sample_time_list)) # 建立采样间隔与降采样比例的字典
test_folder = test_date + '_DTR&TR_3in1_SR12kV 300A' # 试验文件夹
folder_data_raw = cur_dir + '\\' + test_folder + '\\' + '0_Data original' # 原始数据文件夹
folder_data_clean = cur_dir + '\\' + test_folder + '\\' + '1_Data formatted' # 清理完数据文件夹
file_name_sen = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TP.csv'  # 传感器数据文件名
file_name_coup1 = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TC1.csv'  # 热电偶数据文件名'
path_sensor = folder_data_raw + '\\' + file_name_sen  # file location of sensor data
path_coup1 = folder_data_raw + '\\' + file_name_coup1  # file location of coupler data
raw_sen = TR.DataClean.read_sensor_data(path_sensor)  # 读取数据
raw_coup1 = TR.DataClean.read_coupler_data(path_coup1)  # 读取数据
syn_sen, syn_coup1 = TR.DataClean.synch_data_group(raw_sen, raw_coup1)  # 同步三组数据时间戳

chn_num_tc1 = syn_coup1.shape[1] - 6  # 热电偶1的列数：前四列为无效数据，后两列为时间戳和索引

# 生成不同采样间隔的清洗数据
for sample_ind, sample_str in enumerate(sample_str_list):
    path_data_clean = folder_data_clean + '\\' + '%s_data_clean_%s.csv' % (test_date, sample_str)  # 清洗完文件位置
    syn_sen = TR.DataClean.down_sample(
        syn_sen, sample_time_dict[sample_str])  # 传感器数据降采样
    syn_coup1 = TR.DataClean.down_sample(
        syn_coup1, sample_time_dict[sample_str])  # 热电偶1数据降采样
    # syn_coup2 = dc.DataClean.down_sample(syn_coup2,sample_time_dict[sample_str]) # 降采样
	
    for i in range(4, chn_num_tc1+3): # 遍历所有热电偶通道的数据，实际温度为本通道温度+通道60的温度
        syn_coup1.iloc[:, i] = syn_coup1.iloc[:, i].values + syn_coup1.iloc[:, 63].values
    # for i in range(4,chn_num_tc1+3):
    # 		syn_coup2.iloc[:,i] = syn_coup2.iloc[:,i].values + syn_coup2.iloc[:,chn_num_tc2+3].values
    col_sen = ['t_sen1', 't_sen2', 't_sen3', 't_sen4', 't_sen5', 't_sen6', 't_sen7', 't_sen8', 't_sen9',
               't_amb_sen', 'hum_amb_sen', 't_mano', 'p_mano', 'p20_mano', 'P20_avg_mano', 'data_time_sen', 'ind_sen']
    col_data_sen = [syn_sen.iloc[:, i].values for i in list(range(2, 19))]
    col_tc1 = ['ch%d' % i for i in range(
        1, 61)] + ['data_time_coup1', 'ind_coup1']
    col_data_tc1 = [syn_coup1.iloc[:, i].values for i in list(range(4, 66))]
    # col_tc2 = ['ch%d' % i for i in range(61,61+chn_num_tc2)]
    # col_data_tc2 = [syn_coup2.iloc[:,i].values for i in list(range(4,4+chn_num_tc2))]
    col_all = col_sen + col_tc1
    col_data_all = col_data_sen + col_data_tc1
    data_clean_df = pd.DataFrame(dict(zip(col_all, col_data_all)))
    data_clean_df.to_csv(path_data_clean)
# %%
