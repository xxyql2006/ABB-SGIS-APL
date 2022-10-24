

from datetime import datetime
from matplotlib.pyplot import title
import bob_GP_lib_V12 as GP
import pandas as pd
import numpy as np
import os

#====================================================================================
# 初始化数据
#====================================================================================
ver = 'bob_GP_run_V5_1'
full_date_list= ['210312','210315','210316','210317','210318','210319','210618','210621','210622','210625']
time_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') # 提取执行程序时间
print(time_now)
result_folder = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\2_Gas pressure\06_Analytics' + '\\' + ver # 结果存放文件夹
data_folder = r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\1_Temperature rise\11_Data\40kV TR test\Data_Cleansed'# 清洗过数据存放文件夹
if os.path.exists(result_folder): # 如果文件夹不存在就创建
	pass
else:
	os.mkdir(result_folder)


#%%
# ==========================================================
# 拼接训练集数据
# ========================================================== 
sample_time = '30min' 
train_len_list = [0]
data_stack_df = pd.DataFrame(columns=['datetime','t_meter','t_amb','t_tank','p_meter','p20_meter'])
for j,train_date in enumerate(full_date_list):			
	path_data_stack = (r'C:\Users\cnbofan1\ABB\Safe Digital in CN - Documents - Documents\1_Temperature rise\11_Data\40kV TR test'
	+ '\\' + 'data_cleansed' + '\\' + '%s_data_clean_%s_gp.csv' % (train_date,sample_time))
	data_stack_df = data_stack_df.append(pd.read_csv(path_data_stack,header = 0))
	train_len_list.append(len(data_stack_df))
t = data_stack_df['t_amb'].values # 环温
t1 = data_stack_df['t_meter'].values # 气压表温度             
t_sf6_true = data_stack_df['t_tank'].values # 气箱中部温度   	

x5_train = np.array([i for num,i in enumerate(np.diff(t1,prepend=t1[0])) if num not in train_len_list[0:-1]])
x6_train = np.array([i for num,i in enumerate(np.diff(t1-t,prepend=t1[0]-t[0])) if num not in train_len_list[0:-1]])

# 由于拼接点差分长度比原始数据减少1，故删除原始数据拼接点位置数据
x1_train = np.array([i for num,i in enumerate(t1 - t) if num not in train_len_list[0:-1]])
x2_train = x1_train ** 2
x3_train = np.array([i for num,i in enumerate(t) if num not in train_len_list[0:-1]])
x4_train = np.array([i for num,i in enumerate(t1) if num not in train_len_list[0:-1]])
x7_train = x6_train * abs(x6_train)
y_train = np.array([i for num,i in enumerate(t_sf6_true - t) if num not in train_len_list[0:-1]])
y1_train = np.array([i for num,i in enumerate(t_sf6_true) if num not in train_len_list[0:-1]])

# 把向量拼接成矩阵
X_e1 = np.column_stack((x1_train,x6_train)) 
X_e2 = np.column_stack((x1_train,x2_train,x6_train,x7_train)) 
X_b1 = np.column_stack((x3_train,x4_train)) 
X_b2 = np.column_stack((x3_train,x4_train,x5_train))

# 计算线性回归系数
# lsr_e1_coef = np.around(GP.GasPress.cal_lsr_reg_coef(y_train,X_e1),decimals=2) 
# lsr_e2_coef = np.around(GP.GasPress.cal_lsr_reg_coef(y_train,X_e2),decimals=2)
# lsr_b1_coef = np.around(GP.GasPress.cal_lsr_reg_coef(y1_train,X_b1),decimals=2)
lsr_b2_coef = np.around(GP.GasPress.cal_lsr_reg_coef(y1_train,X_b2),decimals=2)
print('Least Square Regression Coeficients for t,t1,t1_diff1 and constant are : ',lsr_b2_coef)
#%%

