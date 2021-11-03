# -*- coding: utf-8 -*-
"""@author: CNBOFAN1
版本号		日期		 	描述		
Rev 0 		2021.10.20	   初版 
Rev 1       2021.11.01     减少可视化绘图中每个图片曲线个数

"""

# %%

from safedigital import temperature as TR
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

os.chdir('../')
curDirectory = os.getcwd()
# print(curDirectory)

# %%
file_name = curDirectory + '\\20211019_TR_SafeRing 40.5kV cable skin TR test\\1_Data formatted\\20211019_data_clean_10s.csv'
config_file = curDirectory + '\\20211019_TR_SafeRing 40.5kV cable skin TR test\\0_Data original\\2021-10-19_config.json'
# fig_dir = curDirectory + '\\20211019_TR_SafeRing 40.5kV cable skin TR test\\3_Graphs\\a.png'
data = pd.read_csv(file_name)
with open(config_file,'r') as f:
    config = json.load(f)
# print(data.shape)

#%%
data = data.rename(columns=config)


# %%
fig = plt.Figure()
sns.set(color_codes=True)

plt.plot(data['tc_index'].values, data['C1_phaseA_bushing'].values, label="C1_phaseA_bushing")
plt.plot(data['tc_index'].values, data['C1_phaseB_bushing'].values, label="C1_phaseB_bushing")
plt.plot(data['tc_index'].values, data['C1_phaseC_bushing'].values, label="C1_phaseC_bushing")
# plt.plot(data['ind_sen'].values, data['t_sen4'].values)
# plt.xticks(data['ind_sen'].values, data['data_time_sen'].values)
# plt.xlabel('Time (hours:minutes)')
plt.title('hello')
plt.legend()
plt.show()

# %%
case = TR.TempRiseExperiment(dir1, dir2)