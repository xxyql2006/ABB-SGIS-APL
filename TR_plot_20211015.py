# -*- coding: utf-8 -*-
"""
TR test 20211015 data plot
Notes:
    using safedigital package
    using temperature module as TR
"""

# %%
from safedigital import temperature as TR
import os

os.chdir('../')
curDirectory = os.getcwd()
print(curDirectory)

#%%
file_path = curDirectory + '\\20211015_TR SR40.5kV cable skin horz 630A\\1_Data formatted\\20211015_data_clean_10s.csv'
config_path = curDirectory + '\\20211015_TR SR40.5kV cable skin horz 630A\\1_Data formatted\\2021-10-15_config.json'
fig_path = curDirectory + '\\20211015_TR SR40.5kV cable skin horz 630A\\3_Graphs'

# %%
col_list = ["t_C1_phA_snsr_skin_horz_top",
            "t_C1_phA_cplr_skin_horz_top",
            "t_C1_phA_snsr_bushing",
            "t_C1_phA_cplr_bushing"]

line_style = ['-', '--', '-', '--']
line_color = ['r', 'r', 'g', 'g']


case1014 = TR.TempRiseExperiment(file_path, config_path)
case1014.find_balance_index(list(range(18, 48)))
case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase A @ 630A')

# %%
col_list = ["t_C1_phB_snsr_skin_horz_top",
            "t_C1_phB_cplr_skin_horz_top",
            "t_C1_phB_snsr_bushing",
            "t_C1_phB_cplr_bushing"]

line_style = ['-', '--', '-', '--']
line_color = ['r', 'r', 'g', 'g']


case1014 = TR.TempRiseExperiment(file_path, config_path)
case1014.find_balance_index(list(range(18, 48)))
case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase B @ 630A')

# %%
col_list = ["t_C1_phC_snsr_skin_horz_top",
            "t_C1_phC_cplr_skin_horz_top",
            "t_C1_phC_snsr_bushing",
            "t_C1_phC_cplr_bushing"]

line_style = ['-', '--', '-', '--']
line_color = ['r', 'r', 'g', 'g']


case1014 = TR.TempRiseExperiment(file_path, config_path)
case1014.find_balance_index(list(range(18, 48)))
case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase C @ 630A')