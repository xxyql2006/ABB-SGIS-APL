# -*- coding: utf-8 -*-
"""
TR test 20211101 data plot
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
file_path = curDirectory + '\\20211101_DTR&TR_SR12kV 300A\\1_Data formatted\\20211101_data_clean_10s.csv'
config_path = curDirectory + '\\20211101_DTR&TR_SR12kV 300A\\1_Data formatted\\2021-11-01_config.json'
fig_path = curDirectory + '\\20211101_DTR&TR_SR12kV 300A\\3_Graphs'

line_style = ['-', '--', ':', '-', '--']
line_color = ['r', 'r', 'r', 'g', 'g']

case1014 = TR.TempRiseExperiment(file_path, config_path)
case1014.find_balance_index(list(range(18, 52)))

# %%
col_list = ["t_C1_phA_snsr_skin_vert_out",
            "t_C1_phA_skin_vert_out_upper",
            "t_C1_phA_skin_vert_out_low",
            "t_C1_phA_snsr_bushing",
            "t_C1_phA_bushing"]

case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase A @ 300A')

# %%
col_list = ["t_C1_phB_snsr_skin_vert_out",
            "t_C1_phB_skin_vert_out_upper",
            "t_C1_phB_skin_vert_out_low",
            "t_C1_phB_snsr_bushing",
            "t_C1_phB_bushing"]

case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase B @ 300A')

# %%
col_list = ["t_C1_phC_snsr_skin_vert_out",
            "t_C1_phC_skin_vert_out_upper",
            "t_C1_phC_skin_vert_out_low",
            "t_C1_phC_snsr_bushing",
            "t_C1_phC_bushing"]

case1014.t_plot(col_name_list=col_list,
                fig_path=fig_path,
                line_style=line_style,
                line_color=line_color,
                title='Skin TR sensor test Phase C @ 300A')