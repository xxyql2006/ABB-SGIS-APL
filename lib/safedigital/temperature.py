# -*- coding: utf-8 -*-
"""
Created on 20211103
Rev1 - Initial document
Rev2 - Add dynamic TR method

@author: Bob/Eric
"""
# %% load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from datetime import datetime
from datetime import timedelta
from datetime import time
import seaborn as sns
import json
import math
from scipy import optimize


# %%


class TempRiseExperiment(object):
    """Class of a set of temperature rise experiment.
    """

    def __init__(self, clean_data_file, config_file):
        """Init the data of temperature rise experiment.

        Args:
        -------
            clean_data_file - dir of clean data file.
            config_file     - dir of config file specifying the name of channels, json file.
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # load clean data and config file
        self.data = pd.read_csv(clean_data_file)
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # rename the columns of data(DataFrame) by using config.json file
        self.data = self.data.rename(columns=self.config)
        self.data.index = [datetime.strptime(self.data.loc[i,'cplr_datetime'], 
                                             '%Y-%m-%d %H:%M:%S') for i in range(len(self.data))]

        # initial parameters
        self.bal_idx = 0
        self.x_idx_list = []
        self.x_str_list = []
        self.t_oil = self.data.loc[:, 't_oil_bottle_1']
        self.t_env = self.data.loc[:, 't_env']
        self.current = [0] * self.data.shape[0]
        self.cur_phA = [0] * self.data.shape[0]
        self.cur_phB = [0] * self.data.shape[0]
        self.cur_phC = [0] * self.data.shape[0]

    def interp_data_zero(self, col_name):
        """interpolate the "0" in data column.

        Args:
        -------
            col_name - column name to be interpolated.
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # find the index of zero points
        zero_index = []
        for i in range(len(self.data[col_name])):
            if self.data.loc[i, col_name] == 0:
                zero_index.append(i)
        # print(zero_index)

        # drop zero of original series
        raw_drop = self.data[col_name].drop(zero_index)

        # drop zero of time index
        time_index_drop = self.data['snsr_time_index'].drop(zero_index)

        # do interpolation
        f_raw = interpolate.interp1d(
            time_index_drop, raw_drop, kind='linear', fill_value='extrapolate')
        for k in zero_index:
            self.data.loc[k, col_name] = np.around(f_raw(k), 1)

    def interp_data_nan(self, col_name):
        # find the index of zero points
        nan_index = []
        for i in range(len(self.data[col_name])):
            if np.isnan(self.data.loc[i, col_name]):
                nan_index.append(i)
        # print(nan_index)

        # drop nan of original series
        raw_drop = self.data[col_name].drop(nan_index)

        # drop zero of time index
        time_index_drop = self.data['snsr_time_index'].drop(nan_index)

        # do interpolation
        f_raw = interpolate.interp1d(time_index_drop,
                                     raw_drop,
                                     kind='linear',
                                     fill_value='extrapolate')
        for k in nan_index:
            self.data.loc[k, col_name] = np.around(f_raw(k), 1)

    def drop_data_zero(self, col_name):
        """drop zero element of a data column.

        Args:
        -------
            col_name - column name to be dropped.
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # find the index of zero points
        zero_index_list = []
        for i in range(len(self.data[col_name])):
            if self.data.loc[self.data.index[i], col_name] == 0:
                zero_index_list.append(self.data.index[i])

        # drop zero of original series
        data_drop = self.data[col_name].drop(zero_index_list)
        return data_drop

    def drop_data_negative(self, data_series):
        """drop negative element of a data array.

        Args:
        -------
            data_array - to be dropped.
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # find the index of zero points
        zero_index_list = []
        for i in range(len(data_series)):
            if data_series.iloc[i] < 0:
                zero_index_list.append(data_series.index[i])

        # drop zero of original series
        data_drop = data_series.drop(zero_index_list)
        return data_drop
    
    def drop_data_threshold(self, data_series, threshold):
            """drop elements less than a certain threshold  of a data array.

            Args:
            -------
                data_array - to be dropped.
            Return:
            -------
                NA
            Notes:
            -------
                NA
            """
            # find the index of zero points
            zero_index_list = []
            for i in range(len(data_series)):
                if (data_series.iloc[i] > threshold) or (data_series.iloc[i] < -threshold) :
                    zero_index_list.append(data_series.index[i])

            # drop zero of original series
            data_drop = data_series.drop(zero_index_list)
            return data_drop

    def np_move_avg(self, a, n, mode="valid"):
        return (np.convolve(a, np.ones((n,)) / n, mode=mode))

    def t_plot(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature Rise')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')

        # figure
        plt.figure(dpi=200)
        # sns.set(color_codes=True)

        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # plot curves as per "col_name_list"
        # try:
        for i, name in enumerate(col_name_list):
            # self.interp_data_zero(name)
            plt.plot(self.data['snsr_time_index'].values,
                     self.data[name].values -
                     self.data['t_oil_bottle_1'].values,
                     linestyle=line_style[i],
                     color=line_color[i],
                     label=name + ' ({:.1f}K)'.format(self.data.loc[self.bal_idx, name] - self.t_oil[self.bal_idx]))
        # except Exception:
        #     print("Plot {col_name} data error".format(col_name=name))

        # plot balance time marker
        plt.plot([self.bal_idx, self.bal_idx],
                 [self.data[col_name_list].max().max() - self.t_env[self.bal_idx] + 1,
                  self.data[col_name_list].max().min() - self.t_env[self.bal_idx] - 1],
                 color='k',
                 linewidth=0.5,
                 linestyle="--")

        # specify figure properties
        self.x_idx_list = [
            i * 360 for i in range(int(self.data.shape[0] / 360) + 1)]
        self.x_str_list = [self.datetime_to_xtick(
            self.data.iloc[i, -2]) for i in self.x_idx_list]
        plt.xticks(self.x_idx_list, self.x_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()
    
    def t_plot_gen(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves in °C
        Args:
        -------
            col_name_list   - list of column names to be plotted

        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature (°C)')

        # figure
        plt.figure(dpi=200)
        # sns.set(color_codes=True)

        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # plot curves as per "col_name_list"
        try:
            for i, name in enumerate(col_name_list):
                # self.interp_data_zero(name)
                plt.plot(self.data['snsr_time_index'].values,
                        self.data[name].values,
                        linewidth=0.5,
                        linestyle=line_style[i],
                        color=line_color[i],
                        label=name)
        except Exception:
            print("Plot {col_name} data error".format(col_name=name))


        # specify figure properties
        self.x_idx_list = [
            i * 360 for i in range(int(self.data.shape[0] / 360) + 1)]
        self.x_str_list = [self.datetime_to_xtick(
            self.data.iloc[i, -2]) for i in self.x_idx_list]
        plt.xticks(self.x_idx_list, self.x_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        
        # show
        plt.show()
        # plt.close()

    def tr_plot_logger(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature Rise')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')

        # figure
        fig,ax1 = plt.subplots(dpi=200)

        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # plot curves as per "col_name_list"
        t_oil_avg = (self.data['t_oil_bottle_1']
                     + self.data['t_oil_bottle_2']
                     + self.data['t_oil_bottle_3']) / 3
        for i, name in enumerate(col_name_list):

            plt.plot(self.data['cplr_index'].values,
                     self.data[name].values -
                     t_oil_avg.values,
                     linestyle=line_style[i],
                     color=line_color[i],
                     label=name + (' max TR is ' + '({:.1f}K)').format(max(self.data.loc[:,name].values-t_oil_avg.values)))
        # except Exception:
        #     print("Plot {col_name} data error".format(col_name=name))

        # plot steady-state marker
        if self.bal_idx != 0:
            [ymin,ymax] = plt.ylim()
            max_bal_tr = max([self.data.loc[self.bal_idx,i] - t_oil_avg[self.bal_idx] for i in col_name_list])
            # min_bal_tr = min([self.data.loc[self.bal_idx,i] - t_oil_avg[self.bal_idx] for i in col_name_list])
            plt.plot([self.bal_idx, self.bal_idx],
                     [max_bal_tr,ymin],
                     color='k',
                     linewidth=0.5,
                     linestyle="--")
            plt.text(self.bal_idx,
                     ymin+0.1*(ymax-ymin),
                     s='just stable point',
                     horizontalalignment='center')
            plt.ylim([ymin,ymax])

        # specify figure properties
        self.x_idx_list = [
            i * 360 for i in range(int(self.data.shape[0] / 360) + 1)]
        self.x_str_list = [self.datetime_to_xtick(
            self.data.iloc[i, -2]) for i in self.x_idx_list]
        plt.xticks(self.x_idx_list, self.x_str_list)

        ax2 = ax1.twinx()
        ax2.plot(self.data['cplr_index'].values,
                 self.current,
                 linestyle='--',
                 color='k',
                 label='Current')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax2.set_ylabel('Time Variant Current (A)')
        ax1.legend(fontsize=6,
                   loc="upper left")
        ax2.legend(fontsize=6)
        plt.grid(b=False)

        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()

    def t_dtr_plot(self, col_name, cur_list, time_const, conver_const, tr_warning, tr_alarm, delay_const, **kwargs):
        """Plotting dynamic temperature rise algorithm
        Args:
        -------
            col_name        - column name to be plotted
            cur_var_list    - list of current time-variant data points
            time_const      - time constant
            conver_const    - conversion constant
            delay_const     - delay constant
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', 'DTR Performance of {}'.format(col_name))
        x_label = kwargs.get('x_label', 'Sample Point')
        y_label = kwargs.get('y_label', 'Temperature (C)')
        sample_time = kwargs.get('sample_time', 10)
        tr_rated = kwargs.get('tr_rated', 0)
        correction_warning = kwargs.get('correction_warning', 8)
        correction_alarm = kwargs.get('correction_alarm', 10)
        time_const_drop = kwargs.get('time_const_drop', time_const)
        sample_num = int(1800 / sample_time)

        # figure
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        # sns.set(color_codes=True)

        # cut data as long as current data point list
        # when 'loc' command is used [a:b] include b 
        data_cut = self.data.loc[:len(cur_list) - 1, [col_name, 't_oil_bottle_1', 'snsr_datetime']].copy()
        # data_cut['tr'] = data_cut[col_name] - data_cut['t_oil_bottle_1']

        # build current time-variant data column
        data_cut['current'] = cur_list
        # data_cut.to_csv('C:\\Users\\cnbofan1\\Desktop\\output.csv')
        # print(data_cut.loc[1834,'current'])
        # build steady-state temperature column
        data_cut['tao_w_rated'] = [(ele / 630) ** conver_const * tr_rated + data_cut.loc[i, 't_oil_bottle_1']
                                   for i, ele in enumerate(cur_list)]
        data_cut['tao_w_warning'] = [(ele / 630) ** conver_const * tr_warning + data_cut.loc[i, 't_oil_bottle_1']
                                     for i, ele in enumerate(cur_list)]
        data_cut['tao_w_alarm'] = [(ele / 630) ** conver_const * tr_alarm + data_cut.loc[i, 't_oil_bottle_1']
                                   for i, ele in enumerate(cur_list)]
        # print(data_cut['tao_w'])

        # build temperature data structure
        t_rated_list = ([data_cut.loc[0, col_name]] +
                        [0] * (len(cur_list) - 1))
        t_warning_list = ([data_cut.loc[0, col_name]] +
                          [0] * (len(cur_list) - 1))
        t_alarm_list = ([data_cut.loc[0, col_name]] +
                        [0] * (len(cur_list) - 1))
        # use different time constant "time_const_drop" when current is dropping
        for k in range(1, len(cur_list)):
            if data_cut.loc[k, col_name] <= data_cut.loc[k, 'tao_w_warning']:
                t_rated_list[k] = t_rated_list[k - 1] + (data_cut.loc[k, 'tao_w_rated'] -
                                                         t_rated_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const))
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] -
                                                             t_warning_list[k - 1]) * (
                                            1 - np.exp(-1 * sample_time / time_const))
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] -
                                                         t_alarm_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const))
            elif data_cut.loc[k, col_name] > data_cut.loc[k, 'tao_w_warning']:
                t_rated_list[k] = t_rated_list[k - 1] + (data_cut.loc[k, 'tao_w_rated'] -
                                                         t_rated_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const_drop))
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] -
                                                             t_warning_list[k - 1]) * (
                                            1 - np.exp(-1 * sample_time / time_const_drop))
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] -
                                                         t_alarm_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const_drop))
            else:
                print('current value error at {}th element'.format(k))
        # print('tr_fit_list',tr_fit_list)
        if delay_const != 0:
            t_rated_list_not_delay = t_rated_list.copy()
            t_warning_list_not_delay = t_warning_list.copy()
            t_alarm_list_not_delay = t_alarm_list.copy()
            # prepend list with "delay_const" number of data points
            t_rated_list = [t_rated_list[0]] * delay_const + t_rated_list
            t_warning_list = [t_warning_list[0]] * delay_const + t_warning_list
            t_alarm_list = [t_alarm_list[0]] * delay_const + t_alarm_list
            # cut list the last "const" number of element from its tail
            del t_rated_list[-1 * delay_const:]
            del t_warning_list[-1 * delay_const:]
            del t_alarm_list[-1 * delay_const:]

            # add correction const to warning and alarm limits that were not delayed
            t_warning_list_not_delay = [t_warning_list_not_delay[i] + correction_warning for i in
                                        range(len(t_warning_list_not_delay))]
            t_alarm_list_not_delay = [t_alarm_list_not_delay[i] + correction_alarm for i in
                                      range(len(t_alarm_list_not_delay))]

        data_cut['t_rated'] = t_rated_list
        data_cut['t_warning'] = t_warning_list
        data_cut['t_warning'] = data_cut['t_warning'] + correction_warning
        data_cut['t_alarm'] = t_alarm_list
        data_cut['t_alarm'] = data_cut['t_alarm'] + correction_alarm

        for i in range(len(data_cut)):
            if 0 <= data_cut.loc[i, col_name] < data_cut.loc[i, 't_warning']:
                data_cut.loc[i, 't_si'] = 1
            elif data_cut.loc[i, 't_alarm'] > data_cut.loc[i, col_name] >= data_cut.loc[i, 't_warning']:
                # and if condition lasts more than 30min(1800s)
                if (i > sample_num) and ((
                        data_cut.loc[i - sample_num:i, col_name] >= data_cut.loc[i - sample_num:i,
                                                                    't_warning']).all()):
                    data_cut.loc[i, 't_si'] = 2
                else:
                    data_cut.loc[i, 't_si'] = 1

            elif data_cut.loc[i, col_name] > data_cut.loc[i, 't_alarm']:
                if (i > sample_num) and (
                        (data_cut.loc[i - sample_num:i, col_name] > data_cut.loc[i - sample_num:i,
                                                                    't_alarm']).all()):
                    data_cut.loc[i, 't_si'] = 3
                elif (i > sample_num) and ((
                        data_cut.loc[i - sample_num:i, col_name] >= data_cut.loc[i - sample_num:i,
                                                                    't_warning']).all()):
                    data_cut.loc[i, 't_si'] = 2
                else:
                    data_cut.loc[i, 't_si'] = 1
            else:
                data_cut.loc[i, 't_si'] = 0
        # idx_delay = np.array(data_cut.index) + delay_const

        # generate warning, alarm, signal indicator data to mrc
        output = data_cut.loc[:,
                 ['t_warning', 't_alarm', 't_si', 'current', col_name, 'snsr_datetime', 't_rated']].copy()

        plt.scatter(data_cut.index,
                    data_cut['t_rated'],
                    s=5,
                    color='c',
                    marker='x',
                    label='t_rated_scatter')
        plt.plot(data_cut.index,
                 data_cut['t_rated'],
                 color='c',
                 label='t_rated_plot')
        plt.plot(data_cut.index,
                 data_cut[col_name],
                 color='g',
                 label=col_name)
        plt.plot(data_cut.index,
                 data_cut['t_warning'],
                 color='y',
                 label='t_warning')
        plt.plot(data_cut.index,
                 data_cut['t_alarm'],
                 color='r',
                 label='t_alarm')
        if delay_const != 0:
            plt.plot(data_cut.index,
                     t_warning_list_not_delay,
                     color='y',
                     linestyle=':',
                     label='t_warning_not_delay')
            plt.plot(data_cut.index,
                     t_alarm_list_not_delay,
                     color='r',
                     linestyle=':',
                     label='t_alarm_not_delay')
            plt.plot(data_cut.index,
                     t_rated_list_not_delay,
                     color='c',
                     linestyle=':',
                     label='t_rated_not_delay')
        ax2 = ax1.twinx()
        ax2.plot(data_cut.index,
                 data_cut['current'],
                 label='current',
                 linestyle=':',
                 color='k')

        xdata = [datetime.strftime(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'), '%H:%M') for i in
                 data_cut['snsr_datetime']]
        plt.xticks(data_cut.index[::math.floor(3600 / sample_time)], xdata[::math.floor(3600 / sample_time)])
        plt.title(title)
        plt.xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax2.set_ylabel('Time Variant Current (A)')
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)

        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        return output

    def t_dtr_plot_inst_alarm(self, col_name, cur_list, time_const, conver_const, tr_warning, tr_alarm, delay_const, **kwargs):
        """Plotting dynamic temperature rise algorithm
        Args:
        -------
            col_name        - column name to be plotted
            cur_var_list    - list of current time-variant data points
            time_const      - time constant
            conver_const    - conversion constant
            delay_const     - delay constant
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', 'DTR Performance of {}'.format(col_name))
        x_label = kwargs.get('x_label', 'Sample Point')
        y_label = kwargs.get('y_label', 'Temperature (C)')
        sample_time = kwargs.get('sample_time', 10)
        tr_rated = kwargs.get('tr_rated', 0)
        correction_warning = kwargs.get('correction_warning', 8)
        correction_alarm = kwargs.get('correction_alarm', 10)
        time_const_drop = kwargs.get('time_const_drop', time_const)
        sample_num = int(1800 / sample_time)

        # figure
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        # sns.set(color_codes=True)

        # cut data as long as current data point list
        # when 'loc' command is used [a:b] include b
        data_cut = self.data.loc[:len(cur_list) - 1, [col_name, 't_oil_bottle_1', 'snsr_datetime']].copy()
        # data_cut['tr'] = data_cut[col_name] - data_cut['t_oil_bottle_1']

        # build current time-variant data column
        data_cut['current'] = cur_list
        # data_cut.to_csv('C:\\Users\\cnbofan1\\Desktop\\output.csv')
        # print(data_cut.loc[1834,'current'])
        # build steady-state temperature column
        data_cut['tao_w_rated'] = [(ele / 630) ** conver_const * tr_rated + data_cut.loc[i, 't_oil_bottle_1']
                                   for i, ele in enumerate(cur_list)]
        data_cut['tao_w_warning'] = [(ele / 630) ** conver_const * tr_warning + data_cut.loc[i, 't_oil_bottle_1']
                                     for i, ele in enumerate(cur_list)]
        data_cut['tao_w_alarm'] = [(ele / 630) ** conver_const * tr_alarm + data_cut.loc[i, 't_oil_bottle_1']
                                   for i, ele in enumerate(cur_list)]
        # print(data_cut['tao_w'])

        # build temperature data structure
        t_rated_list = ([data_cut.loc[0, col_name]] +
                        [0] * (len(cur_list) - 1))
        t_warning_list = ([data_cut.loc[0, col_name]] +
                          [0] * (len(cur_list) - 1))
        t_alarm_list = ([data_cut.loc[0, col_name]] +
                        [0] * (len(cur_list) - 1))
        # use different time constant "time_const_drop" when current measured temperature greater than 
        # balanced warning temperature based on present current levels
        for k in range(1, len(cur_list)):
            if data_cut.loc[k, col_name] <= data_cut.loc[k, 'tao_w_warning']:
                t_rated_list[k] = t_rated_list[k - 1] + (data_cut.loc[k, 'tao_w_rated'] -
                                                         t_rated_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const))
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] -
                                                             t_warning_list[k - 1]) * (
                                            1 - np.exp(-1 * sample_time / time_const))
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] -
                                                         t_alarm_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const))
            elif data_cut.loc[k, col_name] > data_cut.loc[k, 'tao_w_warning']:
                t_rated_list[k] = t_rated_list[k - 1] + (data_cut.loc[k, 'tao_w_rated'] -
                                                         t_rated_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const_drop))
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] -
                                                             t_warning_list[k - 1]) * (
                                            1 - np.exp(-1 * sample_time / time_const_drop))
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] -
                                                         t_alarm_list[k - 1]) * (
                                          1 - np.exp(-1 * sample_time / time_const_drop))
            else:
                print('current value error at {}th element'.format(k))
        # print('tr_fit_list',tr_fit_list)
        if delay_const != 0:
            t_rated_list_not_delay = t_rated_list.copy()
            t_warning_list_not_delay = t_warning_list.copy()
            t_alarm_list_not_delay = t_alarm_list.copy()
            # prepend list with "delay_const" number of data points
            t_rated_list = [t_rated_list[0]] * delay_const + t_rated_list
            t_warning_list = [t_warning_list[0]] * delay_const + t_warning_list
            t_alarm_list = [t_alarm_list[0]] * delay_const + t_alarm_list
            # cut list the last "const" number of element from its tail
            del t_rated_list[-1 * delay_const:]
            del t_warning_list[-1 * delay_const:]
            del t_alarm_list[-1 * delay_const:]

            # add correction const to warning and alarm limits that were not delayed
            t_warning_list_not_delay = [t_warning_list_not_delay[i] + correction_warning for i in
                                        range(len(t_warning_list_not_delay))]
            t_alarm_list_not_delay = [t_alarm_list_not_delay[i] + correction_alarm for i in
                                      range(len(t_alarm_list_not_delay))]

        data_cut['t_rated'] = t_rated_list
        data_cut['t_warning'] = t_warning_list
        data_cut['t_warning'] = data_cut['t_warning'] + correction_warning
        data_cut['t_alarm'] = t_alarm_list
        data_cut['t_alarm'] = data_cut['t_alarm'] + correction_alarm

        for i in range(len(data_cut)):
            if 0 <= data_cut.loc[i, col_name] < data_cut.loc[i, 't_warning']:
                data_cut.loc[i, 't_si'] = 1

            elif data_cut.loc[i, 't_alarm'] > data_cut.loc[i, col_name] >= data_cut.loc[i, 't_warning']:
                # and if condition lasts more than 30min(1800s)
                data_cut.loc[i, 't_si'] = 2

            elif data_cut.loc[i, col_name] > data_cut.loc[i, 't_alarm']:
                data_cut.loc[i, 't_si'] = 3

            else:
                data_cut.loc[i, 't_si'] = 0
        # idx_delay = np.array(data_cut.index) + delay_const

        # generate warning, alarm, signal indicator data to mrc
        output = data_cut.loc[:,
                 ['t_warning', 't_alarm', 't_si', 'current', col_name, 'snsr_datetime', 't_rated']].copy()

        plt.scatter(data_cut.index,
                    data_cut['t_rated'],
                    s=5,
                    color='c',
                    marker='x',
                    label='t_rated_scatter')
        plt.plot(data_cut.index,
                 data_cut['t_rated'],
                 color='c',
                 label='t_rated_plot')
        plt.plot(data_cut.index,
                 data_cut[col_name],
                 color='g',
                 label=col_name)
        plt.plot(data_cut.index,
                 data_cut['t_warning'],
                 color='y',
                 label='t_warning')
        plt.plot(data_cut.index,
                 data_cut['t_alarm'],
                 color='r',
                 label='t_alarm')
        if delay_const != 0:
            plt.plot(data_cut.index,
                     t_warning_list_not_delay,
                     color='y',
                     linestyle=':',
                     label='t_warning_not_delay')
            plt.plot(data_cut.index,
                     t_alarm_list_not_delay,
                     color='r',
                     linestyle=':',
                     label='t_alarm_not_delay')
            plt.plot(data_cut.index,
                     t_rated_list_not_delay,
                     color='c',
                     linestyle=':',
                     label='t_rated_not_delay')
        ax2 = ax1.twinx()
        ax2.plot(data_cut.index,
                 data_cut['current'],
                 label='current',
                 linestyle=':',
                 color='k')

        xdata = [datetime.strftime(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'), '%H:%M') for i in
                 data_cut['snsr_datetime']]
        plt.xticks(data_cut.index[::math.floor(3600 / sample_time)], xdata[::math.floor(3600 / sample_time)])
        plt.title(title)
        plt.xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax2.set_ylabel('Time Variant Current (A)')
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)

        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        return output
    def scatter_hist_plot(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    def imbalance_plot_logger(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # keywords arguments list
        title = kwargs.get('title', '3-phase Imbalance Temperature')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Imbalance Temperature(K)')

        # figure
        fig,ax1 = plt.subplots(dpi=200)

        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'r', 'g'])

        # plot curves as per "col_name_list"
        imb_list = [max([self.data.loc[i,j] for j in col_name_list]) -
                    min([self.data.loc[i,j] for j in col_name_list]) for i in range(len(self.data))]

        plt.plot(self.data['cplr_index'].values,
                 imb_list,
                 color='k',
                 label= 'max imbalance temperature is' + ' ({:.1f}K)'.format(max(imb_list)))

        # specify figure properties
        self.x_idx_list = [
            i * 360 for i in range(int(self.data.shape[0] / 360) + 1)]
        self.x_str_list = [self.datetime_to_xtick(
            self.data.iloc[i, -2]) for i in self.x_idx_list]
        plt.xticks(self.x_idx_list, self.x_str_list)

        ax2 = ax1.twinx()
        ax2.plot(self.data['cplr_index'].values,
                 self.cur_phA,
                 linestyle='--',
                 color='b',
                 label='PhA Current')
        ax2.plot(self.data['cplr_index'].values,
                 self.cur_phB,
                 linestyle='--',
                 color='r',
                 label='PhB Current')
        ax2.plot(self.data['cplr_index'].values,
                 self.cur_phC,
                 linestyle='--',
                 color='g',
                 label='PhC Current')
        plt.title(title)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax2.set_ylabel('Time Variant Current (A)')
        ax1.legend(fontsize=6,
                   loc='center right')
        ax2.legend(fontsize=6,
                   loc="lower right")
        plt.grid(b=False)

        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()

    def datetime_to_xtick(self, datetime_str):
        datetime_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return datetime_dt.strftime('%H:%M')

    def find_balance_index(self, col_num_list):
        """method that find the temperature balancing point of channels assigned
        Args:
        -------
            col_num_list   - list of column numbers to be checked

        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # build sliced data of interest

        data_sliced = self.data.iloc[:, col_num_list].copy()
        # mask = [self.data.columns[i] for i in col_num_list]
        # print(mask)
        for i in range(len(col_num_list)):
            # print('i=',i,'type of i',type(i))
            data_sliced.iloc[:, i] = data_sliced.iloc[:, i] - self.t_oil
        try:
            for k in range(1440, len(data_sliced)):
                data_sliced_diff = (
                        data_sliced.iloc[k, :] - data_sliced.iloc[k - 360, :]).abs()

                if (data_sliced_diff < 1.0).all(axis=None):
                    self.bal_idx = k
                    print('Temperature balance time is {time}.'.format(
                        time=self.data.iloc[k, -2]))
                    break
                elif k == len(data_sliced) - 1:
                    self.bal_idx = k
                    print('Temperature is not balanced.')
                    break
        except Exception:
            print("Find balance point of {name} data error".format(
                name=col_num_list))

    def cal_dynamic_time_const(self, col_name, **kwargs):
        """method that calculate the DTR time constant of one data column of given name;
           compare the fitted curve with scatter points
        Args:
        -------
            t_amb      - ambient temperature(t_env/t_oil) 

        Return:
        -------
            const     - time constant calculated   
        Notes:
        -------
            NA,
        """
        # load steady-state rated temperature rise t0
        t_amb = kwargs.get('t_amb', 't_env')
        if t_amb == 't_oil':

            tw = kwargs.get('tw',
                            self.data.loc[self.bal_idx, col_name] -
                            self.t_oil.loc[self.bal_idx])
            ydata = (self.data[col_name] - self.t_oil).copy().values
        else:
            tw = kwargs.get('tw',
                            self.data.loc[self.bal_idx, col_name] -
                            self.t_env.loc[self.bal_idx])
            ydata = (self.data[col_name] - self.t_env).copy().values
            # xdata = self.data['snsr_timeindex'].copy().values
        xdata = np.array(range(len(ydata)))

        # cut ydata >= 0
        # ydata_pos = ydata[ydata >= 0]
        # print(xdata,ydata)

        # calculate initial temperature rise
        t0 = ydata[0]

        # print(xdata,ydata,t0,tw)
        # create function to be fitted
        def f_transient_tr(x, T):
            return tw * (1 - np.exp(-1 * x * 10 / T)) + t0 * np.exp(-1 * x * 10 / T)

        # fitting
        const = optimize.curve_fit(f_transient_tr, xdata, ydata)[0][0]
        print('time constant T of {} is'.format(col_name), const)

        # plt.figure(dpi=500, figsize=(10,6))
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        # sns.set(color_codes=True)
        ax2 = ax1.twinx()
        rmse = self.get_rmse(f_transient_tr(xdata, const),
                             ydata)

        # create scatter plot of testing data
        ax1.scatter(xdata, ydata, color='r', label='temperature rise tested')

        # create line plot of fitted data with time const calculated from fitting
        ax1.plot(xdata,
                 f_transient_tr(xdata, const),
                 color='g',
                 label='temperature rise fitted')

        # create line plot of difference data
        ax2.plot(xdata,
                 f_transient_tr(xdata, const) - ydata,
                 color='k',
                 label='difference with RMSE of {:.1f}'.format(rmse))
        ax1.set_title('{} time constant is {:.0f}'.format(col_name, const))
        ax1.set_xlabel('sample point')
        ax1.set_ylabel('Temperature Rise (K)')
        ax2.set_ylabel('Difference (K)')
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)
        self.x_idx_list = [
            i * 360 for i in range(int(self.data.shape[0] / 360) + 1)]
        # print('viewer',self.data.iloc[0, -2])
        self.x_str_list = [self.datetime_to_xtick(
            self.data.iloc[i, -2]) for i in self.x_idx_list]
        plt.xticks(self.x_idx_list, self.x_str_list)
        # return const

    @staticmethod
    def cal_dynamic_conver_const(xdata, ydata, **kwargs):
        """method that calculate the DTR conversion constant of one data set;

        Args:
        -------
            xdata   - x data to be fitted
            ydata   - y data to be fitted

        Return:
        -------
            const   - conversion constant calculated   
        Notes:
        -------
            the last data point must be the TR rated (630A)
        """
        # initialize data
        title = kwargs.get('title',
                           'Fitting for DTR conversion constant')
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        tr_rated = ydata[-1]

        # create function to be fitted
        def f_steady_state_tr(x, a):
            return (x / 630) ** a * tr_rated

        # fitting
        const = optimize.curve_fit(f_steady_state_tr,
                                   xdata,
                                   ydata)[0][0]
        print('conversion constant a is',
              const)
        plt.figure(dpi=200)
        sns.set(color_codes=True)
        # create scatter plot of input data point
        plt.scatter(xdata,
                    ydata,
                    color='r')
        # create line plot of fitted curve
        x_span = np.array(range(min(xdata),
                                max(xdata) + 1,
                                10))
        plt.plot(x_span,
                 f_steady_state_tr(x_span, const),
                 color='g',
                 label='Conversion Constant is {:.2f}'.format(const))
        plt.xlabel('current (A)')
        plt.ylabel('Temperature Rise (K)')
        plt.legend()
        plt.title(title)
        # return const

    @staticmethod
    def get_rmse(predictions, targets):

        return np.sqrt(((predictions - targets) ** 2).mean())

    @staticmethod
    def get_max_abs(predictions, targets):

        return max(np.abs(predictions - targets))

    def data_filter(self, col_name, low_lim, high_lim, diff_lim):
        """method that find out point that beyond hard limit and point with instant change;

        Args:
        -------
            col_name     - colunm data to be filtered
            low_lim      - low hard limit
            high_lim     - high hard limit
            diff_lim     - changing limit


        Return:
        -------
            NA   
        Notes:
        -------
        """
        diff = np.diff(self.data[col_name].values, prepend=self.data.loc[0, col_name])
        for j in range(len(self.data)):
            if ((self.data.loc[j, col_name] > high_lim) or
                    (self.data.loc[j, col_name] < low_lim)):
                self.data.loc[j, col_name] = np.nan

            else:
                pass
        for k in range(len(self.data) - 1):
            if (((diff[k] > diff_lim) or
                 (diff[k] < -diff_lim)) &
                    ((diff[k + 1] > diff_lim) or
                     (diff[k + 1] < -diff_lim))):
                self.data.loc[k, col_name] = np.nan


class DataClean(object):
    
    def read_sensor_data(path):
        """discontinued"""
        """method that read the data from intelligent temperature sensor,
           intelligent gas pressure sensor and intelligent temperature 
           and humidity sensor;

        Args:
        -------
            path                - full path of the data file

        Return:
        -------
            raw_data_sliced     - raw data cutted from starting index to ending index
        Notes:
        -------

        """
        raw_data = pd.read_csv(path, header=None)
        # convert string data to float
        raw_data.iloc[:, 2:].astype(float)
        # add new column of datetime
        raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i, 0] + ':' + raw_data.iloc[i, 1],
                                                  '%m/%d/%Y:%I:%M:%S %p') for i in range(len(raw_data))]
        
        # search test start index
        for i in range(len(raw_data)):
            if (raw_data.iloc[i, 2] != 0 or
                    raw_data.iloc[i, 3] != 0 or
                    raw_data.iloc[i, 4] != 0 or
                    raw_data.iloc[i, 5] != 0 or
                    raw_data.iloc[i, 6] != 0 or
                    raw_data.iloc[i, 7] != 0 or
                    raw_data.iloc[i, 8] != 0 or
                    raw_data.iloc[i, 9] != 0 or
                    raw_data.iloc[i, 10] != 0):
                t0 = i
                print('test started from %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                print('test data was all "0"')
            else:
                continue

        # search test end index from 1 hour after test started
        for i in range(t0 + 360, len(raw_data)):
            if (raw_data.iloc[i, 2] == 0 and raw_data.iloc[i, 3] == 0 and raw_data.iloc[i, 4]
                    == 0 and raw_data.iloc[i, 5] == 0 and raw_data.iloc[i, 6] == 0 and raw_data.iloc[i, 7] == 0
                    and raw_data.iloc[i, 8] == 0 and raw_data.iloc[i, 9] == 0 and raw_data.iloc[i, 10] == 0):
                tn = i
                print('test ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                tn = i + 1
                print('test data are not fully recorded')
            else:
                continue

        # check if the sample time is 10s, if not down-sample to 10s
        time_interval = (raw_data.iloc[1, -1] - raw_data.iloc[0, -1]).seconds
        if time_interval != 10:
            multiple = 10 / time_interval
            raw_data = raw_data[raw_data.index % multiple == 0]
        else:
            pass

        # slice data from start index to end index
        raw_data_sliced = raw_data.iloc[t0:tn, :]
        raw_data_sliced.index = range(len(raw_data_sliced))
        return raw_data_sliced

    def read_logger_data(path):
        """method that read the data sampled by TR and GP data logger,
           intelligent gas pressure sensor and intelligent temperature;

        Args:
        -------
            path                - full path of the data file

        Return:
        -------
            raw_data_sliced     - raw data cutted from starting index to ending index
        Notes:
        -------

        """
        raw_data = pd.read_csv(path, header=0)
        raw_data = raw_data.dropna()
        # convert string data to float
        raw_data.iloc[:, 1:].astype(float)
        print(raw_data['Time'])
        # convert str to datetime
        raw_data['Time'] = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y-%m-%d %H:%M:%S') for i in range(len(raw_data))]
        raw_data = raw_data.rename(columns={'Time':'datetime'})
  
        # search test start index
        for i in range(len(raw_data)):
            if (raw_data.iloc[i, 3:21] == 0).all():
                if i == (len(raw_data) - 1):
                    print('test data was all "0"')
                else:
                    pass
            else:
                t0 = i
                print('logger started from %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break


        # search test end index from 1 hour after test started
        for i in range(t0 + 360, len(raw_data)):
            if ((raw_data.iloc[i:i+10, 3:21] == 0).all()).all():
                tn = i
                print('logger ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                tn = i + 1
                print('test data are not fully recorded')
            else:
                continue

        # slice data from start index to end index
        raw_data_sliced = raw_data.iloc[t0:tn, :]
        raw_data_sliced.index = range(len(raw_data_sliced))
 
        return raw_data_sliced
    
    def read_logger_data_DTR_PD(path, **kwargs):
        """method that read the data sampled by TR,GP,DTR,PD data logger,
        Args:
            path                - full path of the data file
        Return:
            raw_data_sliced     - raw data cutted from starting index to ending index
        Notes:
        """
        start_ix = 0
        raw_data = pd.read_csv(path, header=0)
        raw_data.iloc[:, 1:].astype(float)
        raw_data = raw_data.fillna(0)

        format = kwargs.get('format', '%Y-%m-%d %H:%M:%S')
            
        # convert str to datetime
        raw_data.index = [datetime.strptime(raw_data.loc[i, 'Time'], format) for i in range(len(raw_data))]
        # print(raw_data)
        # search test start index
        for i in range(len(raw_data)):
            if (raw_data.iloc[i, 3:9] == 0).all():
                if i == (len(raw_data) - 1):
                    print('test data was all "0"')
                else:
                    pass
            else:
                print('logger started from %s' %
                    (raw_data.index[i].strftime("%H:%M:%S")))
                start_ix = i
                break
        data_cut = raw_data.iloc[i:,:]
        return data_cut

    def read_logger_data_GP(path):
        """method that read the gas pressure and humidity data
           sampled by data logger, not to find starting point
        Args:
            path                - full path of the data file
        Return:
            raw_data_sliced     - raw data cutted from starting index to ending index
        Notes:
        """
        raw_data = pd.read_csv(path, header=0)
        raw_data = raw_data.dropna()

        # convert string data to float
        output = raw_data.iloc[:, 1:].astype(float)

        # convert str to datetime
        output.index = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y-%m-%d %H:%M:%S') for i in
                          range(len(raw_data))]

        return output

    def read_logger_data_simple(path):
        """simple method that only read logger data without finding starting time,


        Args:
            path       - full path of the data file
        Return:
            output     - data read into dataframe
        Notes:
            updated on 20230406
        """
        raw_data = pd.read_csv(path, header=0)
        raw_data = raw_data.dropna()

        # convert string data to float
        output = raw_data.iloc[:, 1:].astype(float)

        # convert str to datetime
        output.index = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y-%m-%d %H:%M:%S') for i in
                          range(len(raw_data))]

        return output

    def read_coupler_data(path):
        """method that read the data from thermalcouples;

        Args:
        -------
            path        - full path of the data file

        Return:
        -------
            raw_data    - raw data read
        Notes:
        -------

        """
        raw_data = pd.read_csv(path,
                               header=25,
                               na_values=['          ', '     -OVER','   INVALID', '     +OVER'])
        # print(raw_data)
        print(raw_data.columns)
        raw_data.iloc[:, 4:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i, 1] + ':' + raw_data.iloc[i, 2],
                                                  '%Y/%m/%d:%H:%M:%S') for i in range(len(raw_data.iloc[:, 2]))]


        return raw_data

    def read_couple_datetime(path):
        """method that read the data from thermal couples with index of datetime;

        Args:

            path      -     full path of the data file

        Return:

            output    -     data saved as Dataframe
        Notes:


        """
        raw_data = pd.read_csv(path,
                               header=25,
                               na_values=['          ', '     -OVER','   INVALID', '     +OVER'])

        
        raw_data.iloc[:, 4:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data.index = [datetime.strptime(raw_data.iloc[i, 1] + ':' + raw_data.iloc[i, 2],
                                                  '%Y/%m/%d:%H:%M:%S') for i in range(len(raw_data.iloc[:, 2]))]
        # print(raw_data.columns)
        return raw_data

    def read_couple_flex_format(path, **kwargs):
        """method that read the data from thermal couples with index of datetime 
           in a flexible datetime format;

        Args:

            path      -     full path of the data file
            format    -     format of datetime
        Return:

            output    -     data saved as Dataframe
        Notes:


        """
        format = kwargs.get('format', '%m/%d/%Y:%H:%M:%S')
        raw_data = pd.read_csv(path,
                               header=25,
                               na_values=['          ', '     -OVER','   INVALID'])

        
        raw_data.iloc[:, 4:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data.index = [datetime.strptime(raw_data.iloc[i, 1] + ':' + raw_data.iloc[i, 2],
                                                  format) for i in range(len(raw_data.iloc[:, 2]))]
        # print(raw_data.columns)
        return raw_data

    def read_gaoli_logger_data(path):
        """method that read the data measured by direct temp sensors and 
           recorded by data logger developed by Gaoli;

        Args:
            path                - full path of the data file

        Return:
            raw_data_sliced     - raw data cutted from starting index to ending index
        Notes:
            default sampling time interval is 2s, need to be downsampled to 10s 

        """
        raw_data = pd.read_csv(path, header=0)
        print(raw_data.columns)

        raw_data = DataClean.down_sample(raw_data,5)
        # convert string data to float
        raw_data.loc[:, ['GL1A', 'GL1B', 'GL1C']].astype(float)
        print(raw_data['Time'])
        # convert str to datetime
        raw_data['datetime_direct'] = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y%m%d %H:%M:%S') for i in range(len(raw_data))]
        
        # search test start index: 
        for i in range(len(raw_data)):
            print(i)
            if (raw_data.loc[i, ['GL1A','GL1B','GL1C']] == 0).all():
                if i == (len(raw_data) - 1):
                    print('Gaoli logger data was all "0"')
                else:
                    pass
                
            else:
                t0 = i
                print('Gaoli logger started from %s' %
                      (raw_data.loc[i, 'datetime_direct'].strftime("%H:%M:%S")))
                break

        # search test end index from 1 hour after test started
        for i in range(t0 + 360, len(raw_data)):
            if ((raw_data.loc[i:i+10, ['GL1A','GL1B','GL1C']] == 0).all()).all():
                tn = i
                print('logger ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                tn = i + 1
                print('test data are not fully recorded')
            else:
                continue
     
        # slice data from start index to end index
        raw_data_sliced = raw_data.iloc[t0:tn, :]
        raw_data_sliced.index = range(len(raw_data_sliced))
 
        return raw_data_sliced

    def synch_data_group(*data):
        """method that synchronize groups of data;

        Args:
        -------
            data        - groups of data to be synchronized

        Return:
        -------
            data_list   - synchronized list of data groups 
        Notes:
        -------

        """
        # search the common starting time
        start_time = max([data[i].iloc[0, -1]
                          for i in range(len(data))])
        # search the common ending time
        end_time = min([data[i].iloc[-1, -1]
                        for i in range(len(data))])
        data_list = []
        print('sensor & couplers common start time = ',
              start_time)
        print('sensor & couplers common end time =',
              end_time)
        j = 0
        count = 0
        # perform on each group of data
        for element in data:
            j += 1
            # add one column of index relative to the common starting time
            element['ind'] = [math.ceil(
                (element.iloc[i, -1] - start_time).seconds / 10) for i in range(len(element))]
            element = element.loc[(element['ind'] >= 0)
                                  & (element['ind'] <= 4320), :]

            # check and fix that if there are any duplicated index
            k = 0
            while (k < len(element) - 1):
                # print('len element', len(element))
                # print('k', k)
                if element.iloc[k, -1] == element.iloc[k + 1, -1]:
                    element = element.drop(k + 1)
                    count += 1
                else:
                    pass
                k = k + 1
            print('count', count)
            # check and fix that if there are any discontinued index
            if (element.iloc[-1, -1] - element.iloc[0, -1]) != (len(element) - 1):
                print('%dth group of data has discontinued points' % j)
                # print('head,tail and length are ',
                #       element.loc[0, 'ind'],
                #       element.loc[-1, 'ind'],
                #       len(element))
                # build the full length index 
                full_index = pd.Series(
                    range(element.iloc[0, -1], element.iloc[-1, -1] + 1))
                # find the discontinued index    
                miss_index = full_index.loc[~full_index.isin(
                    list(element['ind']))]
                element_full = pd.DataFrame(data=0, columns=element.columns, index=range(len(full_index)))

                # fill in data 
                for i in range(len(element)):
                    element_full.iloc[element.iloc[i, -1] -
                                      element.iloc[0, -1], :] = element.iloc[i, :]

                # fill in with missed index
                for i in list(miss_index):
                    element_full.iloc[i - element.iloc[0, -1], -1] = i
                data_list.append(element_full)
            else:
                print('%dth group of data has no discontinued points' % j)
                data_list.append(element)
                pass
        # 重新
        # 找到三组数据重叠数据点的起止索引
        start_ind = max([i.iloc[0, -1] for i in data_list])
        end_ind = min([i.iloc[-1, -1] for i in data_list])
        # print('sensor & couplers common end time = ', data_list[0].loc[data_list[0].loc,-2])

        # 用共同起止索引截取三组数据重叠的数据点
        for n, ele in enumerate(data_list):
            data_list[n] = ele.loc[ele.iloc[:, -1] <= end_ind, :]

            data_list[n] = data_list[n].loc[data_list[n].iloc[:, -1]
                                            >= start_ind, :]

            data_list[n].index = range(len(data_list[n]))

        return data_list

    def synch_logger_couple(*data):
        """method that synchronize data of data_logger with groups of data of thermocouples;

        Args:
        -------
            data_logger        - data of data_logger
            data               - list of data of thermocouples

        Return:
        -------
            data_list          - synchronized list of data groups
        Notes:
        -------

        """
        # search the common starting time
        start_time = max([i.loc[i.index[0], 'datetime']
                          for i in data])
        # search the common ending time
        end_time = min([i.loc[i.index[-1], 'datetime']
                          for i in data])
        data_list = []
        print('sensor & couplers common start time = ',
              start_time)
        print('sensor & couplers common end time =',
              end_time)

        count = 0

        # perform on each group of data
        for j,element in enumerate(data):
            # add one column of index relative to the common starting time
            element['idx'] = [math.ceil(
                (element.loc[element.index[i], 'datetime'] - start_time).seconds / 10) for i in range(len(element))]

            element = element.loc[(element['idx'] >= 0)
                                  & (element['idx'] <= 4320), :]

            # check and fix that if there are any duplicated index
            k = 0
            while (k < len(element) - 1):
                # print('len element', len(element))
                if element.loc[element.index[k], 'idx'] == element.loc[element.index[k + 1], 'idx']:
                    element = element.drop(k + 1)
                    count += 1
                    k = k + 1
                else:
                    pass
                k = k + 1
            print('number of duplicated index is : ', count)

            # check and fix that if there are any discontinued index
            if (element.loc[element.index[-1], 'idx'] - element.loc[element.index[0], 'idx']) != (len(element) - 1):
                print('%dth group of data has discontinued points' % j)
                # print('head,tail and length are ',
                #       element.loc[0, 'ind'],
                #       element.loc[-1, 'ind'],
                #       len(element))

                # build the full length index
                full_index = pd.Series(
                    range(element.loc[element.index[0], 'idx'], element.loc[element.index[-1], 'idx'] + 1))
                # find the discontinued index
                miss_index = full_index.loc[~full_index.isin(list(element.idx))]
                element_full = pd.DataFrame(data=0, columns=element.columns, index=range(len(full_index)))

                # fill in data
                for m in range(len(element)):
                    element_full.iloc[element.loc[element.index[m], 'idx'] -
                                      element.loc[element.index[0], 'idx'], :] = element.iloc[m, :]

                # fill in with missed index
                for n in range(1,len(element_full)):
                    if element_full.loc[element_full.index[n],'idx'] == 0:
                        element_full.loc[element_full.index[n], 'idx'] = element_full.loc[element_full.index[n-1], 'idx'] + 1
                data_list.append(element_full)
            else:
                print('%dth group of data has no discontinued points' % j)
                data_list.append(element)
                pass

        # Find the common start and ending index of t
        start_ind = max([i.loc[i.index[0],'idx'] for i in data_list])
        end_ind = min([i.loc[i.index[-1],'idx'] for i in data_list])
        print('sensor & couplers common start index :',start_ind)
        print('sensor & couplers common end index :',end_ind)

        # 用共同起止索引截取三组数据重叠的数据点
        for n, ele in enumerate(data_list):
            data_list[n] = ele.loc[ele.loc[:, 'idx'] <= end_ind, :]

            data_list[n] = data_list[n].loc[data_list[n].loc[:, 'idx']
                                            >= start_ind, :]

            data_list[n].index = range(len(data_list[n]))

        return data_list

    def synch_logger_couple_resample(logger_df, couple_df, sample_time):
        """method that synchronize data of data_logger with groups of data of thermal couples;

        Args:
        -------
            logger_data_df - data of data_logger in DataFrame
            couple_data_df - data of thermal couples in DataFrame
            sample_time    - string, default value is '1min', refer to np.resample settings

        Return:
        -------
            output         - synchronized list of data groups
        Notes:
        -------

        """
        # resampling logger and couple data with an interval of 1min
        logger_df_resample = logger_df.resample(sample_time).median()
        couple_df_resample = couple_df.resample(sample_time).median()
        # print('logger_df', logger_df_resample)
        # print('couple_df', couple_df_resample)
        # concatenate logger data with thermal couples data for rows with same index
        sync_df = pd.concat((logger_df_resample, couple_df_resample), axis=1, join='inner')
        # print(sync_df)
        # Find the common start and ending time of t
        start_time = sync_df.index[0]
        end_time = sync_df.index[-1]
        print('MDC4-M & thermal couples common start datetime :',start_time)
        print('MDC4-M & thermal couples common end datetime :',end_time)
    
        return sync_df

    # def synch_logger_couple_5min(logger_df, couple_df):
    #     """method that synchronize data of data_logger with groups of data of thermal couples;

    #     Args:
    #     -------
    #         logger_data_df - data of data_logger in DataFrame
    #         couple_data_df - data of thermal couples in DataFrame

    #     Return:
    #     -------
    #         output         - synchronized list of data groups
    #     Notes:
    #     -------

    #     """

    #     # resampling logger and couple data with an interval of 1min
    #     logger_df_1min = logger_df.resample('5min').median()
    #     couple_df_1min = couple_df.resample('5min').median()
        
    #     # concatenate logger data with thermal couples data for rows with same index
    #     sync_df = pd.concat((logger_df_1min, couple_df_1min), axis=1, join='inner')
        
    #     # Find the common start and ending time of t
    #     start_time = sync_df.index[0]
    #     end_time = sync_df.index[-1]
    #     print('MDC4-M & thermal couples common start datetime :',start_time)
    #     print('MDC4-M & thermal couples common end datetime :',end_time)
    
    #     return sync_df

    @staticmethod
    def down_sample(data, ratio):
        index = np.arange(len(data))
        data_out = data[index % ratio == 0]
        return data_out

    @staticmethod
    def datetime_to_xtick(datetime_str):
        datetime_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return datetime_dt.strftime('%H:%M:%S')

    @staticmethod
    def datetime_to_time(datetime_str):
        date_time_dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return date_time_dt.strftime('%H:%M:%S')

    def data_clean_resample(cur_dir, test_date, test_id, **kwargs):
        sample_time = kwargs.get('sample_time', '1min') # get sample_time data
        datetime_format = kwargs.get('datetime_format', '%m/%d/%Y:%H:%M:%S') # get datetime format
        test_folder = test_date + test_id  # test folder
        folder_data_raw = cur_dir + '\\' + test_folder + '\\' + '0_Data original'  # raw data folder
        file_name_sen = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TP.csv'  # sensor data file name
        file_name_coup1 = test_date[0:4] + '-' + test_date[4:6] + '-' + test_date[6:8] + '_TC1.csv'  # thermal couples data file name
        path_sensor = folder_data_raw + '\\' + file_name_sen  # file location of sensor data
        path_coup1 = folder_data_raw + '\\' + file_name_coup1  # file location of coupler data
        path_config = cur_dir + '\\' + test_folder + '\\' + '1_Data formatted' + '\\' + test_date + '_config_new.json'
        raw_sen = DataClean.read_logger_data_DTR_PD(path_sensor)  # read MDC4-M data
        raw_coup1 = DataClean.read_couple_flex_format(path_coup1, datetime_format)  # read thermal couples data
        # print('raw_sen', raw_sen)
        # print('raw_coup1', raw_coup1)
        sync_df = DataClean.synch_logger_couple_resample(raw_sen, raw_coup1, sample_time)  # synchronize two data sets
        # print(sync_df.index)
        # print(sync_df.columns)
        # rename columns based on .json config file
        with open(path_config, 'r') as f:
            config = json.load(f)
        sync_df = sync_df.rename(columns=config)
        print(sync_df.columns)
        return sync_df
        # sync_df.to_csv(path_data_clearn)

    def sync_by_resample(cur_dir_raw, file_name_log, file_name_couple, file_name_json, **kwargs):
        """method that synchronize the data files MDC4-M and thermal couples;

		Args:
			cur          - directory of data files

		Return:
			

		Notes:

		"""
        sample_time = kwargs.get('sample_time', '1min') # get sample_time data
        format_couple = kwargs.get('format_couple', '%m/%d/%Y:%H:%M:%S') # get datetime format
        format_log = kwargs.get('format_log', '%Y-%m-%d %H:%M:%S') # get datetime format

        path_log = cur_dir_raw + '\\' + file_name_log  # sensor data file path
        path_couple = cur_dir_raw + '\\' + file_name_couple  # thermal couple data file path
        path_config = cur_dir_raw + '\\' + file_name_json # json file file pass
        raw_log = DataClean.read_logger_data_DTR_PD(path_log, 
                                                    format=format_log)  # read MDC4-M data
        raw_couple = DataClean.read_couple_flex_format(path_couple, 
                                                       format=format_couple)  # read thermal couples data
        sync_df = DataClean.synch_logger_couple_resample(raw_log, raw_couple, sample_time)  # synchronize two data sets
        # print(sync_df.index)
        # print(sync_df.columns)
        # rename columns based on .json config file
        with open(path_config, 'r') as f:
            config = json.load(f)
        sync_df = sync_df.rename(columns=config)
        print(sync_df.columns)
        return sync_df
        # sync_df.to_csv(path_data_clearn)
        
    # def interp_zero(data):
    #     """interpolate the "0" in data array.
    #
    #     Args:
    #     -------
    #         data - array to be interpolated.
    #     Return:
    #     -------
    #         NA
    #     Notes:
    #         Updated on 20230406
    #     """
    #     # find the index of zero points
    #     zero_ix = []
    #     data_ser = data.copy()
    #     data_ser =
    #     for ix,value in enumerate(data_ser):
    #         if value == 0:
    #             zero_ix.append(ix)
    #     print(zero_ix)
    #
    #     # drop zero of original series
    #     data_drop_zero = data_ser.drop(zero_ix)
    #
    #     # drop zero of time index
    #     ix_drop = np.arange(0,len(data_arr)).drop(zero_ix)
    #
    #     # do interpolation
    #     f_raw = interpolate.interp1d(
    #         ix_drop, data_drop_zero, kind='linear', fill_value='extrapolate')
    #     for k in zero_ix:
    #         data_arr[k] = np.around(f_raw(k), 2)
    #     # data['inter'] = data_arr
    #     # data_ser = data['inter']
    #     return data_arr
# %%
class TempRiseExperiment_Norway(object):
    """Class of a set of temperature rise experiment in norway lab."""

    def __init__(self):
        self.bal_idx = 0
        self.x_tick_idx_list = []
        self.x_tick_str_list = []
        self.data = pd.DataFrame()


    def read_data_special(self,path):
        # For test on 20220202 only
        df_str = pd.read_csv(path, sep=';', header=1)
        # change all ";" in data elements into "."
        self.data = pd.concat([df_str.iloc[:, :2], df_str.iloc[:, 2:].applymap(self.semi_to_float)], axis=1)
        self.data['Time.1'] = [datetime.strptime(self.data.iloc[i, 1], '%H:%M:%S') for i in range(len(self.data))]

    @staticmethod
    def semi_to_float(string):
        return float(string.replace(',', '.'))

    def find_balance_index(self,col_num_list):
        """method that find the temperature balancing point of channels assigned
        Args:
        -------
            col_num_list   - list of column numbers to be checked

        Return:
        -------
            NA
        Notes:
            balance condition is TR of cable terminals is less than 1K
        """
        # sliced data that only including temperature columns
        data_sliced = self.data.loc[:, col_num_list].copy()
        # create column for average ambient temp
        data_sliced['Ref_avg'] = (data_sliced['Ref 1'] + data_sliced['Ref 2'] +
                                  data_sliced['Ref 3'] + data_sliced['Ref 4']) / 4
        for j in range(data_sliced.shape[1]):
            data_sliced.iloc[:,j] = data_sliced.iloc[:,j] - data_sliced['Ref_avg']

        # search balance time after test started for 2 hours sample time is 30s
        try:
            for k in range(240, len(data_sliced)):
                data_sliced_diff = (
                        data_sliced.iloc[k, :] - data_sliced.iloc[k - 120, :]).abs()
                # print(data_sliced_diff)
                if (data_sliced_diff < 1.0).all(axis=None):
                    self.bal_idx = k
                    print('Temperature balance time is {time}.'.format(
                        time=self.data.loc[k, 'datetime_log']))
                    break
                elif k == len(data_sliced) - 1:
                    self.bal_idx = k
                    print('Temperature is not balanced.')
                    break
        except Exception:
            print("Find balance point of {name} data error")

    def tr_plot_0202(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            For tests performed on 20220202 only
        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature Rise')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')
        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # figure
        plt.figure(dpi=200)

        # plot curves as per "col_name_list"
        for i, name in enumerate(col_name_list):
            # self.interp_data_zero(name)
            plt.plot(self.data['Time.1'],
                     self.data[name].values -
                     self.data['Avg'].values,
                     linestyle=line_style[i],
                     color=line_color[i],
                     label=name + ' ({:.1f}K)'.format(
                         self.data.loc[self.bal_idx, name] - self.data.loc[self.bal_idx, 'Avg']))
        # except Exception:
        #     print("Plot {col_name} data error".format(col_name=name))

        # plot balance time marker
        plt.plot([self.data.iloc[self.bal_idx,1], self.data.iloc[self.bal_idx,1]],
                 [self.data[col_name_list].max().max() - self.data.loc[self.bal_idx, 'Avg'],
                  self.data[col_name_list].max().min() - self.data.loc[self.bal_idx, 'Avg']],
                 color='k',
                 linewidth=0.5,
                 linestyle="--")

        # specify figure properties
        self.x_tick_idx_list = [self.data.iloc[i,1] for i in range(0,len(self.data),120)]
        self.x_tick_str_list = [datetime.strftime(i, '%H:%M') for i in self.x_tick_idx_list]
        plt.xticks(self.x_tick_idx_list, self.x_tick_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()

    def tr_plot(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            For tests performed after 20220303
        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature Rise')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')
        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # figure
        plt.figure(dpi=200)

        # plot curves as per "col_name_list"
        oil_avg = (self.data['Ref 1'] + self.data['Ref 2'] +
                   self.data['Ref 3'] + self.data['Ref 4'] ) / 4
        for i, name in enumerate(col_name_list):
            # print('data type of oil avg : ',type(oil_avg))
            # print('data type of',name,' : ',type(self.data[name]))
            plt.plot(self.data['idx_log'],
                     self.data[name].values - oil_avg.values,
                     linestyle=line_style[i],
                     color=line_color[i],
                     # label=name + ' ({:.1f}K)'.format(
                     #     self.data.loc[self.bal_idx, name] - oil_avg[self.bal_idx]))
                     label=name + (' stable TR is ' + ' ({:.1f}K)' + ' max TR is' +
                           '({:.1f}K)').format(self.data.loc[self.bal_idx, name] - oil_avg[self.bal_idx],
                           max(self.data.loc[:, name].values - oil_avg.values)))
        # except Exception:
        #     print("Plot {col_name} data error".format(col_name=name))

        # plot balance time marker
        plt.plot([self.data.loc[self.bal_idx,'idx_log'], self.data.loc[self.bal_idx,'idx_log']],
                 [self.data[col_name_list].max().max() - oil_avg[self.bal_idx],
                  self.data[col_name_list].max().min() - oil_avg[self.bal_idx]],
                 color='k',
                 linewidth=0.5,
                 linestyle="--")

        # specify figure properties
        self.x_tick_idx_list = [self.data.loc[i,'idx_log'] for i in range(0,len(self.data),120)]
        self.x_tick_str_list = [datetime.strftime(self.data.loc[i,'datetime_log'],'%H:%M') for i in self.x_tick_idx_list]
        plt.xticks(self.x_tick_idx_list, self.x_tick_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(fontsize='7')


        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()

    def temp_plot(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted
            title           - figure title
            x_label         - x label
            y_label         - y label
        Return:
        -------
            NA
        Notes:
        -------
            For tests performed after 20220303
        """
        # keywords arguments list
        title = kwargs.get('title', 'Absolute Temperature')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature (°C)')
        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # figure
        plt.figure(dpi=200)

        for i, name in enumerate(col_name_list):
            plt.plot(self.data['idx_log'],
                     self.data[name].values,
                     linestyle=line_style[i],
                     color=line_color[i],
                     label=name + ' ({:.1f}K)'.format(
                         self.data.loc[self.bal_idx, name]))

        # plot balance time marker
        plt.plot([self.data.loc[self.bal_idx,'idx_log'], self.data.loc[self.bal_idx,'idx_log']],
                 [self.data[col_name_list].max().max(),
                  self.data[col_name_list].max().min()],
                 color='k',
                 linewidth=0.5,
                 linestyle="--")

        # specify figure properties
        self.x_tick_idx_list = [self.data.loc[i,'idx_log'] for i in range(0,len(self.data),120)]
        self.x_tick_str_list = [datetime.strftime(self.data.loc[i,'datetime_log'],'%H:%M') for i in self.x_tick_idx_list]
        plt.xticks(self.x_tick_idx_list, self.x_tick_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()


        # save figure at 'fig_path'
        if kwargs.get('fig_path'):
            time_now = datetime.now().strftime('%Y%m%d%H%M%S')
            plt.savefig(kwargs.get('fig_path') + '\\' +
                        title + '_' + time_now + '.png', dpi=200)

        # show
        plt.show()
        # plt.close()

    def interp_data_zero(self, col_name):
        """interpolate the "0" in data column.

        Args:
        -------
            col_name - column name to be interpolated.
        Return:
        -------
            NA
        Notes:
        -------
            NA
        """
        # find the index of zero points
        zero_index = []
        for i in range(len(self.data[col_name])):
            if self.data.loc[i, col_name] == 0:
                zero_index.append(i)
        # print(zero_index)

        # drop zero of original series
        raw_drop = self.data[col_name].drop(zero_index)

        # drop zero of time index
        time_index_drop = self.data['idx_log'].drop(zero_index)

        # do interpolation
        f_raw = interpolate.interp1d(
            time_index_drop, raw_drop, kind='linear', fill_value='extrapolate')
        for k in zero_index:
            self.data.loc[k, col_name] = np.around(f_raw(k), 1)

class DataClean_Norway(object):

    def read_logger_data(path):
        """method that read the data sampled by TR data logger,

        Args:
            path                - full path of the .csv data file

        Return:
            raw_data_sliced     - raw data sliced from starting index to ending index
        Notes:
            sample time is 15s need to be down-sampled to 30s

        """
        raw_data = pd.read_csv(path, header=0)

        # convert string data to float
        raw_data.iloc[:, 1:].astype(float)

        # convert str to datetime
        raw_data['Time'] = [datetime.strptime(raw_data.loc[i, 'Time'], '%Y-%m-%d %H:%M:%S') for i in range(len(raw_data))]
        raw_data = raw_data.rename(columns={'Time':'datetime'})

        # Sample time is 15s, down-sample to 30s
        raw_data = raw_data[raw_data.index % 2 == 0]
        raw_data.index = range(len(raw_data))
        print(raw_data.iloc[:,0])
        # search test start index
        for i in range(len(raw_data)):
            if (raw_data.iloc[i, 3:21] == 0).all():
                if i == (len(raw_data) - 1):
                    print('test data was all "0"')
                else:
                    pass
            else:
                t0 = i
                print('logger started from %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break

        # search test end index from 1 hour after test started
        for i in range(t0 + 120, len(raw_data)):
            if ((raw_data.iloc[i:i+10, 3:21] == 0).all()).all():
                tn = i
                print('i = ',i)
                print('logger ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                tn = i + 1
                print('test data are not fully recorded')
            else:
                continue

        # slice data from start index to end index
        raw_data_sliced = raw_data.iloc[t0:tn, :]
        raw_data_sliced.index = range(len(raw_data_sliced))
        return raw_data_sliced

    def read_couple_data(path):
        """method that read the data from thermalcouples;

        Args:
            path        - full path of the data file

        Return:
            raw_data    - raw data read
        Notes:
            sample time is 30s

        """
        raw_data = pd.read_csv(path,
                               header=1,
                               na_values=['          ', '     -OVER'])
        # print(raw_data)
        # print(raw_data.index)
        raw_data.iloc[:, 1:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data['Time'] = [datetime.strptime(raw_data.loc[i, 'Time'], '%d/%m/%Y %H:%M:%S') for i in
                            range(len(raw_data))]
        output = raw_data.rename(columns={'Time': 'datetime'})

        # Re-sampling
        #     output = raw_data[raw_data.index % multiple == 0]
        return output

    def read_couple_data_0307(path):
        """method that read the data from thermalcouples;

        Args:
            path        - full path of the data file

        Return:
            raw_data    - raw data read
        Notes:
            sample time is 30s

        """
        raw_data = pd.read_csv(path,
                               header=1,
                               na_values=['          ', '     -OVER'])
        # delete the space at the end of column 'datetime'
        for i in range(len(raw_data)):
            raw_data.iloc[i, 0] = raw_data.iloc[i, 0].strip()

        # print(raw_data)
        # print(raw_data.index)
        raw_data.iloc[:, 1:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data['Time'] = [datetime.strptime(raw_data.loc[i, 'Time'], '%d/%m/%y %H:%M:%S') for i in
                            range(len(raw_data))]
        output = raw_data.rename(columns={'Time': 'datetime'})

        # Re-sampling
        #     output = raw_data[raw_data.index % multiple == 0]
        return output

    def synch_logger_couple(*data):
        """method that synchronize data of data_logger with groups of data of thermocouples;

        Args:
        -------
            data_logger        - data of data_logger
            data               - list of data of thermocouples

        Return:
        -------
            data_list          - synchronized list of data groups
        Notes:
        -------

        """
        # search the common starting time
        start_time = max([i.loc[i.index[0], 'datetime']
                          for i in data])
        # search the common ending time
        end_time = min([i.loc[i.index[-1], 'datetime']
                          for i in data])
        print(data[0].shape)
        print(data[1].shape)
        data_list = []
        print('sensor & couplers common start time = ',
              start_time)
        print('sensor & couplers common end time =',
              end_time)

        count = 0

        # perform on each group of data
        for j,element in enumerate(data):
            # add one column of index relative to the common starting time
            element['idx'] = [math.ceil(
                (element.loc[element.index[i], 'datetime'] - start_time).seconds / 30) for i in range(len(element))]

            element = element.loc[(element['idx'] >= 0)
                                  & (element['idx'] <= 1440), :]

            # check and fix that if there are any duplicated index
            k = 0
            while (k < len(element) - 1):
                # print('len element', len(element))
                if element.loc[element.index[k], 'idx'] == element.loc[element.index[k + 1], 'idx']:
                    element = element.drop(k + 1)
                    count += 1
                    k = k + 1
                else:
                    pass
                k = k + 1
            print('number of duplicated index is : ', count)

            # check and fix that if there are any discontinued index
            if (element.loc[element.index[-1], 'idx'] - element.loc[element.index[0], 'idx']) != (len(element) - 1):
                print('%dth group of data has discontinued points' % j)
                # print('head,tail and length are ',
                #       element.loc[0, 'ind'],
                #       element.loc[-1, 'ind'],
                #       len(element))

                # build the full length index
                full_index = pd.Series(
                    range(element.loc[element.index[0], 'idx'], element.loc[element.index[-1], 'idx'] + 1))
                # find the discontinued index
                miss_index = full_index.loc[~full_index.isin(list(element.idx))]
                element_full = pd.DataFrame(data=0, columns=element.columns, index=range(len(full_index)))

                # fill in data
                for m in range(len(element)):
                    element_full.iloc[element.loc[element.index[m], 'idx'] -
                                      element.loc[element.index[0], 'idx'], :] = element.iloc[m, :]

                # fill in with missed index
                for n in range(1,len(element_full)):
                    if element_full.loc[element_full.index[n],'idx'] == 0:
                        element_full.loc[element_full.index[n], 'idx'] = element_full.loc[element_full.index[n-1], 'idx'] + 1
                data_list.append(element_full)
            else:
                print('%dth group of data has no discontinued points' % j)
                data_list.append(element)
                pass

        # Find the common start and ending index of t
        start_ind = max([i.loc[i.index[0],'idx'] for i in data_list])
        end_ind = min([i.loc[i.index[-1],'idx'] for i in data_list])
        print('sensor & couplers common start index :',start_ind)
        print('sensor & couplers common end index :',end_ind)

        # 用共同起止索引截取三组数据重叠的数据点
        for n, ele in enumerate(data_list):
            data_list[n] = ele.loc[ele.loc[:, 'idx'] <= end_ind, :]

            data_list[n] = data_list[n].loc[data_list[n].loc[:, 'idx']
                                            >= start_ind, :]

            data_list[n].index = range(len(data_list[n]))

        return data_list

class TempRiseExperiment_IN(object):

    def __init__(self, file_path):
        """Init the data of gas pressure experiment.

        Args:
        -------
            data_df 	 - 	raw data of dataframe.

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

    def tr_plot(self, col_name_list, **kwargs):
        """General Plotting function of all temp curves
        Args:
        -------
            col_name_list   - list of column names to be plotted

        Return:
        -------
            NA
        Notes:
        -------

        """
        # keywords arguments list
        title = kwargs.get('title', 'Temperature Rise')
        x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')
        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # figure
        plt.figure(dpi=200)
        # plot curves as per "col_name_list"
        oil_avg = (self.data['Point 29'].values + 
                    self.data['Point 30'].values +
                    self.data['Point 31'].values) / 3

        for i, name in enumerate(col_name_list):

            plt.plot(self.data.index,
                        self.data[name].values - oil_avg,
                        linestyle=line_style[i],
                        color=line_color[i],
                        label=name + (' max TR is' +
                                    '({:.1f}K)').format(max(self.data.loc[:, name].values - oil_avg))
                                    )


        # specify figure properties
        self.x_tick_idx_list = [i for i in range(0,len(self.data),60)]
        self.x_tick_str_list = ['{0:02d}:{1:02d}'.format(math.floor((self.data.loc[i,'date_time'] -                                                                     self.data.loc[0,'date_time']
                                                                    ).seconds / 3600),
                                                         math.floor(((self.data.loc[i,'date_time'] -
                                                                      self.data.loc[0,'date_time']).seconds % 3600) / 60)
                                                        ) 
                                for i in self.x_tick_idx_list
                                ]
        plt.xticks(self.x_tick_idx_list, self.x_tick_str_list)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(fontsize='7')

    def imbalance_plot(self, col_name_list, **kwargs):
            """General Plotting function of all temp curves
            Args:
            -------
                col_name_list   - list of column names to be plotted
                title           - figure title
                x_label         - x label
                y_label         - y label
            Return:
            -------
                NA
            Notes:
            -------
                NA
            """
            # keywords arguments list
            title = kwargs.get('title', '3-phase Imbalance Temperature')
            x_label = kwargs.get('x_label', 'Time (Hours:Minutes)')
            y_label = kwargs.get('y_label', 'Imbalance Temperature(K)')

            # figure
            fig,ax1 = plt.subplots(dpi=200)

            # plot curves as per "col_name_list"
            imb_list = [max([self.data.loc[i,j] for j in col_name_list]) -
                        min([self.data.loc[i,j] for j in col_name_list]) for i in range(len(self.data))]

            plt.plot(self.data.index,
                    imb_list,
                    color='k',
                    label= 'max imbalance temperature is' + ' ({:.1f}K)'.format(max(imb_list)))

            # specify figure properties
            self.x_tick_idx_list = [i for i in range(0,len(self.data),60)]
            self.x_tick_str_list = ['{0:02d}:{1:02d}'.format(math.floor((self.data.loc[i,'date_time'] -                                                                     self.data.loc[0,'date_time']
                                                                        ).seconds / 3600),
                                                            math.floor(((self.data.loc[i,'date_time'] -
                                                                        self.data.loc[0,'date_time']).seconds % 3600) / 60)
                                                            ) 
                                    for i in self.x_tick_idx_list
                                    ]
            plt.xticks(self.x_tick_idx_list, self.x_tick_str_list)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(fontsize='7')

class DynTempRise(object):
    def find_bal_idx(data, col_list, sample_min):
        """ find balance datetime index of columns
            Args:
            -------
                col_list   - list of column names
                data       - data in dataframe
            Return:
            -------
                NA
            Notes:
            -------
                NA
        """   
        data_copy = data.loc[:, col_list].copy()
        sample_cyc_1h = int(60 / sample_min)
        for k in range(sample_cyc_1h * 4, len(data_copy)):
            diff = (data_copy.iloc[k, :] - 
                    data_copy.iloc[k - sample_cyc_1h, :]).abs()
            if (diff < 1.0).all(axis=None):
                bal_idx = data_copy.index[k]
                print('Temperature balance time is {}.'.format(bal_idx))
                break
            elif k == len(data_copy) - 1:
                bal_idx = data_copy.index[k]
                print('Temperature is not balanced.')  
        return bal_idx  
    
    def cal_time_const(data, col_name, **kwargs):
        """ find balance datetime index of columns
            Args:
            -------
                col_list   - list of column names
                data       - data in dataframe
            Return:
            -------
                const      - time constant in minutes
            Notes:
            -------
                NA
        """
        data_copy = data.copy()
        amb_col_name = kwargs.get('amb_col_name', 't_oil_avg')
        date = kwargs.get('date', '')
        tw = max(data_copy[col_name] - data_copy[amb_col_name]) # max tr
        print("tw is",tw)
        ydata = data_copy[col_name].values - data_copy[amb_col_name].values # 
        xdata = np.array(range(len(ydata)))
        t0 = ydata[0]
        # print(ydata)
        def f_trans_tr(x, T):
            return tw * (1 - np.exp(-1 * x * 5 / T)) +  t0 * np.exp(-1 * x * 5 / T)
        const = optimize.curve_fit(f_trans_tr, xdata, ydata)[0][0]
        print('time constant T of is', const)

        # plotting time constant calculation result
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        ax2 = ax1.twinx()
        rmse = TempRiseExperiment.get_rmse(f_trans_tr(xdata, const),
                                           ydata)
        print('RMSE is ',rmse)
        ax1.scatter(xdata, 
                    ydata, 
                    color='r', 
                    label='temperature rise tested')
        ax1.plot(xdata,
                f_trans_tr(xdata, const),
                color='g',
                label='temperature rise fitted')

        # create line plot of difference data
        ax2.plot(xdata,
                f_trans_tr(xdata, const) - ydata,
                color='k',
                label='difference with RMSE of {:.1f}'.format(rmse))
        ax1.set_title('{} {} time constant is {:.0f}'.format(date, 
                                                             col_name, 
                                                             const))
        ax1.set_xlabel('sample point')
        ax1.set_ylabel('Temperature Rise (K)')
        ax2.set_ylabel('Difference (K)')
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)
        return const               

    def cal_dynamic_conver_const(xdata, ydata, tr_rated, **kwargs):
        """method that calculate the DTR conversion constant of one data set;

        Args:
        -------
            xdata   - x data to be fitted
            ydata   - y data to be fitted

        Return:
        -------
            const   - conversion constant calculated   
        Notes:
        -------
            the last data point must be the TR rated (630A)
        """
        # initialize data
        title = kwargs.get('title',
                           'Fitting for DTR conversion constant')
        xdata = np.array(xdata)
        ydata = np.array(ydata)

        # create function to be fitted
        def f_steady_state_tr(x, a):
            return (x / 630) ** a * tr_rated

        # fitting
        const = optimize.curve_fit(f_steady_state_tr,
                                xdata,
                                ydata)[0][0]
        print('conversion constant a is',
            const)
        plt.figure(dpi=200)
        sns.set(color_codes=True)
        # create scatter plot of input data point
        plt.scatter(xdata,
                    ydata,
                    color='r')
        # create line plot of fitted curve
        x_span = np.array(range(min(xdata),
                                max(xdata) + 1,
                                10))
        plt.plot(x_span,
                f_steady_state_tr(x_span, const),
                color='g',
                label='Conversion Constant is {:.2f}'.format(const))
        plt.xlabel('current (A)')
        plt.ylabel('Temperature Rise (K)')
        plt.legend()
        plt.title(title)
        # return const
    
    def dtr_sim_plot(data, col_name, cur_col_name, tr_rated, tr_warn, tr_alarm, time_const, conver_const, **kwargs):
        correction_warn = kwargs.get('correction_warn', 5)
        correction_alarm = kwargs.get('correction_alarm', 5)
        time_const_drop = kwargs.get('time_const_drop', time_const)
        delay_const = kwargs.get('delay_const', 0)
        title = kwargs.get('title', 'dtr simulation')
        input_temp_label = kwargs.get('input_temp_label', col_name)
        data_df = data.copy()
        current = data_df[cur_col_name]
        data_df['t_amb_avg'] = (data_df['t_oil_bottle_4'] +
                                data_df['t_oil_bottle_3'] +
                                data_df['t_oil_bottle_2'] +
                                data_df['t_oil_bottle_1'] * 4) / 4  
        data_df['tw_rated'] = (current / 630) ** conver_const * tr_rated + data_df['t_amb_avg']
                            
        data_df['tw_warn'] = (current / 630) ** conver_const * tr_warn + data_df['t_amb_avg']

        data_df['tw_alarm'] = (current / 630) ** conver_const * tr_alarm + data_df['t_amb_avg']
        
        # build temperature data structure
        data_df['t_rated'] = 0
        data_df['t_warn'] = 0
        data_df['t_alarm'] = 0
        data_df['si'] = 0
        data_df.loc[data_df.index[0], 't_rated'] = data_df.loc[data_df.index[0], col_name]
        data_df.loc[data_df.index[0], 't_warn'] = data_df.loc[data_df.index[0], col_name]
        data_df.loc[data_df.index[0], 't_alarm'] = data_df.loc[data_df.index[0], col_name]
        
        # form temperature sequences
        # use different time constant "time_const_drop" when current is dropping
        for k in range(1, len(data_df)):
            if data_df.loc[data_df.index[k], col_name] <= data_df.loc[data_df.index[k], 'tw_warn']:
                data_df.loc[data_df.index[k], 't_rated'] = (data_df.loc[data_df.index[k - 1], 't_rated'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_rated'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_rated']) * 
                                                            (1 - np.exp(-1 * 5 / time_const)))
                data_df.loc[data_df.index[k], 't_warn'] = (data_df.loc[data_df.index[k - 1], 't_warn'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_warn'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_warn']) * 
                                                            (1 - np.exp(-1 * 5 / time_const)))
                data_df.loc[data_df.index[k], 't_alarm'] = (data_df.loc[data_df.index[k - 1], 't_alarm'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_alarm'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_alarm']) * 
                                                            (1 - np.exp(-1 * 5 / time_const)))                                             
            elif data_df.loc[data_df.index[k], col_name] > data_df.loc[data_df.index[k], 'tw_warn']:    
                data_df.loc[data_df.index[k], 't_rated'] = (data_df.loc[data_df.index[k - 1], 't_rated'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_rated'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_rated']) * 
                                                            (1 - np.exp(-1 * 5 / time_const_drop)))
                data_df.loc[data_df.index[k], 't_warn'] = (data_df.loc[data_df.index[k - 1], 't_warn'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_warn'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_warn']) * 
                                                            (1 - np.exp(-1 * 5 / time_const_drop)))
                data_df.loc[data_df.index[k], 't_alarm'] = (data_df.loc[data_df.index[k - 1], 't_alarm'] + 
                                                            (data_df.loc[data_df.index[k], 'tw_alarm'] - 
                                                            data_df.loc[data_df.index[k - 1], 't_alarm']) * 
                                                            (1 - np.exp(-1 * 5 / time_const_drop)))       
            else:
                print('current value error')
        # delay 'delay_const' number of sampling points 
        if delay_const != 0:
            t_rated_list = list(data_df['t_rated'])
            t_warn_list = list(data_df['t_warn'])
            t_alarm_list = list(data_df['t_alarm'])
            
            # prepend list with "delay_const" number of data points
            t_rated_list_delay = [t_rated_list[0]] * delay_const + t_rated_list
            t_warn_list_delay = [t_warn_list[0]] * delay_const + t_warn_list
            t_alarm_list_delay = [t_alarm_list[0]] * delay_const + t_alarm_list
            
            # cut list the last "const" number of element from its tail
            del t_rated_list_delay[-1 * delay_const:]
            del t_warn_list_delay[-1 * delay_const:]
            del t_alarm_list_delay[-1 * delay_const:]

            data_df['t_rated'] = t_rated_list_delay
            data_df['t_warn'] = t_warn_list_delay
            data_df['t_alarm'] = t_alarm_list_delay
        else:
            pass
            
        # adding correction const to warning and alarm values
        data_df['t_warn'] = data_df['t_warn'] + correction_warn
        data_df['t_alarm'] = data_df['t_alarm'] + correction_alarm

        # form the signal indicator data column
        
        for i in range(len(data_df)):
            if (0 <= data_df.loc[data_df.index[i], col_name] < 
                data_df.loc[data_df.index[i], 't_warn']):
                data_df.loc[data_df.index[i], 'si'] = 1
            elif (data_df.loc[data_df.index[i], 't_alarm'] > 
                  data_df.loc[data_df.index[i], col_name] >= 
                  data_df.loc[data_df.index[i], 't_warn']):         
                data_df.loc[data_df.index[i], 'si'] = 2

            elif (data_df.loc[data_df.index[i], col_name] > 
                  data_df.loc[data_df.index[i], 't_alarm']):
                data_df.loc[data_df.index[i], 'si'] = 3

            else:
                data_df.loc[data_df.index[i], 'si'] = 0

        # figure
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        # plt.scatter(data_df.index,
        #             data_df['t_rated'],
        #             s=5,
        #             color='c',
        #             marker='x',
        #             label='t_rated_scatter')
        plt.plot(data_df['t_rated'],
                color='c',
                label='t_rated')
        plt.plot(data_df[col_name],
                color='g',
                label=input_temp_label)
        plt.plot(data_df['t_warn'],
                color='y',
                label='t_warn')
        plt.plot(data_df['t_alarm'],
                color='r',
                label='t_alarm')

        ax2 = ax1.twinx()
        ax2.plot(data_df[cur_col_name],
                label='current',
                linestyle=':',
                color='k')
        plt.grid(b=None)
        plt.title(title)
        ax1.set_ylabel('Temperature (C)')
        ax2.set_ylabel('Time Variant Current (A)')
        ax1.legend(fontsize=7,
                   loc='upper left')
        ax1.legend(fontsize=7,
                   loc='upper right')
        ax1.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        return data_df