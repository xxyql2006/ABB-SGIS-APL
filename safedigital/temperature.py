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

        # initial parameters
        self.bal_idx = 0
        self.x_idx_list = []
        self.x_str_list = []
        self.t_oil = self.data.loc[:, 't_oil_bottle_1']

    def interp_data(self, col_name):
        # find the index of zero points
        zero_index = []
        for i in range(len(self.data[col_name])):
            if self.data.loc[i, col_name] == 0:
                zero_index.append(i)
        # print(zero_index)

        # drop zero of original series
        raw_drop = self.data[col_name].drop(zero_index)

        # drop zero of time index
        time_index_drop = self.data['snsr_timeindex'].drop(zero_index)

        # do interpolation
        f_raw = interpolate.interp1d(
            time_index_drop, raw_drop, kind='linear', fill_value='extrapolate')
        for k in zero_index:
            self.data.loc[k, col_name] = np.around(f_raw(k), 1)

    def np_move_avg(self, a, n, mode="valid"):
        return(np.convolve(a, np.ones((n,))/n, mode=mode))

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
        plt.figure(dpi=100, figsize=(3, 2))
        sns.set(color_codes=True)

        # default line color and styles
        line_style = kwargs.get('line_style',
                                ['-'] * len(col_name_list))
        line_color = kwargs.get('line_color',
                                ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * len(col_name_list))

        # plot curves as per "col_name_list"
        try:
            for i, name in enumerate(col_name_list):
                self.interp_data(name)
                plt.plot(self.data['snsr_timeindex'].values,
                         self.data[name].values -
                         self.data['t_oil_bottle_1'].values,
                         linestyle=line_style[i],
                         color=line_color[i],
                         label=name + ' ({:.1f}K)'.format(self.data.loc[self.bal_idx, name] - self.t_oil[self.bal_idx]))
        except Exception:
            print("Plot {col_name} data error".format(col_name=name))

        # plot balance time marker
        plt.plot([self.bal_idx, self.bal_idx],
                 [self.data[col_name_list].max().max() - self.t_oil[self.bal_idx] + 1,
                  self.data[col_name_list].max().min() - self.t_oil[self.bal_idx] - 1],
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
        # plt.show()
        plt.close()

    def t_dtr_plot(self, col_name, cur_var_list, time_const, conver_const, tr_rated, const, **kwargs):
        """Plotting dynamic temperature rise algorithm
        Args:
        -------
            col_name        - column name to be plotted
            cur_var_list    - list of current time-variant data points
            time_const      - time constant
            conver_const    - conversion constant
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
        title = kwargs.get('title',
                           'Dynamic Temperature Rise Performance of {}'.format(col_name))
        x_label = kwargs.get('x_label',
                             'Sample Point')
        y_label = kwargs.get('y_label',
                             'Temperature Rise(K)')

        # figure
        fig, ax1 = plt.subplots(1, 1, dpi=300)
        sns.set(color_codes=True)

        # cut data as long as current data point list
        self.interp_data(col_name)
        data_cut = self.data.loc[:len(
            cur_var_list)-1, [col_name, 't_oil_bottle_1']].copy()
        data_cut[col_name] = data_cut[col_name] - data_cut['t_oil_bottle_1']

        # build current time-variant data column
        data_cut['cur_var'] = cur_var_list
        # build steady-state temperature rise column
        data_cut['tao_w'] = [(i / 630) ** conver_const *
                             tr_rated for i in cur_var_list]
        # build fitted temperature rise data column
        tr_fit_list = ([data_cut.loc[0, col_name]] +
                       [0] * (len(cur_var_list) - 1))
        for k in range(1, len(cur_var_list)):
            tr_fit_list[k] = tr_fit_list[k - 1] + (
                data_cut.loc[k, 'tao_w'] - tr_fit_list[k - 1]) * (1 - np.exp(-10 / time_const))
        # prepend list with "const" number of data points
        tr_fit_list = [tr_fit_list[0]] * const + tr_fit_list
        # cut list the last "const" number of element from its tail
        del tr_fit_list[-1 * const:]
        data_cut['tr_fit'] = tr_fit_list
        # idx_delay = np.array(data_cut.index) + const
        plt.plot(data_cut.index,
                 data_cut[col_name],
                 color='k',
                 label=col_name)
        plt.plot(data_cut.index,
                 data_cut['tr_fit'],
                 color='g',
                 label='tr_fit')
        plt.plot(data_cut.index,
                 1.1 * data_cut['tr_fit'],
                 color='y',
                 label='tr_warning')
        plt.plot(data_cut.index,
                 1.2 * data_cut['tr_fit'],
                 color='r',
                 label='tr_alarm')
        ax2 = ax1.twinx()
        ax2.plot(data_cut.index,
                 data_cut['cur_var'],
                 label='cur_var',
                 linestyle='--',
                 color='lightblue')
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
        data_sliced = self.data.iloc[:, col_num_list]

        try:
            for k in range(1440, len(data_sliced)):
                data_sliced_diff = (
                    data_sliced.iloc[k, :] - data_sliced.iloc[k - 360, :]).abs()

                if (data_sliced_diff <= 1.0).all(axis=None):
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

    def cal_dynamic_time_const(self, col_name):
        """method that calculate the DTR time constant of one data column of given name;
           compare the fitted curve with scatter points
        Args:
        -------
            col_name   - column name of data to be calculated 

        Return:
        -------
            const     - time constant calculated   
        Notes:
        -------
            NA,
        """
        # load x,y data
        ydata = (self.data[col_name] - self.t_oil).copy().values
        xdata = self.data['snsr_timeindex'].copy().values
        # calculate steady-state temperature rise
        tw = (self.data.loc[self.bal_idx, col_name] -
              self.t_oil.loc[self.bal_idx])
        # calculate initial temperature rise
        t0 = self.data.loc[0, col_name] - self.t_oil[0]
        # create function to be fitted

        def f_transient_tr(x, T):
            return tw * (1 - np.exp(-1 * x * 10 / T)) + t0 * np.exp(-1 * x * 10 / T)
        # fitting
        const = optimize.curve_fit(f_transient_tr, xdata, ydata)[0][0]
        print('time constant T of {} is'.format(col_name), const)
        # plt.figure(dpi=500, figsize=(10,6))
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        ax2 = ax1.twinx()
        sns.set(color_codes=True)
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
                 f_transient_tr(xdata, const)-ydata,
                 color='k',
                 label='difference with RMSE of {:.1f}'.format(rmse))
        ax1.set_title('{} time constant fitting result'.format(col_name))
        ax1.set_xlabel('sample point')
        ax1.set_ylabel('Temperature Rise (K)')
        ax2.set_ylabel('Difference (K)')
        ax1.legend(fontsize=7)
        ax2.legend(fontsize=7)
        return const

    @ staticmethod
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
                                max(xdata)+1,
                                10))
        plt.plot(x_span,
                 f_steady_state_tr(x_span, const),
                 color='g',
                 label='Conversion Constant is {:.2f}'.format(const))
        plt.xlabel('current (A)')
        plt.ylabel('Temperature Rise (K)')
        plt.legend()
        plt.title(title)
        return const

    @ staticmethod
    def get_rmse(predictions, targets):

        return np.sqrt(((predictions - targets) ** 2).mean())


class DataClean(object):

    def read_sensor_data(path):
        # read data
        raw_data = pd.read_csv(path, header=None)
        raw_data.iloc[:, 2:].astype(float)  # 第2列以后全部强制转换为浮点数
        raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i, 0] + ':' + raw_data.iloc[i, 1],
                                                  '%m/%d/%Y:%I:%M:%S %p') for i in range(len(raw_data))]  # 粘合第0列和第1列行程datetime类型数据存入到新的一列

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
                print('sensor data start from %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                print('sensor data was all "0"')
            else:
                continue

        # search test end index
        for i in range(t0, len(raw_data)):
            if (raw_data.iloc[i, 2] == 0 and raw_data.iloc[i, 3] == 0 and raw_data.iloc[i, 4]
                    == 0 and raw_data.iloc[i, 5] == 0 and raw_data.iloc[i, 6] == 0 and raw_data.iloc[i, 7] == 0
                    and raw_data.iloc[i, 8] == 0 and raw_data.iloc[i, 9] == 0 and raw_data.iloc[i, 10] == 0):
                tn = i
                print('sensor data ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
                break
            elif i == (len(raw_data) - 1):
                tn = i + 1
                print('sensor data ended at %s' %
                      (raw_data.loc[i, 'datetime'].strftime("%H:%M:%S")))
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
        return raw_data.iloc[t0:tn, :]

    # ====================================================================================
    # 读取热电偶数据
    # ====================================================================================

    def read_coupler_data(path):

        raw_data = pd.read_csv(path, header=25, na_values=[
                               '          ', '     -OVER'])
        raw_data.iloc[:, 4:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i, 1] + ':' + raw_data.iloc[i, 2],
                                                  '%Y/%m/%d:%H:%M:%S') for i in range(len(raw_data.iloc[:, 2]))]

        # 检查采样间隔，不是十秒的话降采样至10秒
        time_interval = (raw_data.iloc[1, -1] - raw_data.iloc[0, -1]).seconds
        if time_interval != 10:
            multiple = 10 / time_interval
            raw_data = raw_data[raw_data.index % multiple == 0]
        else:
            pass

        return raw_data

    # ====================================================================================
    # 找到三组数据重叠的起止时间，并且同步三组数据的时间轴
    # ====================================================================================

    def synch_data_group(*data):

        start_time = max([data[i].iloc[0, -1]
                         for i in range(len(data))])  # 找到所有组数据重叠部分的共同开始时间
        end_time = min([data[i].iloc[-1, -1]
                       for i in range(len(data))])  # 找到所有组数据重叠部分的共同结束时间
        data_list = []
        print('sensor & couplers common start time = ',
              start_time)  # 打印所有数据重叠起始时间
        print('sensor & couplers common end time =', end_time)  # 打印所有数据重叠结束时间
        j = 0
        for element in data:
            j += 1
            element['ind'] = [math.ceil(
                (element.iloc[i, -1] - start_time).seconds / 10) for i in range(len(element))]
            element = element.loc[(element['ind'] >= 0)
                                  & (element['ind'] <= 4320), :]

            # check and fix that if there are any duplicated index
            for k in range(len(element) - 1):
                if element.iloc[k, -1] == element.iloc[k + 1, -1]:
                    element.iloc[k + 1, -1] += 1
                else:
                    pass
            # 检查数据是否存在间断索引
            if (element.iloc[-1, -1] - element.iloc[0, -1]) != (len(element) - 1):
                print('%dth group of data has discontinued points' % j)
                print('head,tail and length are ',
                      element.iloc[0, -1], element.iloc[-1, -1], len(element))
                full_index = pd.Series(
                    range(element.iloc[0, -1], element.iloc[-1, -1] + 1))
                miss_index = full_index.loc[~full_index.isin(
                    list(element['ind']))]
                element_full = pd.DataFrame(data=0, columns=range(
                    element.shape[1]), index=range(len(full_index)))

                # 填入旧数据
                for i in range(len(element)):
                    element_full.iloc[element.iloc[i, -1] -
                                      element.iloc[0, -1], :] = element.iloc[i, :]

                # 填入缺失索引
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

    @staticmethod
    def down_sample(data, ratio):
        index = np.arange(len(data))
        data_out = data[index % ratio == 0]
        return data_out

# %%
