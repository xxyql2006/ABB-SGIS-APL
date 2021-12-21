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
        self.t_env = self.data.loc[:, 't_env']

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
                         self.data[name].values -
                         self.data['t_oil_bottle_1'].values,
                         linestyle=line_style[i],
                         color=line_color[i],
                         label=name + ' ({:.1f}K)'.format(self.data.loc[self.bal_idx, name] - self.t_env[self.bal_idx]))
        except Exception:
            print("Plot {col_name} data error".format(col_name=name))

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
        y_label = kwargs.get('y_label', 'Temperature Rise(K)')
        sample_time = kwargs.get('sample_time', 10)
        tr_rated = kwargs.get('tr_rated', 0)
        correction_warning = kwargs.get('correction_warning', 8)
        correction_alarm = kwargs.get('correction_alarm', 10)
        time_const_drop = kwargs.get('time_const_drop', time_const)

        # figure
        fig, ax1 = plt.subplots(1, 1, dpi=200)
        # sns.set(color_codes=True)

        # cut data as long as current data point list
        # when 'loc' command is used [a:b] include b 
        data_cut = self.data.loc[:len(cur_list) - 1 ,[col_name, 't_oil_bottle_1','snsr_datetime']].copy() 
        # data_cut['tr'] = data_cut[col_name] - data_cut['t_oil_bottle_1']
        
        # build current time-variant data column
        data_cut['current'] = cur_list
        # data_cut.to_csv('C:\\Users\\cnbofan1\\Desktop\\output.csv')
        # build steady-state temperature column
        data_cut['tao_w_warning'] = [(ele / 630) ** conver_const * tr_warning + data_cut.loc[i, 't_oil_bottle_1']
                            for i, ele in enumerate(cur_list)]
        data_cut['tao_w_alarm'] = [(ele / 630) ** conver_const * tr_alarm + data_cut.loc[i, 't_oil_bottle_1']
                            for i, ele in enumerate(cur_list)]
        # print(data_cut['tao_w'])
       
        # build temperature data structure
        t_warning_list = ([data_cut.loc[0, col_name]] +
                       [0] * (len(cur_list) - 1))
        t_alarm_list = ([data_cut.loc[0, col_name]] +
                       [0] * (len(cur_list) - 1))
        # use different time constant "time_const_drop" when current is dropping
        for k in range(1, len(cur_list)):
            if cur_list[k] >= cur_list[k-1]:
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] - 
                t_warning_list[k - 1]) * (1 - np.exp(-1 * sample_time / time_const))               
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] - 
                t_alarm_list[k - 1]) * (1 - np.exp(-1 * sample_time / time_const))
            elif cur_list[k] < cur_list[k-1]:
                t_warning_list[k] = t_warning_list[k - 1] + (data_cut.loc[k, 'tao_w_warning'] - 
                t_warning_list[k - 1]) * (1 - np.exp(-1 * sample_time / time_const))               
                t_alarm_list[k] = t_alarm_list[k - 1] + (data_cut.loc[k, 'tao_w_alarm'] - 
                t_alarm_list[k - 1]) * (1 - np.exp(-1 * sample_time / time_const))
            else:
                print('current value error at {}th element'.format(k))
        # print('tr_fit_list',tr_fit_list)
        if delay_const != 0:
            # prepend list with "delay_const" number of data points
            tr_warning_list = [t_warning_list[0]] * delay_const + t_warning_list       
            # cut list the last "const" number of element from its tail
            del tr_warning_list[-1 * delay_const:]

            tr_alarm_list = [t_alarm_list[0]] * delay_const + t_alarm_list        
            # cut list the last "const" number of element from its tail
            del tr_alarm_list[-1 * delay_const:]
        # data_cut['t_fit'] = tr_fit_list
        data_cut['t_warning'] = t_warning_list
        data_cut['t_warning'] = data_cut['t_warning'] + correction_warning
        data_cut['t_alarm'] = t_alarm_list
        data_cut['t_alarm'] = data_cut['t_alarm'] + correction_alarm


        for i in range(len(data_cut)):
            if data_cut.loc[i,col_name] < data_cut.loc[i,'t_warning']:
                data_cut.loc[i,'t_si'] = 1
            elif data_cut.loc[i,'t_warning'] <= data_cut.loc[i,col_name] < data_cut.loc[i,'t_alarm']:
                data_cut.loc[i,'t_si'] = 2
            elif data_cut.loc[i,col_name] >= data_cut.loc[i,'t_alarm']:
                data_cut.loc[i,'t_si'] = 3
            else:
                data_cut.loc[i,'t_si'] = 0
        # idx_delay = np.array(data_cut.index) + delay_const
        
        # generate warning, alarm, signal indicator data to mrc
        output =  data_cut.loc[:,['t_warning','t_alarm','t_si','current',col_name,'snsr_datetime']].copy()
        if tr_rated != 0:
            data_cut['tao_w_rated'] = [(ele / 630) ** conver_const * tr_rated + data_cut.loc[i, 't_oil_bottle_1']
                            for i, ele in enumerate(cur_list)]
            t_rated_list = ([data_cut.loc[0, col_name]] + [0] * (len(cur_list) - 1))
            for k in range(1, len(cur_list)):
                t_rated_list[k] = t_rated_list[k - 1] + (data_cut.loc[k, 'tao_w_rated'] - 
                                   t_rated_list[k - 1]) * (1 - np.exp(-1 * sample_time / time_const))
            if delay_const != 0:
                # prepend list with "delay_const" number of data points
                t_rated_list = [t_rated_list[0]] * delay_const + t_rated_list       
                # cut list the last "const" number of element from its tail
                del t_rated_list[-1 * delay_const:]
            data_cut['t_rated'] = t_rated_list
            plt.plot(data_cut.index,
                    data_cut['t_rated'],
                    color='c',
                    label='t_fitted')

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
        ax2 = ax1.twinx()
        ax2.plot(data_cut.index,
                 data_cut['current'],
                 label='current',
                 linestyle=':',
                 color='k')
        xdata = [datetime.strftime(datetime.strptime(i, '%Y-%m-%d %H:%M:%S'),'%H:%M') for i in data_cut['snsr_datetime']]
        plt.xticks(data_cut.index[::math.floor(3600/sample_time)],xdata[::math.floor(3600/sample_time)])        
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
            data_sliced.iloc[:,i] = data_sliced.iloc[:,i] - self.t_oil
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
        t_amb = kwargs.get('t_amb','t_env')
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
        print(xdata,ydata,t0,tw)
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
        ax1.set_title('{} time constant is {:.0f}'.format(col_name,const))
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
    
    def data_filter(self,col_name,low_lim,high_lim,diff_lim):
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
        diff = np.diff(self.data[col_name].values,prepend=self.data.loc[0,col_name])
        for j in range(len(self.data)):
            if  ((self.data.loc[j,col_name] > high_lim) or 
                (self.data.loc[j,col_name] < low_lim)):
                self.data.loc[j,col_name] = np.nan

            else:
                pass
        for k in range(len(self.data)-1):
            if 	(((diff[k] > diff_lim) or 
                (diff[k] < -diff_lim)) & 
                ((diff[k+1] > diff_lim) or 
                (diff[k+1] < -diff_lim))):
                self.data.loc[k,col_name] = np.nan
       

class DataClean(object):

    def read_sensor_data(path):
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

    # ====================================================================================
    # 读取热电偶数据
    # ====================================================================================

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
                               na_values=['          ', '     -OVER'])

        raw_data.iloc[:, 4:].astype(float)
        raw_data = raw_data.fillna(0)
        raw_data['datetime'] = [datetime.strptime(raw_data.iloc[i, 1] + ':' + raw_data.iloc[i, 2],
                                                  '%Y/%m/%d:%H:%M:%S') for i in range(len(raw_data.iloc[:, 2]))]

        # check if the sample time is 10s, if not down-sample to 10s
        time_interval = (raw_data.iloc[1, -1] - raw_data.iloc[0, -1]).seconds
        if time_interval != 10:
            multiple = 10 / time_interval
            raw_data = raw_data[raw_data.index % multiple == 0]
        else:
            pass

        return raw_data

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
                print('len element',len(element))
                print('k',k)
                if element.iloc[k, -1] == element.iloc[k + 1, -1]:
                    element = element.drop(k + 1)
                    count += 1
                else:
                    pass
                k = k + 1
            print('count',count)
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
                element_full = pd.DataFrame(data=0, columns=range(
                    element.shape[1]), index=range(len(full_index)))

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

    @staticmethod
    def down_sample(data, ratio):
        index = np.arange(len(data))
        data_out = data[index % ratio == 0]
        return data_out

# %%
