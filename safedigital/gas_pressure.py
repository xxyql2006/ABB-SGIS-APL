import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


class GasPressureExperiment(object):

	# plot regression result
	def plot_reg_result(y_true,y_est,**prop):
					
		sigma = round((np.mean((y_est - y_true) ** 2)) ** 0.5,1)
		max_abs = round(max(np.abs(y_est - y_true)),1)
		# print('sigma = ',sigma)
		plt.figure()
		ax1 = plt.gca()
		ax1.plot(y_est,color ='g',linewidth=0.5,label='estimated with sigma = %.2f,max_abs = %.2f' % (sigma,max_abs))
		ax1.plot(y_true,color ='k',label='true',linewidth=0.5)
		ax1.set_xlabel('test timeline')
		ax1.set_ylabel('t_sf6')
		ax2 = ax1.twinx()
		ax2.plot(y_est - y_true,color ='r',label='error',linewidth=0.5)   
		ax2.set_ylabel('error between est and true')     
		ax1.legend(loc="lower left",fontsize=6)
		ax2.legend(loc="lower right",fontsize=6)		
		plt.title(title)
		plt.grid()
		plt.tight_layout()
		if prop['save'] == True:
			plt.savefig(folder + '\\' + "%s.png"%(title),dpi=200)
		else:
			pass
		if prop['show'] == False:
			plt.close()
		else:
			pass	
		return sigma,max_abs

	# plot comparison of different fitting approaches
	def plot_p20_result(p_meter,p20_meter,p20_tank,*p20_est,**prop):
		
		# 子图1：比较P20
		title_veri_date = prop['title_veri_date']
		title_sample_time = prop['title_sample_time']
		folder = prop['folder']
		plt.figure()
		plt.subplot(2,1,1)
		plt.plot(p_meter,color='r',linewidth=0.5,label='P20 uncompensated')
		plt.plot(p20_meter,color ='y',linewidth=0.5,label='P20 from manometer')
		plt.plot(p20_tank,color ='k',linewidth=0.5,label='P20 tank mid')
		label_list = prop['est_name_list']
		for i,element in enumerate(p20_est):
			plt.plot(element,linewidth=0.5,label=label_list[i])
		plt.xlabel('test time in number of sampling interval')
		plt.ylabel('p20 in bar')
		plt.title(str(title_veri_date) + title_sample_time + 'P20 Comparison')
		plt.grid()
		plt.tight_layout()
		plt.legend(loc="lower right",fontsize=4)
		
		# 子图2：比较各种算法预测P20,气压表P20以及不进行拟合的P与气箱中温度补偿P20_tank的差值
		plt.subplot(2,1,2)
		for i,element in enumerate(p20_est):
			plt.plot(element-p20_tank,linewidth=0.5,label=label_list[i])
		plt.plot(p_meter-p20_tank,linewidth=0.5,label='p_meter')
		plt.plot(p20_meter-p20_tank,linewidth=0.5,label='p20_meter')
		plt.xlabel('test time in number of sampling interval')
		plt.ylabel('p20 difference in bar')			
		plt.title(str(title_veri_date) + title_sample_time + 'P20 Difference Comparison')
		plt.grid()
		plt.tight_layout()
		plt.legend(loc="lower right",fontsize=4)	

		if prop['save'] == True:
			plt.savefig(folder + '\\' + "P20_plot_%s_%s.png" % (str(title_veri_date),title_sample_time),dpi=200)
		else:
			pass
		if prop['show'] == False:
			plt.close()
		else:
			pass	

	@staticmethod
	def down_sample(data,ratio):
		index = np.arange(len(data))
		data_out = data[index%ratio == 0]
		return data_out
	
	@staticmethod
	def poly_fea(deg,x):
		poly = PolynomialFeatures(degree=deg,include_bias=False,interaction_only=False)	
		x_polyfea = poly.fit_transform(x)
		return x_polyfea

	# calculate polynomial coefficients
	@staticmethod
	def cal_polyfit_coef(order, y, x):
		coef = np.polyfit(x, y, order)
		return coef

	# calculate least square regression coefficients
	@staticmethod
	def cal_lsr_reg_coef(y, x):
		reg = linear_model.LinearRegression()
		reg.fit(x, y)
		coef = np.append(reg.coef_, reg.intercept_, axis=None)

		return coef

	# calculate ridge regression coefficients
	@staticmethod
	def cal_ridge_reg_coef(al, y, x):
		reg = linear_model.Ridge(alpha=al)
		reg.fit(x, y)
		coef = np.append(reg.coef_, reg.intercept_, axis=None)

		return coef