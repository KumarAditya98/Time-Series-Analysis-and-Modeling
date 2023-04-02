import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\GW\Time series Analysis\toolbox')
from toolbox import Cal_autocorr, Plot_Rolling_Mean_Var, ADF_Cal, kpss_test, difference

import statsmodels.api as sm


np.random.seed(6313)
mean_e = 0
var_e = 1
N = 1000
e = np.random.normal(mean_e,var_e, N)

num = [-.25, 0, 0]
den = [-2.5, 2, -0.5]

na = 3
nb = 1
ar = np.r_[1,den]
ma = np.r_[1,num]

arma_process = sm.tsa.ArmaProcess(ar, ma)
print("Is this a stationary process : ",arma_process.isstationary)

y = arma_process.generate_sample(N, scale=1) + mean_e


# system = (num, den,1)
# _,y = signal.dlsim(system, e)
# print(f'the experimental variance of the y is {np.var(y):.4f}')
# plt.figure()
# plt.plot(y)
# plt.show()
# Plot_Rolling_Mean_Var(y,'original')
# ADF_Cal(y)
#
#
# y_diff_01 = difference(y,1)
# y_diff_02 = difference(y_diff_01,1)
# ADF_Cal(y_diff_02)
# Plot_Rolling_Mean_Var(y_diff_02,'y_diff')
# ryy = Cal_autocorr(y_diff_02,20, 'y')
#
# date = pd.date_range(start = '1-1-2018',
#                      periods=len(e),
#                      freq = 'D')
# df1 = pd.DataFrame(y[2:],
#                   columns=['original'],
#                   index=date[2:])
# df2 = pd.DataFrame(y_diff_02,
#                   columns=['2nd order diff'],
#                   index=date[2:])
# df = pd.concat([df1,df2], axis=1)
# df.plot()
# plt.grid()
# plt.show()
# num = [1, .25]

# num = [1, .25]
# den = [1, 0.5]
# system = (num, den,1)
# _,y = signal.dlsim(system, e)
# print(f'the experimental variance of the y is {np.var(y):.4f}')
# ryy = Cal_autocorr(y,20, 'y')
# # num = [1, .25]
# den = [1, 0]
# system = (num, den,1)
# _,y = signal.dlsim(system, e)
# print(f'the experimental variance of the y is {np.var(y):.4f}')
# ryy = Cal_autocorr(y,20, 'y')
# num = [1, 0]
# den = [1, 0.5]
# system = (num, den,1)
# _,y = signal.dlsim(system, e)
# print(f'the experimental variance of the y is {np.var(y):.4f}')
# ryy = Cal_autocorr(y,20, 'y')