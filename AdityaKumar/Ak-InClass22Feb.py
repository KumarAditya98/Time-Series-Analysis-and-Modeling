import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from toolbox import Cal_rolling_mean_var, cal_autocorr


url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv'

df = pd.read_csv(url,index_col=0,parse_dates=True)
df.index = pd.to_datetime(df.index)

df.plot()
plt.grid()
plt.title('Daily Minimum Temperatures')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.tight_layout()
plt.show()

# def cal_autocorr(array,lag,title,ax=None):
#     if ax == None:
#         ax = plt.gca()
#     mean = np.mean(array)
#     denominator = 0
#     for x in range(len(array)):
#         denominator = (array[x] - mean)**2 + denominator
#     ry = []
#     for tau in range(lag):
#         numerator = 0
#         for t in range(tau,len(array)):
#             numerator = (array[t]-mean)*(array[t-tau]-mean) + numerator
#         val = numerator/denominator
#         ry.append(val)
#     ryy = ry[::-1]
#     Ry = ryy[:-1] + ry[:]
#
#     bins = np.linspace(-(len(ryy) - 1), len(ryy) - 1, len(Ry))
#     ax.stem(bins,Ry,markerfmt='ro',basefmt='C5')
#     ax.axhspan(1.96/len(array)**0.5,-1.96/len(array)**0.5,alpha = 0.2, color = 'blue')
#     ax.locator_params(axis='x', tight = True, nbins=9)
#     ax.set_title('Autocorrelation Function of ' + title)
#     ax.set_xlabel('Lags')
#     ax.set_ylabel('Magnitude')

lags = 50
cal_autocorr(df.Temp.values,lags,'Temperature')
plt.show()

# def Cal_rolling_mean_var(x):
#     rMean = []
#     rVariance = []
#     for i in range(len(x)):
#         mean = x.iloc[0:i].mean()
#         rMean.append(mean)
#         variance = np.var(x.iloc[0:i])
#         rVariance.append(variance)
#     fig, ax = plt.subplots(2,1, figsize=(12,12))
#     ax[0].plot(rMean)
#     ax[0].set(xlabel="Samples",ylabel="Magnitude")
#     name1 = 'Rolling Mean - ' + x.name
#     ax[0].set_title(name1)
#     ax[0].legend(['Varying Mean'])
#     ax[1].plot(rVariance)
#     ax[1].set(xlabel="Samples", ylabel="Magnitude")
#     name2 = 'Rolling Variance - ' + x.name
#     ax[1].set_title(name2)
#     ax[1].legend(['Varying Variance'])
#     plt.show()

Cal_rolling_mean_var(df.Temp)
from statsmodels.tsa.seasonal import STL
date = pd.date_range(start='1981-01-01',
                     periods=len(df))
Temp = pd.Series(df.Temp.values, index=date,name = 'Daily-Temp')
# -------------
# STL Decomposition
STL = STL(Temp)
res = STL.fit()
fig = res.plot()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

F_s = max(0,(1-(np.var(R)/(np.var(R+T)))))
F_s*100

F_t = max(0,(1-(np.var(R)/(np.var(S+T)))))
F_t*100

