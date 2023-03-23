import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
# import sys
# sys.path.append('../toolbox')
from toolbox import cal_autocorr, ADF_Cal, Cal_rolling_mean_var

import numpy as np
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv'
df = pd.read_csv(url, index_col='Date',)
date = pd.date_range(start = '1981-01-01',
                    periods = len(df),
                    freq='D')
df.index = date
plt.figure(figsize=(12,10))
df.plot()
plt.tight_layout()
plt.grid()
plt.show()

Temp = pd.Series(df['Temp'].values,index = date,
                 name = 'daily-temp')
#=====================
# STL Decomposition
#=====================
STL = STL(Temp)
res = STL.fit()
fig = res.plot()
plt.show()


T = res.trend
S = res.seasonal
R = res.resid



# lags = 100
# title = 'daily temp'
# ryy = Cal_autocorr(df.values,lags, title)
#
# ADF_Cal(df.values)
#
# Plot_Rolling_Mean_Var(df.values, title)
plt.figure()
plt.plot(T[:50].values,label = 'trend')
plt.plot(S[:50].values,label = 'Seasonal')
plt.plot(R[:50].values,label = 'residuals')
plt.plot(df['Temp'][:50].values, label = 'original data')
plt.legend()
plt.show()