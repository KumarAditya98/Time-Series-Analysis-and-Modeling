import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# sys.path.append('../toolbox')
# from toolbox import Cal_autocorr
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT', 'FRD']
start_date = '2000-01-01'
end_date = '2023-02-02'
df = data.get_data_yahoo(stocks[0],start = start_date,
                                    end = end_date)
print(df.tail().to_string())
lags = 50
title = 'Apple stock'
# Visualization
col = df.columns

df[col[3]].plot()
plt.grid()
plt.show()
print(df.describe().to_string())
ryy = Cal_autocorr(df[col[3]].values,lags, title)

# np.random.seed(6313)
# N = 10000
# lags = 50
# title = 'white noise'
# e = np.random.normal(0,1,N)
# ree = Cal_autocorr(e,lags, title)

# simulation of AR process
#=====
# y(t) + 0.9y(t-1) = e(t) ----e(t)~WN(0,1)
#===
# title = 'AR process'
# y = np.zeros(len(e))
# for t in range(len(e)):
#     if t==0:
#         y[t] = e[t]
#     else:
#         y[t] = 0.9*y[t-1] + e[t]
# ryy = Cal_autocorr(y,lags, title)
# # y = [1,2,3,4,5]
# lags = 5
# title = 'dummy'
# ryy = Cal_autocorr(y,lags, title)