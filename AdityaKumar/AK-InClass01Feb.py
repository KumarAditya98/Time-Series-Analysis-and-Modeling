import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# sys.path.append('/toolbox') # Folder for saving tool box in hardrive anywhere.
# from toolbox import

# Creating a random white noise.
np.random.seed(6313)
N = 1000
e = np.random.normal(0,1,N)

# Simulation of AR(Auto regressive) process
#======
# y(t) + 0.9y(t-1) = e(t)---- e(t) ~ WN(0,1)
#======
y = np.zeros(len(e))
for t in range(len(e)):
    if t == 0:
        y[t] = e[t]
    else:
        y[t] = -0.9y[t-1]+e[t]

# plt.stem command for plotting autocorrelation function.

# How to make dataset symmetric
#.expanding()
ry = [1,0.4,-0.1,-0.4,-0.4]
ryy = ry[::-1]
Ry = ryy[:-1] + ry

# Stocks api to get daily stocks
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

stocks = ['AAPL','ORCL','TSLA','IBM','YELP','MSFT']
start_date = '2000-01-01'
end_date = '2023-02-01'
df = data.get_data_yahoo(stocks[0],start=start_date,end=end_date)

print(df.tail().to_string())
print(df.describe().to_string())

col = df.columns

df[col[3]].plot()
plt.show()

fig, ax = plt.subplot(321,figsize = (16,8))
ax[0,0].plot(df[col[3]])
plt.show()
ax[0]



