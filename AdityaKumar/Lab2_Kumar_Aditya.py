import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

# import sys
# sys.path.append('../toolbox')

from toolbox import cal_autocorr

# Q2. Random White Noise
np.random.seed(6313)
x = np.random.normal(0,1,1000)
date = pd.date_range(start = '1/1/2000',
                     end = '12/31/2000',
                     periods = len(x))
df = pd.DataFrame(x, columns = ['WN_value'])
df.index = date

df.plot()
plt.grid()
plt.title('Time Series Plot for Random Noise')
plt.xlabel('Time')
plt.ylabel('Random Noise Value')
plt.show()

plt.hist(x,bins=20,label='Histogram of WN',edgecolor='black', linewidth=1.2)
plt.title('Histogram of Random WN')
plt.xlabel('Value of Random WN')
plt.ylabel('Frequency')
plt.show()

print(f'The Mean of generated Random Noise is : {round(df.WN_value.mean(),3)} and Standard Deviation is : {round(np.std(df.WN_value),2)}')

# Q3. ACF Function has been created in my Python File called toolbox. This Python file will be part of the zipped folder.
# a)
# Retrieving the make-up dataset in Q1
array = [3,9,27,81,243]
lags = 4
title = 'Make-up Dataset'

# Plotting the ACF
cal_autocorr(array,lags,title)
plt.show()
# Comparison will be provided in the report

# b)
lags = 20
title = 'Random White Noise'

# Plotting the ACF of the Generated Random White Noise
cal_autocorr(df.WN_value.values,lags,title)
plt.show()

# c) To answer this question, i will make use of the datasets we used in lab1. Since i know that ACF of white noise is an impulse, i can't say much. Observations will be written in the report.

# Fetching the tute1 dataset since I know all columns were stationary, and also fetching the AirPassengers dataset since it was not stationary

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df1 = pd.read_csv(url,index_col=0)

url1 = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv'
df2 = pd.read_csv(url1,index_col=0)

lags = 20
title1 = 'Sales'
title2 = 'Passengers'

cal_autocorr(df1.Sales.values,lags,title1)
plt.show()
cal_autocorr(df2['#Passengers'].values,lags,title2)
plt.show()
# Observations will be noted down in the report

# Q4. Retrieving the following stocks from Yahoo API

stocks = ['AAPL','ORCL', 'TSLA','IBM','YELP','MSFT']
start_date = '2000-01-01'
end_date = '2023-02-01'
df0 = data.get_data_yahoo(stocks[0],start = start_date,
                                    end = end_date)
df1 = data.get_data_yahoo(stocks[1],start = start_date,
                                    end = end_date)
df2 = data.get_data_yahoo(stocks[2],start = start_date,
                                    end = end_date)
df3 = data.get_data_yahoo(stocks[3],start = start_date,
                                    end = end_date)
df4 = data.get_data_yahoo(stocks[4],start = start_date,
                                    end = end_date)
df5 = data.get_data_yahoo(stocks[5],start = start_date,
                                    end = end_date)
df = [df0,df1,df2,df3,df4,df5]

# a) Plotting the Time-Series data for each graph
fig, axs = plt.subplots(3,2,figsize=(16,8))
count = 0
for i in range(3):
    for j in range(2):
        df[count].Close.plot(ax=axs[i, j])
        axs[i,j].grid()
        axs[i,j].set_title('Closing Price of ' + stocks[count] + ' over the years')
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Closing Price of Stock')
        plt.tight_layout()
        count+=1
plt.show()

# b) Plotting the ACF of all the functions
lags = 50
fig, axs = plt.subplots(3,2,figsize=(16,8))
count = 0
for i in range(3):
    for j in range(2):
        cal_autocorr(df[count].Close.values,lags,stocks[count],ax=axs[i,j])
        count+=1
plt.tight_layout()
plt.show()

# Confirming whether this time-series is stationary or non-stationary
from toolbox import Cal_rolling_mean_var
Cal_rolling_mean_var(df1.Close)
Cal_rolling_mean_var(df2.Close)

