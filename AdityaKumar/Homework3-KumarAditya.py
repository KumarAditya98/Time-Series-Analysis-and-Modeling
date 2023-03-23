import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from toolbox import ADF_Cal

# Q1.
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/daily-min-temperatures.csv'
df = pd.read_csv(url, index_col='Date',)
date = pd.date_range(start = '1981-01-01',
                    periods = len(df),
                    freq='D')
df.index = date

def Cal_moving_avg(series):
    """
    :param series: Type of argument should be Series only
    :return: Calculated Moving/Weighted Average corresponding to the index positions
    :Note: In this code, for m = even moving average, expectation for fold is always 2. Will generalize this later with more time.
    """
    m = int(input("Enter the order for moving average:"))
    if m == 1 or m == 2:
        print("Order = 1 or Order = 2 will not be accepted. Run program again.")
        return None
    if m%2 == 0:
        fold = int(input("Enter the folding order which should be even:"))
        if fold%2 != 0:
            print("Incorrect input. Run program again.")
            return None
        k = int(m/2)
        transform = []
        for i in range(k,len(series)-k):
            value = (1/(2*m))*(series.iloc[i-k] + series.iloc[i+k]) + (1/m)*(np.sum(series[i-(k-1):i+k]))
            transform.append(value)
        index = series[k:len(series)-k].index.values
        final = pd.Series(transform, index=index)
        return final

    elif m%2 != 0:
        k = int((m-1)/2)
        transform = []
        for i in range(k,len(series)-k):
            avg = np.average(series[i-k:i+k+1])
            transform.append(avg)
        index = series[k:len(series)-k].index.values
        final = pd.Series(transform, index = index)
        return final


# Q2.
Temp = pd.Series(df['Temp'].values,index = date,
                 name = 'daily-temp')
date_50 = pd.date_range(start = '1981-01-01',
                    periods = 5,
                    freq='SM')

ma3 = Cal_moving_avg(Temp)
ma5 = Cal_moving_avg(Temp)
ma7 = Cal_moving_avg(Temp)
ma9 = Cal_moving_avg(Temp)

detrend_ma3_add = Temp - ma3
detrend_ma5_add = Temp - ma5
detrend_ma7_add = Temp - ma7
detrend_ma9_add = Temp - ma9

detrend_ma3_mul = Temp/ma3
detrend_ma5_mul = Temp/ma5
detrend_ma7_mul = Temp/ma7
detrend_ma9_mul = Temp/ma9

fig, ax = plt.subplots(2,2,figsize=(16,8))
Temp[:50].plot(ax = ax[0,0],label = "Original Data")
ma3[:50].plot(ax=ax[0,0],label = "3 - MA")
detrend_ma3_add[:50].plot(ax =ax[0,0],label = "De-trended Data")
ax[0,0].grid()
ax[0,0].set_title('3 - Moving Average')
ax[0,0].set_xlabel('Time (t)')
ax[0,0].set_ylabel('Daily-Min-Temp')
ax[0,0].legend(loc = "center right")
Temp[:50].plot(ax =ax[0,1],label = "Original Data")
ma5[:50].plot(ax =ax[0,1],label = "5 - MA")
detrend_ma5_add[:50].plot(ax =ax[0,1],label = "De-trended Data")
ax[0,1].grid()
ax[0,1].set_title('5 - Moving Average')
ax[0,1].set_xlabel('Time (t)')
ax[0,1].set_ylabel('Daily-Min-Temp')
ax[0,1].legend(loc="center right")
Temp[:50].plot(ax =ax[1,0],label = "Original Data")
ma7[:50].plot(ax =ax[1,0],label = "7 - MA")
detrend_ma7_add[:50].plot(ax =ax[1,0],label = "De-trended Data")
ax[1,0].grid()
ax[1,0].set_title('7 - Moving Average')
ax[1,0].set_xlabel('Time (t)')
ax[1,0].set_ylabel('Daily-Min-Temp')
ax[1,0].legend(loc="center right")
Temp[:50].plot(ax =ax[1,1],label = "Original Data")
ma9[:50].plot(ax =ax[1,1],label = "9 - MA")
detrend_ma9_add[:50].plot(ax =ax[1,1],label = "De-trended Data")
ax[1,1].grid()
ax[1,1].set_title('9 - Moving Average')
ax[1,1].set_xlabel('Time (t)')
ax[1,1].set_ylabel('Daily-Min-Temp')
ax[1,1].legend(loc="center right")
plt.tight_layout()
fig.suptitle("Odd Order Moving Average - Additive Decomposition")
plt.show()

fig, ax = plt.subplots(2,2,figsize=(16,8))
Temp[:50].plot(ax = ax[0,0],label = "Original Data")
ma3[:50].plot(ax = ax[0,0],label = "3 - MA")
detrend_ma3_mul[:50].plot(ax = ax[0,0],label = "De-trended Data")
ax[0,0].grid()
ax[0,0].set_title('3 - Moving Average')
ax[0,0].set_xlabel('Time (t)')
ax[0,0].set_ylabel('Daily-Min-Temp')
ax[0,0].legend(loc = "lower right")
Temp[:50].plot(ax=ax[0,1],label = "Original Data")
ma5[:50].plot(ax=ax[0,1],label = "5 - MA")
detrend_ma5_mul[:50].plot(ax=ax[0,1],label = "De-trended Data")
ax[0,1].grid()
ax[0,1].set_title('5 - Moving Average')
ax[0,1].set_xlabel('Time (t)')
ax[0,1].set_ylabel('Daily-Min-Temp')
ax[0,1].legend(loc = "lower right")
Temp[:50].plot(ax=ax[1,0],label = "Original Data")
ma7[:50].plot(ax=ax[1,0],label = "7 - MA")
detrend_ma7_mul[:50].plot(ax=ax[1,0],label = "De-trended Data")
ax[1,0].grid()
ax[1,0].set_title('7 - Moving Average')
ax[1,0].set_xlabel('Time (t)')
ax[1,0].set_ylabel('Daily-Min-Temp')
ax[1,0].legend(loc = "lower right")
Temp[:50].plot(ax=ax[1,1],label = "Original Data")
ma9[:50].plot(ax=ax[1,1],label = "9 - MA")
detrend_ma9_mul[:50].plot(ax=ax[1,1],label = "De-trended Data")
ax[1,1].grid()
ax[1,1].set_title('9 - Moving Average')
ax[1,1].set_xlabel('Time (t)')
ax[1,1].set_ylabel('Daily-Min-Temp')
ax[1,1].legend(loc = "lower right")
plt.tight_layout()
fig.suptitle("Odd Order Moving Average - Multiplicative Decomposition")
plt.show()


# Q3.
ma4 = Cal_moving_avg(Temp)
ma6 = Cal_moving_avg(Temp)
ma8 = Cal_moving_avg(Temp)
ma10 = Cal_moving_avg(Temp)

detrend_ma4_add = Temp - ma4
detrend_ma6_add = Temp - ma6
detrend_ma8_add = Temp - ma8
detrend_ma10_add = Temp - ma10

detrend_ma4_mul = Temp/ma4
detrend_ma6_mul = Temp/ma6
detrend_ma8_mul = Temp/ma8
detrend_ma10_mul = Temp/ma10

fig, ax = plt.subplots(2,2,figsize=(16,8))
Temp[:50].plot(ax=ax[0,0],label = "Original Data")
ma4[:50].plot(ax=ax[0,0],label = "2X4 - MA")
detrend_ma4_add[:50].plot(ax=ax[0,0],label = "De-trended Data")
ax[0,0].grid()
ax[0,0].set_title('2X4 - Moving Average')
ax[0,0].set_xlabel('Time (t)')
ax[0,0].set_ylabel('Daily-Min-Temp')
ax[0,0].legend(loc = "center right")
Temp[:50].plot(ax=ax[0,1],label = "Original Data")
ma6[:50].plot(ax=ax[0,1],label = "2X6 - MA")
detrend_ma6_add[:50].plot(ax=ax[0,1],label = "De-trended Data")
ax[0,1].grid()
ax[0,1].set_title('2X6 - Moving Average')
ax[0,1].set_xlabel('Time (t)')
ax[0,1].set_ylabel('Daily-Min-Temp')
ax[0,1].legend(loc = "center right")
Temp[:50].plot(ax=ax[1,0],label = "Original Data")
ma8[:50].plot(ax=ax[1,0],label = "2X8 - MA")
detrend_ma8_add[:50].plot(ax=ax[1,0],label = "De-trended Data")
ax[1,0].grid()
ax[1,0].set_title('2X8 - Moving Average')
ax[1,0].set_xlabel('Time (t)')
ax[1,0].set_ylabel('Daily-Min-Temp')
ax[1,0].legend(loc = "center right")
Temp[:50].plot(ax=ax[1,1],label = "Original Data")
ma10[:50].plot(ax=ax[1,1],label = "2X10 - MA")
detrend_ma10_add[:50].plot(ax=ax[1,1],label = "De-trended Data")
ax[1,1].grid()
ax[1,1].set_title('2X10 - Moving Average')
ax[1,1].set_xlabel('Time (t)')
ax[1,1].set_ylabel('Daily-Min-Temp')
ax[1,1].legend(loc="center right")
plt.tight_layout()
fig.suptitle("Even Order Moving Average - Additive Decomposition")
plt.show()

fig, ax = plt.subplots(2,2,figsize=(16,8))
Temp[:50].plot(ax=ax[0,0],label = "Original Data")
ma4[:50].plot(ax=ax[0,0],label = "2X4 - MA")
detrend_ma4_mul[:50].plot(ax=ax[0,0],label = "De-trended Data")
ax[0,0].grid()
ax[0,0].set_title('2X4 - Moving Average')
ax[0,0].set_xlabel('Time (t)')
ax[0,0].set_ylabel('Daily-Min-Temp')
ax[0,0].legend(loc = "center right")
Temp[:50].plot(ax=ax[0,1],label = "Original Data")
ma6[:50].plot(ax=ax[0,1],label = "2X6 - MA")
detrend_ma6_mul[:50].plot(ax=ax[0,1],label = "De-trended Data")
ax[0,1].grid()
ax[0,1].set_title('2X6 - Moving Average')
ax[0,1].set_xlabel('Time (t)')
ax[0,1].set_ylabel('Daily-Min-Temp')
ax[0,1].legend(loc="center right")
Temp[:50].plot(ax=ax[1,0],label = "Original Data")
ma8[:50].plot(ax=ax[1,0],label = "2X8 - MA")
detrend_ma8_mul[:50].plot(ax=ax[1,0],label = "De-trended Data")
ax[1,0].grid()
ax[1,0].set_title('2X8 - Moving Average')
ax[1,0].set_xlabel('Time (t)')
ax[1,0].set_ylabel('Daily-Min-Temp')
ax[1,0].legend(loc="center right")
Temp[:50].plot(ax=ax[1,1],label = "Original Data")
ma10[:50].plot(ax=ax[1,1],label = "2X10 - MA")
detrend_ma10_mul[:50].plot(ax=ax[1,1],label = "De-trended Data")
ax[1,1].grid()
ax[1,1].set_title('2X10 - Moving Average')
ax[1,1].set_xlabel('Time (t)')
ax[1,1].set_ylabel('Daily-Min-Temp')
ax[1,1].legend(loc="center right")
plt.tight_layout()
fig.suptitle("Even Order Moving Average - Multiplicative Decomposition")
plt.show()

# Q4.
print(f"ADF test for original dataset:")
ADF_Cal(Temp)
print("\n")
print(f"ADF test for 3-MA Additive de-trended data:")
ADF_Cal(detrend_ma3_add.dropna())

# Q5.
STL = STL(Temp)
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid

Temp.index = pd.to_datetime(Temp.index)

# First plotting the decomposition for 50 obervations
plt.figure()
T[:50].plot(label = 'Trend Component')
S[0:50].plot(label = 'Seasonality Component')
R[0:50].plot(label = 'Residuals Component')
Temp[0:50].plot(label = 'Original data',alpha = 0.6)
plt.title("Time Series Decomposition of the Data")
plt.xlabel("time (t)")
plt.ylabel("Daily-Min-Temp")
plt.legend(loc="center right")
plt.tight_layout()
plt.show()

# Plotting the time series decomposition for all observations
plt.figure(figsize=(16,8))
T.plot(label = 'Trend Component')
S.plot(label = 'Seasonality Component')
R.plot(label = 'Residuals Component')
Temp.plot(label = 'Original data',alpha = 0.6)
plt.title("Time Series Decomposition of the Data")
plt.xlabel("time (t)")
plt.ylabel("Daily-Min-Temp")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Q6.
Seasonal_adjust_add = Temp - S
Seasonal_adjust_mul = Temp/S

# For additive - Full observations
plt.figure(figsize=(16,8))
Temp.plot(label='Original Data')
Seasonal_adjust_add.plot(label="Seasonally adjusted Data",alpha=0.7)
plt.legend()
plt.title("Seasonally Adjusted Data - Additive Decomposition")
plt.xlabel("time (t)")
plt.ylabel("Daily-Min-Temp")
plt.tight_layout()
plt.show()

# For additive - 50 observations
plt.figure(figsize=(16,8))
Temp[:50].plot(label='Original Data')
Seasonal_adjust_add[:50].plot(label="Seasonally adjusted Data",alpha=0.7)
plt.legend()
plt.title("Seasonally Adjusted Data - Additive Decomposition")
plt.xlabel("time (t)")
plt.ylabel("Daily-Min-Temp")
plt.tight_layout()
plt.show()

# # For multiplicative - not including because non-insightful
# plt.figure()
# plt.plot(Temp,label='Original Data')
# plt.plot(Seasonal_adjust_mul,label="Seasonally adjusted Data")
# plt.legend()
# plt.title("Seasonally Adjusted Data - Multiplicative")
# plt.xlabel("time (t)")
# plt.ylabel("Daily-Min_Temp")
# plt.tight_layout()
# plt.show()

# Q7.
# To calculate strength of trend
SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
print(f"The strength of trend for this data set is: {SOT*100:.2f}%")

# Q8.
SOS = max(0,(1-((np.var(R))/(np.var(R+S)))))
print(f"The strength of seasonality for this data set is: {SOS*100:.2f}%")

# Q9. Answered in the report.