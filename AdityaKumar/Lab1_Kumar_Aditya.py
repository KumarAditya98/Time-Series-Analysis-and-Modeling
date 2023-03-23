import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# Reading the Dataset from a relative path
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url,index_col=0)

# Basic Data Exploration
print(df.head())
print(df.shape)
print(df.info())

# Converting index to datetime format for better visualization
df.index = pd.to_datetime(df.index)

# Q1. Plotting the time series data
df.plot()
plt.grid()
plt.title('1981 Time-Series Data')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('USD($)')
plt.tight_layout()
plt.show()


# Q2. The time-series statistics
print(f'The Sales mean is : {round(df.Sales.mean(),2)} and the variance is : {round(np.var(df.Sales),2)} with standard deviation : {round(np.std(df.Sales),2)} and median: {round(np.median(df.Sales),2)}')
print(f'The AdBudget mean is : {round(df.AdBudget.mean(),2)} and the variance is : {round(np.var(df.AdBudget),2)} with standard deviation : {round(np.std(df.AdBudget),2)} and median: {round(np.median(df.AdBudget),2)}')
print(f'The GDP mean is : {round(df.GDP.mean(),2)} and the variance is : {round(np.var(df.GDP),2)} with standard deviation : {round(np.std(df.GDP),2)} and median: {round(np.median(df.GDP),2)}\n')

# Q3. Visual Stationarity test with custom function - Custom function is placed here, but I will compile them all in a separate file called toolbox that i will use to call all such functions for later use.

def Cal_rolling_mean_var(x):
    rMean = []
    rVariance = []
    for i in range(len(x)):
        mean = x.iloc[0:i].mean()
        rMean.append(mean)
        variance = np.var(x.iloc[0:i])
        rVariance.append(variance)
    fig, ax = plt.subplots(2,1, figsize=(12,12))
    ax[0].plot(rMean)
    ax[0].set(xlabel="Samples",ylabel="Magnitude")
    name1 = 'Rolling Mean - ' + x.name
    ax[0].set_title(name1)
    ax[0].legend(['Varying Mean'])
    ax[1].plot(rVariance)
    ax[1].set(xlabel="Samples", ylabel="Magnitude")
    name2 = 'Rolling Variance - ' + x.name
    ax[1].set_title(name2)
    ax[1].legend(['Varying Variance'])
    plt.show()

# Visually evaluating Stationarity for Sales, AdBudget, and GDP
Cal_rolling_mean_var(df.Sales)
Cal_rolling_mean_var(df.AdBudget)
Cal_rolling_mean_var(df.GDP)

# Q4. The observations will be noted down in the report.

# Q5. Performing ADF test to see if Sales, AdBudget and GDP are objectively stationary

# Defining custom function first
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print(f"\nThe ADF Test results for Sales are: \n")
ADF_Cal(df.Sales)
print(f"\nThe ADF Test results for AdBudget are: \n")
ADF_Cal(df.AdBudget)
print(f"\nThe ADF Test results for GDP are: \n")
ADF_Cal(df.GDP)

# Q6. Performing KPSS test to see if Sales, AdBudget and GDP are objectively stationary

# Defining custom function first
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

print(f"\nThe KPSS Test results for Sales are: \n")
kpss_test(df.Sales)
print(f"\nThe KPSS Test results for AdBudget are: \n")
kpss_test(df.AdBudget)
print(f"\nThe KPSS Test results for GDP are: \n")
kpss_test(df.GDP)

# Q7. Repeating the above steps for AirPassengers.csv file
url1 = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv'
df1 = pd.read_csv(url1,index_col=0,parse_dates=True)

# Basic Data Exploration with new dataset
print(df1.head())
print(df1.shape)
print(df1.info())

# Changing column name for ease
df1.columns = ['Passengers']

# Plotting the time series data
df1.plot()
plt.grid()
plt.title('Time series trend - # of Air Passengers')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

# The time-series statistics
print(f'The # of Passengers mean is : {round(df1.Passengers.mean(),2)} and the variance is : {round(np.var(df1.Passengers),2)} with standard deviation : {round(np.std(df1.Passengers),2)} and median: {round(np.median(df1.Passengers),2)}')

# Visually evaluating Stationarity for # of Passengers with custom function
Cal_rolling_mean_var(df1.Passengers)
# The observations will be noted down in the report.

# Performing ADF test to see if # of Passengers is objectively stationary
print(f"\nThe ADF Test results for Passengers are: \n")
ADF_Cal(df1.Passengers)

# Performing KPSS test to see if # of Passengers is objectively stationary
print(f"\nThe KPSS Test results for Passengers are: \n")
kpss_test(df1.Passengers)

# Q8.
# a) Performing first order non-seasonal differencing.

# Creating a custom-function that calculates one step,non-seasonal differencing at a time, of specified column.

def order_one_diff(Df,col):
    order = '1'
    name = col + '_Diff_' + order
    if col[-7:-1]=='_Diff_':
        x = int(col[-1])
        order = str(x+1)
        name = col[0:-7] + '_Diff_' + order
    Df[name] = 0
    temp1 = Df[col][::-1]
    temp2 = temp1[0:-1] - temp1.values[1:]
    Df[name] = temp2[::-1]
    return Df

# Applying first order differencing to Passengers column
order_one_diff(df1,'Passengers')

# Visually evaluating Stationarity for first order Diff of # of Passengers with custom function
Cal_rolling_mean_var(df1.Passengers_Diff_1)
# The observations will be noted down in the report.

# Removing the top row since it now contains a missing value which will cause an error
df1 = df1[df1['Passengers_Diff_1'].notna()]

# Performing ADF test to see if first order diff of # of Passengers is objectively stationary
print(f"\nThe ADF Test results for first order Diff of Passengers are: \n")
ADF_Cal(df1.Passengers_Diff_1)

# Performing KPSS test to see if first order diff of # of Passengers is objectively stationary
print(f"\nThe KPSS Test results for first order Diff of Passengers are: \n")
kpss_test(df1.Passengers_Diff_1)

# b) Applying second order differencing to Passengers column
order_one_diff(df1,'Passengers_Diff_1')

# Removing the top row since it now contains a missing value which will cause an error
df1 = df1[df1['Passengers_Diff_2'].notna()]

# Visually evaluating Stationarity for second order Diff of # of Passengers with custom function
Cal_rolling_mean_var(df1.Passengers_Diff_2)
# The observations will be noted down in the report.

# Performing ADF test to see if second order diff of # of Passengers is objectively stationary
print(f"\nThe ADF Test results for second order Diff of Passengers are: \n")
ADF_Cal(df1.Passengers_Diff_2)

# Performing KPSS test to see if second order diff of # of Passengers is objectively stationary
print(f"\nThe KPSS Test results for second oder Diff of Passengers are: \n")
kpss_test(df1.Passengers_Diff_2)

# c) Applying third order differencing to Passengers column
order_one_diff(df1,'Passengers_Diff_2')

# Removing the top row since it now contains a missing value which will cause an error
df1 = df1[df1['Passengers_Diff_3'].notna()]

# Visually evaluating Stationarity for third order Diff of # of Passengers with custom function
Cal_rolling_mean_var(df1.Passengers_Diff_3)
# The observations will be noted down in the report.

# Performing ADF test to see if third order diff of # of Passengers is objectively stationary
print(f"\nThe ADF Test results for third order Diff of Passengers are: \n")
ADF_Cal(df1.Passengers_Diff_3)

# Performing KPSS test to see if third order diff of # of Passengers is objectively stationary
print(f"\nThe KPSS Test results for third order Diff of Passengers are: \n")
kpss_test(df1.Passengers_Diff_3)

# d) Loading the dataset again as we had removed few rows.

df2 = pd.read_csv(url1,index_col=0,parse_dates=True)
df2.columns = ['Passengers']
# Performing a log transformation, followed by first order-differencing.

df2['log_transform'] = np.log(df2.Passengers)
df3 = order_one_diff(df2,'log_transform')

# Visual inspection of stationarity
Cal_rolling_mean_var(df2.log_transform_Diff_1)

df3 = df3[df3['log_transform_Diff_1'].notna()]

# ADF test
print(f"\nThe ADF Test results are: \n")
ADF_Cal(df3.log_transform_Diff_1)

# Performing KPSS test to see if third order diff of # of Passengers is objectively stationary
print(f"\nThe KPSS Test results are: \n")
kpss_test(df3.log_transform_Diff_1)

from toolbox import cal_autocorr

lags = 20
title = 'title'
cal_autocorr(df3.log_transform_Diff_1.values,lags,title)
cal_autocorr(df1.Passengers_Diff_1.values,lags,title)
cal_autocorr(df1.Passengers_Diff_2.values,lags,title)