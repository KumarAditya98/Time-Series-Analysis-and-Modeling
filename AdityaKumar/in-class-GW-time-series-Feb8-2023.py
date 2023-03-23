import pandas as pd
import sys
sys.path.append('../toolbox')
from toolbox import ACF_PACF_Plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv'
df = pd.read_csv(url, index_col='Month', parse_dates= True)
print(df.head())
y = df["#Passengers"]
lags = 40
ACF_PACF_Plot(y, lags)

# split train-test 80-20
yt, yf = train_test_split(y, shuffle= False, test_size=0.2)
#---------------
# SES Method
#---------------
holtt = ets.ExponentialSmoothing(yt,trend=None,damped=False,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for simple exponential smoothing is ", MSE)

fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Simple Exponential Smoothing")
plt.legend(loc='upper left')
plt.title(f'Simple Exponential Smoothing- MSE = {MSE:.2f}')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.grid()
plt.show()

#---------------
# Holt-linear Method
#---------------
holtt = ets.ExponentialSmoothing(yt,trend='mul',damped=True,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for Holt-linear is ", MSE)

fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Holt-linear")
plt.legend(loc='upper left')
plt.title(f'Holt-linear- MSE = {MSE:.2f}')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.grid()
plt.show()

#---------------
# Holt-Winter Method
#---------------
holtt = ets.ExponentialSmoothing(yt,trend='mul',damped=True,seasonal='mul').fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for Holt-Winter is ", MSE)

fig, ax = plt.subplots()
ax.plot(yt,label= "Train Data")
ax.plot(yf,label= "Test Data")
ax.plot(holtf,label= "Holt-Winter")
plt.legend(loc='upper left')
plt.title(f'Holt-Winter- MSE = {MSE:.2f}')
plt.xlabel('Time (monthly)')
plt.ylabel('# of Passengers (thousands)')
plt.grid()
plt.show()