import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from toolbox import cal_autocorr, ADF_Cal, Cal_rolling_mean_var, order_one_diff

np.random.seed(6313)
e = np.random.normal(0,1,1000)
num = [1, 0]
den = [1, 0.5]
system = (num, den, 1)
t, y = signal.dlsim(system, e)
print(f'y() {y[:3]}')
print(f' the experimental variance of y is {np.var(y):.4f}')

lag = 20
cal_autocorr(y,lag,'AR(1)')
plt.show()

# ============================================
# ARIMA
num = [1, -0.25, 0]
den = [1, -1.5, 0.5]
system = (num, den, 1)
t, y = signal.dlsim(system, e)
print(f'y() {y[:3]}')

# ================================================
# Arima model
num = [1, -0.25, 0, 0]
den = [1, -2.5, 2, -0.5]
system = (num, den, 1)
_, y = signal.dlsim(system, e)
date = pd.date_range(start='1-1-2018',periods=len(y),freq='D')
df = pd.DataFrame(y,index = date,columns=['Value'])
df.plot()

plt.show()
y = y.reshape(1000)
x = pd.Series(y,name="value")
Cal_rolling_mean_var(x)
cal_autocorr(x,20,'ARIMA(1,2,1)')
plt.show()

# Differencing this twice

diff_1 = order_one_diff(df,'Value')
diff_2 = order_one_diff(diff_1,'Value_Diff_1')
diff_2 = diff_2.dropna().reset_index()
cal_autocorr(diff_2['Value_Diff_2'],20,'ARIMA(1,2,1) after 2 differences')
plt.show()
Cal_rolling_mean_var(diff_2['Value_Diff_2'])

# ==================
# Alternate way of simulating synthetic processes
import statsmodels.api as sm
num = [-0.25, 0, 0]
den = [-2.5, 2, -0.5]
na = 3
na = 1
ar = np.r_[1,den]
ma = np.r_[1,num]

arma_process = sm.tsa.ArmaProcess(ar,ma)
arma_process.isstationary

y = arma_process.generate_sample(1000, scale = 1 + 0) # scale = variance, 0 = mean
# GPAC we're going to use the last sampling method