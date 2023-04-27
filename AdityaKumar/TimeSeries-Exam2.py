import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toolbox
import statsmodels as sm
from scipy import signal

# Question 1
df1 = pd.read_csv('question1.csv',index_col='Unnamed: 0')
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(2,1,figsize=(14,8))
durations_1 = np.array(df1[df1['type']==1]['time']).tolist()
durations_2 = np.array(df1[df1['type']==2]['time']).tolist()
event_1 = np.array(df1[df1['type']==1]['delta']).tolist()
event_2 = np.array(df1[df1['type']==2]['delta']).tolist()
kmf1 = KaplanMeierFitter()
kmf1.fit(durations_1,event_1, label = 'Number of Survivers having Aneuploid Tumor')
kmf1.plot_survival_function(ax = ax[0])
kmf2 = KaplanMeierFitter()
kmf2.fit(durations_2,event_2, label = 'Number of Survivers having Diploid Tumor')
kmf2.plot_survival_function(ax = ax[1])
ax[0].set_xlabel('Time')
ax[1].set_xlabel('Time')
ax[1].set_xlabel('Time')
ax[0].set_ylabel('Survival Function: Aneuploid Tumor')
ax[1].set_ylabel('Survival Function: Diploid Tumor')
fig.suptitle("Survival Rate of Patients suffering from Tumors")
plt.grid()
plt.show()

# Question2
df2 = pd.read_csv('question2.csv',names=['values'])
date = pd.date_range(start="01/01/1981",periods=len(df2),freq='D')
df2.index = pd.to_datetime(date)

# a)
df2.plot()
plt.title("Time series observations")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#b)
toolbox.Cal_rolling_mean_var(df2['values'])
toolbox.ADF_Cal(df2['values'])
toolbox.kpss_test(df2['values'])

#c)
from statsmodels.tsa.seasonal import STL
STL = STL(df2['values'],period=365)
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid
fig = res.plot()
plt.show()
SOT = max(0,(1-((np.var(R))/(np.var(R+T)))))
print(f"The strength of trend for this data set is: {SOT*100:.2f}%")
SOS = max(0,(1-((np.var(R))/(np.var(R+S)))))
print(f"The strength of seasonality for this data set is: {SOS*100:.2f}%")

# d) performing differencing to make the dataset stationary
df2_diff = toolbox.order_one_diff(df2,'values')
temp = df2_diff[df2_diff['values_Diff_1'].notna()]
toolbox.Cal_rolling_mean_var(temp['values_Diff_1'])
toolbox.ADF_Cal(temp['values_Diff_1'])
toolbox.kpss_test(temp['values_Diff_1'])

# e)
ry = sm.tsa.stattools.acf(temp['values_Diff_1'], nlags=50)
ryy = ry[::-1].tolist()
Ry = ryy[:-1] + ry[:].tolist()
toolbox.Cal_GPAC(Ry)

# f)
toolbox.ACF_PACF_Plot(temp['values_Diff_1'],50)

#g)
toolbox.lm_param_estimate(temp['values_Diff_1'],1,0)

# Question3
df3 = pd.read_csv('question3.csv')

# a)
toolbox.Cal_rolling_mean_var(df3['y'])
toolbox.ADF_Cal(df3['y'])
toolbox.kpss_test(df3['y'])

# b)
df3_diff = toolbox.diff(df3,'y',3)
temp = df3_diff[df3_diff['y_3_Diff'].notna()]
toolbox.Cal_rolling_mean_var(temp['y_3_Diff'])
toolbox.ADF_Cal(temp['y_3_Diff'])
toolbox.kpss_test(temp['y_3_Diff'])

#c)
ry = sm.tsa.stattools.acf(temp['y_3_Diff'], nlags=50)
ryy = ry[::-1].tolist()
Ry = ryy[:-1] + ry[:].tolist()
toolbox.Cal_GPAC(Ry)

# d)
toolbox.ACF_PACF_Plot(temp['y_3_Diff'],50)

# e)
toolbox.lm_param_estimate(temp['y_3_Diff'],3,0)

# f)
df3.plot()
plt.title("Time series observations")
plt.xlabel("Sample")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

df3.iloc[:50].plot()
plt.title("Time series observations")
plt.xlabel("Sample")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()