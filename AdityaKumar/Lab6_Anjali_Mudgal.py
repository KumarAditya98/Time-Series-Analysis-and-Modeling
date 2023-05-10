import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from scipy import signal
import numpy.linalg as LA
import matplotlib.pyplot as plt
import statsmodels.api as sm
#sys.path.append('/Users/anjalimudgal/Desktop/GWU/TimeSeries/FinalExam/finalExam_Anjali/toolbox.py')
from toolbox import *
np.random.seed(6313)

samples = int(input('Enter the number of data samples : '))
mean_wn = float(input('Enter the mean of white noise : '))
var_wn = float(input('Enter the variance of white noise : '))
samples = 1000
mean_wn = 0
var_wn = 1

n_a = int(input('Enter the AR order : '))
n_b = int(input('Enter the MA order : '))
ar_coef=[1]
ma_coef=[1]
for i in range(0,n_a):
    ar_coef.append(float(input('Enter a'+str(i+1) + ' :')))
print('')
for i in range(0,n_b):
    ma_coef.append(float(input('Enter b'+str(i+1)+' :')))

title = "Please select the title from code"
#example 1
#no differencing req
# n_a = 3
# n_b = 0
# ar_coef = [1,0,0,-0.5]
# ma_coef = [1]
# title = "(1-0.5q-3)y(t) = e(t)"

#no differencing required
#example 2
# n_a = 6
# n_b = 0
# ar_coef = [1,0,0,-0.5,0,0,0.6]
# ma_coef = [1]
# title = "(1 - 0.5q-3 + 0.6q-6) y(t)= e(t)"


#example3
#differencing required
# n_a = 6
# n_b = 0
# ar_coef = [1,0,0,-1.5,0,0,0.5]
# ma_coef = [1]
title = "(1 – 0.5q-3)(1 – q-3)y(t) = et"

#example 4
#differencing required
# n_a = 9
# n_b = 0
# ar_coef= [1,0,0,-1.5,0,0,1.1,0,0,-0.6]
# ma_coef = [1]
# title = "(1 –0.5q-3 +0.6q-6)(1 –q-3)y(t) = et"

#example 5
# n_a = 0
# n_b = 4
# ar_coef = [1]
# ma_coef = [1,0,0,-0.9]
# title = "y(t) = (1 -0.9q-4)et"

#example 6
# n_a = 0
# n_b = 8
# ar_coef = [1]
# ma_coef = [1,0,0,0,-0.2,0,0,0,-0.8]
# title = "y(t) = (1 -0.2q-4 -0.8q-8)et"

#example 7
# n_a = 4
# n_b = 4
# ar_coef = [1,0,0,0,-1]
# ma_coef = [1,0,0,0,-0.9]
# title = "(1- q-4)y(t) = (1- 0.9q-4)e(t)"

#example 8
# n_a = 4
# n_b = 8
# ar_coef = [1,0,0,0,-1]
# ma_coef = [1,0,0,0,-0.5,0,0,0,0.6]
# title = "(1- q-4)y(t) = (1- 0.5q-4 +0.6q-8)e(t)"

#example 9
#differencing required s=4
# n_a = 8
# n_b = 4
# ar_coef = [1,0,0,0,-1.5,0,0,0,0.5]
# ma_coef = [1,0,0,0,-0.2]
# title = "(1- 0.5q-4)(1- q-4)y(t) = (1- 0.2q-4)e(t)"

#example 10
#differencing required s=3 order 2
# n_a = 6
# n_b = 10
# ar_coef = [1,-2,1,-1,2,-1]
# ma_coef = [1,-0.2,0,0.35,-0.07]
# title = "(1- q-1)2(1- q-3)y(t) = (1- 0.2q-1)(1+0.35q-3)e(t)"

white_noise = generateWhiteNoise(mean = mean_wn,std=var_wn,num_sample=samples)
def padNumDen(num,den):
    if len(den) > len(num):
        num = np.pad(num, (0, len(den) - len(num)), 'constant')
    else:
        den = np.pad(den, (0, len(num) - len(den)), 'constant')
    return num,den
def plotGraph(y,diff):
    plt.plot(y[:500], label = "Raw Data")
    plt.plot(diff[:500], label = "Differenced Data" )
    plt.legend(loc = "upper left")
    plt.grid()
    plt.title("Raw and Differenced data")
    plt.tight_layout()
    plt.show()
#===passing coefficients
ar_coef = np.array(ar_coef)
ma_coef = np.array(ma_coef)
ar_coef,ma_coef = padNumDen(ar_coef,ma_coef)

arma_process = sm.tsa.ArmaProcess(ar_coef,ma_coef)
y= arma_process.generate_sample(samples,scale=var_wn)#+(mean_wn*((np.sum(ma_coef))/(np.sum(ar_coef))))
lag = 50
plt.plot(y[:500])
plt.grid()
plt.title(title)
plt.show()

#ACF pacf of original
ACF_PACF_Plot(y,lag)
#
stationarityCheck(y.ravel(),title=title)

#question 1,2,5,6
# ry, conf_int = sm.tsa.acf(y, nlags=lag, alpha=0.5)
# ry1 = ry[::-1]
# ry2 = np.concatenate((np.reshape(ry1, lag + 1), ry[1:]))
# calcGPAC(ry2, j=10, k=10)

#question 3,4
# order1 = differencing(y,3,1)
# stationarityCheck(order1,title=" Season 3 differencing")
# plotGraph(y,order1)
# order1= removeNone(order1)
# order1 = order1.reset_index(drop=True)
# ACF_PACF_Plot(order1,lag)
# ry, conf_int = sm.tsa.acf(order1, nlags=lag, alpha=0.5)
# ry1 = ry[::-1]
# ry2 = np.concatenate((np.reshape(ry1, lag + 1), ry[1:]))
# calcGPAC(ry2, j=10, k=10)

#question 7,8,9
# order1 = differencing(y,4,1)
# stationarityCheck(order1,title=" Season 4 differencing")
# plotGraph(y,order1)
# order1= removeNone(order1)
# order1 = order1.reset_index(drop=True)
# ACF_PACF_Plot(order1,lag)
# ry, conf_int = sm.tsa.acf(order1, nlags=lag, alpha=0.5)
# ry1 = ry[::-1]
# ry2 = np.concatenate((np.reshape(ry1, lag + 1), ry[1:]))
# calcGPAC(ry2, j=10, k=10)



#
# dacf = autoCorrelationFunction(order1,lag,title="dummy variable",axes=None)
# #stationarityCheck(pd.Series(y.ravel()),title=" Generated data")
# # lag = 20
# #ry= arma_process.acf(lags=lag)
# ry,conf_int = sm.tsa.acf(order1,nlags=lag,alpha=0.5)
# # ### calculating GPAC
# ry1 = ry[::-1]
# ry2 = np.concatenate((np.reshape(ry1,lag+1),ry[1:]))
# calcGPAC(ry2,j=10,k=10)


#======= Question 10 ========
# order2 = differencing(y,1,2)
# stationarityCheck(order2,title="Differencing Second order")
# order2= removeNone(order2)
# order2 = order2.reset_index(drop=True)
# ACF_PACF_Plot(order2,lag)
#
# order3 = differencing(order2,3,1)
# stationarityCheck(order3,title=" Differencing seasonality 3")
# order3= removeNone(order3)
# order3 = order3.reset_index(drop=True)
# ACF_PACF_Plot(order3,lag)
#
#
# ry,conf_int = sm.tsa.acf(order3,nlags=lag,alpha=0.5)
# # ### calculating GPAC
# ry1 = ry[::-1]
# ry2 = np.concatenate((np.reshape(ry1,lag+1),ry[1:]))
# calcGPAC(ry2,j=10,k=10)
#
# plotGraph(y,order3)


#Example 3
model_hat,e,Q = lm_predict(pd.Series(y[:100]),n_a,n_b,ar_coef,ma_coef)
def forecastFunction(y,index=949):
    #y_hat =[0]
    y_hat = [0]
    y_hat.append(1.5*y[index-2] - 0.5*y[index-5])#1st step
    y_hat.append(1.5*y[index-1] - 0.5*y[index-4])#2nd step
    y_hat.append(1.5*y[index]- 0.5*y[index-3])#3rd step
    y_hat.append(1.5*y_hat[1] - 0.5*y[index-2])#4th step
    y_hat.append(1.5*y_hat[2] - 0.5*y[index-1])#5th step
    y_hat.append(1.5*y_hat[3] - 0.5*y[index])#6step
    for h in range(7,51):
        y_hat.append(1.5*y_hat[h-3] - 0.5*y_hat[h-6]) #7 - 51 step predictions
    return pd.Series(y_hat)

test = y[950:]
y_hat = forecastFunction(y)
plt.plot(pd.Series(test),label ="Test Set")
plt.plot(y_hat[1:].reset_index(drop=True),label = "h-step Prediction")
plt.legend(loc = "upper right")
plt.grid()
plt.title("Train versus h step prediction")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()
