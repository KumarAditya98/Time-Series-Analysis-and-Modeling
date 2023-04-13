import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from toolbox import ARMA_process, Cal_GPAC, ACF_PACF_Plot

# Q1. Creating a function for process generation and GPAC
# Created in the toolbox python supporting file. Pasted below for reference:
# def ARMA_process():
#     np.random.seed(6313)
#     N = int(input("Enter the number of data samples:"))
#     mean_e = int(input("Enter the mean of white noise:"))
#     var_e = int(input("Enter the variance of white noise:"))
#     ar_order = int(input("Enter the AR portion order:"))
#     ma_order = int(input("Enter the MA portion order:"))
#     if ar_order == 0 and ma_order == 0:
#         print("This is just a white noise. Run program again.")
#         return None
#     print('Enter the respective coefficients in the form:\n[y(t) + a1*y(t-1)) + a2*y(t-2) + ... = e (t) + b1*e(t-1) + b2*e(t-2) + ...]\n')
#     ar_coeff = []
#     for i in range(ar_order):
#         prompt = "Enter the coefficient for a" + str((i+1))
#         ar_coeff.append(float(input(prompt)))
#     ma_coeff = []
#     for i in range(ma_order):
#         prompt = "Enter the coefficient for b" + str((i+1))
#         ma_coeff.append(float(input(prompt)))
#     ar = np.r_[1,ar_coeff]
#     ma = np.r_[1,ma_coeff]
#     arma_process = sm.tsa.ArmaProcess(ar, ma)
#     mean_y = mean_e*(1+np.sum(ma_coeff))/(1+np.sum(ar_coeff))
#     y = arma_process.generate_sample(N, scale=np.sqrt(var_e)) + mean_y
#     lags = 60
#     if arma_process.isstationary:
#         print('Process with given coefficients is Stationary.')
#         ry = arma_process.acf(lags=lags)
#     else:
#         print('Process with given coefficients is Non-Stationary.')
#         ry = sm.tsa.stattools.acf(y, nlags=lags)
#     ryy = ry[::-1]
#     Ry = np.concatenate((ryy, ry[1:]))
#     return y, Ry
#
# def Cal_GPAC(ry,j=7,k=7):
#     matrix = np.empty((j,k))
#     mid_point = int(len(ry)/2)
#     for i in range(k):
#         col = []
#         for l in range(j):
#             if i == 0:
#                 col.append(round(float(ry[mid_point+l+1]/ry[mid_point+l]),3))
#             else:
#                 den = np.empty((i+1,i+1))
#                 for b in range(i+1):
#                     temp = []
#                     for a in range(mid_point + l - i + b, mid_point+b+l+1):
#                         temp.append(ry[a])
#                     temp1 = np.array(temp)
#                     den[:,b] = temp1
#                 Denn = den[:,::-1]
#                 den_det = np.linalg.det(Denn)
#                 num = Denn.copy()
#                 temp = []
#                 for c in range(i+1):
#                     temp.append(ry[mid_point+l+1+c])
#                 temp1 = np.array(temp)
#                 num[:,-1] = temp1
#                 num_det = np.linalg.det(num)
#                 col.append(np.divide(num_det,den_det))
#         col = np.array(col)
#         matrix[:,i] = col
#     fig, ax = plt.subplots(figsize = (12,8))
#     sns.heatmap(matrix,annot=True,cmap='coolwarm',ax=ax,fmt='.3f',xticklabels=list(range(1,k+1)),yticklabels=list(range(j)),annot_kws={"size": 30 / np.sqrt(len(matrix)),"fontweight":'bold'},robust=True)
#     ax.tick_params(labelsize=30 / np.sqrt(len(matrix)))
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=30 / np.sqrt(len(matrix)),width=2)
#     fig.subplots_adjust(top=0.88)
#     fig.suptitle('Generalized Partial Autocorrelation (GPAC) Table',fontweight='bold',size=24)
#     plt.xticks(weight='bold')
#     plt.yticks(weight='bold')
#     plt.show()
#     print(matrix)


# Q2. and Q3. 𝑦(𝑡)−0.5𝑦(𝑡−1)=𝑒(𝑡) where e(t) ~ WN(1,2)
y1, ry1 = ARMA_process()

# Q4.
Cal_GPAC(ry1)

# Q5.
ACF_PACF_Plot(y1,20)

# Q6.
# 5000 Samples
# Example 2: ARMA (0,1): y(t) = e(t) + 0.5e(t 1)
y2, ry2 = ARMA_process()

# Q4.
Cal_GPAC(ry2)

# Q5.
ACF_PACF_Plot(y2,20)
#

