import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from toolbox import ARMA_process, Cal_GPAC, ACF_PACF_Plot

# Q1. Creating a function for process generation and GPAC

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
#     print('Enter the respective coefficients in the form:\n[y(t) + a1.y(t-1)) + a2.y(t-2) + ... = e (t) + b1*e(t-1) + b2*e(t-2) + ...]\n')
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
#         print('Process with given coefficeints is Stationary.')
#         ry = arma_process.acf(lags=lags)
#     else:
#         print('Process with given coefficeints is Non-Stationary.')
#         ry = sm.tsa.stattools.acf(y, nlags=lags)
#     ryy = ry[::-1]
#     Ry = np.concatenate((ryy, ry[1:]))
#     return y, Ry

y, ry = ARMA_process()

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
#                 num = np.empty((i+1,i+1))
#                 for b in range(1,i+1):
#                     temp = []
#                     for a in range(mid_point - i +b +l,mid_point+b+l+1):
#                         temp.append(ry[a])
#                     temp1 = np.array(temp)
#                     num[:,b] = temp1
#                 temp = []
#                 for c in range(i+1):
#                     temp.append(ry[mid_point+l+1+c])
#                 temp1 = np.array(temp)
#                 num[:,0] = temp1
#                 Numm = num[:, ::-1]
#                 num_det = np.linalg.det(Numm)
#                 col.append(round(float(num_det/den_det),3))
#         col = np.array(col)
#         matrix[:,i] = col
#     np.set_printoptions(precision=3,floatmode='fixed')
#     fig, ax = plt.subplots(figsize = (12,8))
#     sns.heatmap(matrix,annot=True,cmap='coolwarm',ax=ax,fmt='.3f',xticklabels=list(range(1,k+1)),yticklabels=list(range(j)),annot_kws={"size": 30 / np.sqrt(len(matrix)),"fontweight":'bold'})
#     ax.tick_params(labelsize=30 / np.sqrt(len(matrix)))
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=30 / np.sqrt(len(matrix)),width=2)
#     fig.subplots_adjust(top=0.88)
#     fig.suptitle('Generalized Partial Autocorrelation (GPAC) Table',fontweight='bold',size=24)
#     plt.xticks(weight='bold')
#     plt.yticks(weight='bold')
#     plt.show()
#     print(matrix)

# ry1 = [13/12,-7/24,7/(24*2),-7/(24*4),7/(24*8),-7/(24*16)]
# ryy = ry1[::-1]
# ry = np.concatenate((ryy,ry1[1:]))
Cal_GPAC(ry,7,7)

# from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
# def ACF_PACF_Plot(y,lags):
#     acf = sm.tsa.stattools.acf(y, nlags=lags)
#     pacf = sm.tsa.stattools.pacf(y, nlags=lags)
#     fig = plt.figure(figsize=(16,8))
#     plt.subplot(211)
#     plt.title('ACF/PACF of the raw data')
#     plot_acf(y, ax=plt.gca(), lags=lags)
#     plt.subplot(212)
#     plot_pacf(y, ax=plt.gca(), lags=lags)
#     fig.tight_layout(pad=3)
#     plt.show()

ACF_PACF_Plot(samples,60)