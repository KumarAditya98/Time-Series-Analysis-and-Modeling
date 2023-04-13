import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf

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

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %.2f" %result[0])
    print('p-value: %.2f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.2f' % (key, value))

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series([round(i,2) for i in kpsstest[0:3]], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = round(value,2)
    print (kpss_output)

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

def diff(Df,col,order):
    name = col + '_' + str(order) + '_Diff'
    Df[name] = 0
    temp1 = Df[col][::-1]
    temp2 = temp1[0:-order] - temp1.values[order:]
    Df[name] = temp2[::-1]
    return Df

def cal_autocorr(array,lag,title,ax=None):
    if ax == None:
        ax = plt.gca()
    mean = np.mean(array)
    denominator = 0
    for x in range(len(array)):
        denominator = (array[x] - mean)**2 + denominator
    ry = []
    for tau in range(lag):
        numerator = 0
        for t in range(tau,len(array)):
            numerator = (array[t]-mean)*(array[t-tau]-mean) + numerator
        val = numerator/denominator
        ry.append(val)
    ryy = ry[::-1]
    Ry = ryy[:-1] + ry[:]

    bins = np.linspace(-(len(ryy) - 1), len(ryy) - 1, len(Ry))
    ax.stem(bins,Ry,markerfmt='ro',basefmt='C5')
    ax.axhspan(1.96/len(array)**0.5,-1.96/len(array)**0.5,alpha = 0.2, color = 'blue')
    ax.locator_params(axis='x', tight = True, nbins=9)
    ax.set_title('Autocorrelation Function of ' + title)
    ax.set_xlabel('Lags')
    ax.set_ylabel('Magnitude')

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
def ARMA_process():
    np.random.seed(6313)
    N = int(input("Enter the number of data samples:"))
    mean_e = int(input("Enter the mean of white noise:"))
    var_e = int(input("Enter the variance of white noise:"))
    ar_order = int(input("Enter the AR portion order:"))
    ma_order = int(input("Enter the MA portion order:"))
    if ar_order == 0 and ma_order == 0:
        print("This is just a white noise. Run program again.")
        return None
    print('Enter the respective coefficients in the form:\n[y(t) + a1*y(t-1)) + a2*y(t-2) + ... = e (t) + b1*e(t-1) + b2*e(t-2) + ...]\n')
    ar_coeff = []
    for i in range(ar_order):
        prompt = "Enter the coefficient for a" + str((i+1))
        ar_coeff.append(float(input(prompt)))
    ma_coeff = []
    for i in range(ma_order):
        prompt = "Enter the coefficient for b" + str((i+1))
        ma_coeff.append(float(input(prompt)))
    ar = np.r_[1,ar_coeff]
    ma = np.r_[1,ma_coeff]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean_e*(1+np.sum(ma_coeff))/(1+np.sum(ar_coeff))
    y = arma_process.generate_sample(N, scale=np.sqrt(var_e)) + mean_y
    lags = 60
    if arma_process.isstationary:
        print('Process with given coefficients is Stationary.')
        ry = arma_process.acf(lags=lags)
    else:
        print('Process with given coefficients is Non-Stationary.')
        ry = sm.tsa.stattools.acf(y, nlags=lags)
    ryy = ry[::-1]
    Ry = np.concatenate((ryy, ry[1:]))
    return y, Ry

def Cal_GPAC(ry,j=7,k=7):
    matrix = np.empty((j,k))
    mid_point = int(len(ry)/2)
    for i in range(k):
        col = []
        for l in range(j):
            if i == 0:
                col.append(round(float(ry[mid_point+l+1]/ry[mid_point+l]),3))
            else:
                den = np.empty((i+1,i+1))
                for b in range(i+1):
                    temp = []
                    for a in range(mid_point + l - i + b, mid_point+b+l+1):
                        temp.append(ry[a])
                    temp1 = np.array(temp)
                    den[:,b] = temp1
                Denn = den[:,::-1]
                den_det = np.linalg.det(Denn)
                num = Denn.copy()
                temp = []
                for c in range(i+1):
                    temp.append(ry[mid_point+l+1+c])
                temp1 = np.array(temp)
                num[:,-1] = temp1
                num_det = np.linalg.det(num)
                if den_det < 0.0001:
                    col.append(np.inf)
                else:
                    col.append(np.divide(num_det,den_det))
        col = np.array(col)
        matrix[:,i] = col
    np.set_printoptions(precision=3,floatmode='fixed')
    fig, ax = plt.subplots(figsize = (12,8))
    sns.heatmap(matrix,annot=True,cmap='coolwarm',ax=ax,fmt='.3f',xticklabels=list(range(1,k+1)),yticklabels=list(range(j)),annot_kws={"size": 30 / np.sqrt(len(matrix)),"fontweight":'bold'},vmin=np.min(matrix),vmax=np.max(matrix))
    ax.tick_params(labelsize=30 / np.sqrt(len(matrix)))
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30 / np.sqrt(len(matrix)),width=2)
    fig.subplots_adjust(top=0.88)
    fig.suptitle('Generalized Partial Autocorrelation (GPAC) Table',fontweight='bold',size=24)
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.show()
    print(matrix)

def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(16,8))
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()



