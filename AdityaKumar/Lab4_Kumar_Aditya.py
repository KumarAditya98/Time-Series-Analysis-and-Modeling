import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from toolbox import cal_autocorr, Cal_rolling_mean_var, ADF_Cal, kpss_test
from numpy import linalg as la

# Setting random seed
np.random.seed(6313)

# Q1
# a) Simulating this AR (2) process
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)
e = np.random.normal(0, 1, 1000)
# All initial conditions are zero
y = np.zeros(len(e))
for i in range(len(y)):
    if i == 0:
       y[0] = e[0]

    elif i == 1:
        y[i] = 0.5*y[i-1] + e[i]

    else:
        y[i] = 0.5*y[i-1] + 0.2*y[i-2] + e[i]
# y_format = [round(i,2) for i in y]
df = pd.DataFrame(y,index=range(len(y)),columns=['Value'])

# b) Plotting this time-series synthetic data
df.plot()
plt.grid()
plt.title('Synthetic AR(2) Process')
plt.xlabel('Time Index')
plt.ylabel('Y(t)',rotation = 0)
plt.tight_layout()
plt.show()

# c) Plotting the ACF. Observations will be written in the lab.
lags = 20
cal_autocorr(df.values,lags,'Autocorrelation plot for AR(2) process')
plt.show()

# d) Printing the first 5 values of this process on the console, rounded up to 2 digits.
y_format = [round(i,2) for i in y]
print(f'The first 5 values of AR(2) process using for loop are: {y_format[:5]}')

# e) Checking for stationarity
y_Series = pd.Series(y,name='AR(2)')
Cal_rolling_mean_var(y_Series)
ADF_Cal(y_Series)
kpss_test(y_Series)

# Q2. Creating the same process through DLSIM method
num = [1, 0, 0]
den = [1, -0.5, -0.2]
system = (num, den, 1)
_, y_dlsim = signal.dlsim(system, e)

# a) Plotting the first 5 values of this process
y_dlsim = [round(i,2) for i in list(y_dlsim.reshape(len(y_dlsim)))]
print(f'The first 5 values of AR(2) process using dlsim function: {y_dlsim[:5]}')

# b) To show that the first 5 elements of both processes are same, please see the print statements below
print(f'The first 5 values of AR(2) process using for loop are: {y_format[:5]}')
print(f'The first 5 values of AR(2) process using dlsim function: {y_dlsim[:5]}')

# Observations noted in the report.

# Q3
# Since required to find only a1 and a2, na = 2, T = T' - na - 1
na = 2
T = len(y) - 2 - 1
X1 = np.array([-y[i] for i in range(na-1,na+T)]).reshape(998,1)
X2 = np.array([-y[i] for i in range(na-2,na+T-1)]).reshape(998,1)
X_mat = np.hstack((X1,X2))
Y_mat = np.array([y[i] for i in range(na,na+T+1)]).reshape(998,1)
beta_hat = np.dot(la.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, Y_mat))
print(f"The Estimated Coefficients from the Normal equation are [a1, a2] : {np.round(beta_hat.reshape(2),2)}")

# For a sample size of N= 5000
# np.random.seed(6313)
e_5000 = np.random.normal(0, 1, 5000)
y_5000 = np.zeros(len(e_5000))
T_5000 = len(y_5000) - 2 - 1

for i in range(len(y_5000)):
    if i == 0:
       y_5000[0] = e_5000[0]

    elif i == 1:
        y_5000[i] = 0.5*y_5000[i-1] + e_5000[i]

    else:
        y_5000[i] = 0.5*y_5000[i-1] + 0.2*y_5000[i-2] + e_5000[i]

X1_5000 = np.array([-y_5000[i] for i in range(na-1,na+T_5000)]).reshape(4998,1)
X2_5000 = np.array([-y_5000[i] for i in range(na-2,na+T_5000-1)]).reshape(4998,1)
X_mat_5000 = np.hstack((X1_5000,X2_5000))
Y_mat_5000 = np.array([y_5000[i] for i in range(na,na+T_5000+1)]).reshape(4998,1)
beta_hat_5000 = np.dot(la.inv(np.dot(X_mat_5000.T, X_mat_5000)), np.dot(X_mat_5000.T, Y_mat_5000))
print(f"The Estimated Coefficients from the Normal equation with 5000 samples are [a1, a2] : {np.round(beta_hat_5000.reshape(2),2)}")

# For a sample size of N= 10000
# np.random.seed(6313)
e_10000 = np.random.normal(0, 1, 10000)
y_10000 = np.zeros(len(e_10000))
T_10000 = len(y_10000) - 2 - 1
for i in range(len(y_10000)):
    if i == 0:
        y_10000[0] = e_10000[0]

    elif i == 1:
        y_10000[i] = 0.5 * y_10000[i - 1] + e_10000[i]

    else:
        y_10000[i] = 0.5 * y_10000[i - 1] + 0.2 * y_10000[i - 2] + e_10000[i]

X1_10000 = np.array([-y_10000[i] for i in range(na - 1, na + T_10000)]).reshape(9998, 1)
X2_10000 = np.array([-y_10000[i] for i in range(na - 2, na + T_10000 - 1)]).reshape(9998, 1)
X_mat_10000 = np.hstack((X1_10000, X2_10000))
Y_mat_10000 = np.array([y_10000[i] for i in range(na, na + T_10000 + 1)]).reshape(9998, 1)
beta_hat_10000 = np.dot(la.inv(np.dot(X_mat_10000.T, X_mat_10000)), np.dot(X_mat_10000.T, Y_mat_10000))
print(f"The Estimated Coefficients from the Normal equation with 10,000 samples are [a1, a2] : {np.round(beta_hat_10000.reshape(2), 2)}")

# Q4.
# Creating a custom function to solve this part
def AR_process():
    N = int(input("Enter number of samples:"))
    na = int(input("Enter the order of the AR process:"))
    if na == 0:
        print('This is just a white noise. Run program again.')
        return None
    input_string = input('Enter same number of coefficients as desired order separated by space, in the format [y(t) + a1.y(t-1)) + a2.y(t-2) + ... + = e (t)')
    True_coeff = input_string.split()
    if len(True_coeff) != na:
        print("Incorrect number of coefficients entered, run function again")
        return None
    for i in range(len(True_coeff)):
        # convert each item to int type
        True_coeff[i] = float(True_coeff[i])
    np.random.seed(6313)
    e = np.random.normal(0,1,N)
    num = [0 for i in range(na)]
    num.insert(0,1)
    den = True_coeff.copy()
    den.insert(0,1)
    system = (num, den, 1)
    _, y = signal.dlsim(system, e)
    T = len(y) - na - 1
    y = [round(i, 2) for i in list(y.reshape(len(y)))]
    X_mat = np.empty((T+1,0))
    for i in reversed(range(0,na)):
        X = np.array([-y[j] for j in range(i, i + T+1)]).reshape(T+1, 1)
        X_mat = np.hstack((X_mat,X))
    Y_mat = np.array([y[i] for i in range(na, na + T+1)]).reshape(T+1, 1)
    beta_hat = np.dot(la.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, Y_mat))
    print(f"The Estimated Coefficients from the Normal equation in the order [a1, a2, ...] are: {np.round(beta_hat.reshape(na), 2)}")
    print(f"Whereas the True Coefficients that were supplied to the process are, same order: {True_coeff}")

AR_process() # for N = 1000
AR_process() # for N = 10000
AR_process() # for N = 100000


# Q5.
# a) Simulating this MA (2) process
# 𝑦(𝑡)=𝑒(𝑡)+0.5𝑒(𝑡−1)+0.2𝑒(𝑡−2)
np.random.seed(6313)
e = np.random.normal(0, 1, 1000)
# All initial conditions are zero
y = np.zeros(len(e))
for i in range(len(y)):
    if i == 0:
       y[0] = e[0]

    elif i == 1:
        y[i] = 0.5*e[i-1] + e[i]

    else:
        y[i] = 0.5*e[i-1] + 0.2*e[i-2] + e[i]
# y_format = [round(i,2) for i in y]
df1 = pd.DataFrame(y,index=range(len(y)),columns=['Value'])

# b) Plotting this time-series synthetic data
df1.plot()
plt.grid()
plt.title('Synthetic MA(2) Process')
plt.xlabel('Time Index')
plt.ylabel('Y(t)',rotation = 0)
plt.tight_layout()
plt.show()

# c) Plotting the ACF. Observations will be written in the lab.
lags = 20
cal_autocorr(df1.values,lags,'Autocorrelation - MA(2) process - N = 1000')
plt.show()

# d) Generating the MA(2) process for N = 10000 and N = 100000
# For a sample size of N= 10000
# np.random.seed(6313)
e_10000 = np.random.normal(0, 1, 10000)
y_10000 = np.zeros(len(e_10000))
T_10000 = len(y_10000) - 2 - 1
for i in range(len(y_10000)):
    if i == 0:
        y_10000[0] = e_10000[0]

    elif i == 1:
        y_10000[i] = 0.5 * e_10000[i - 1] + e_10000[i]

    else:
        y_10000[i] = 0.5 * e_10000[i - 1] + 0.2 * e_10000[i - 2] + e_10000[i]

df2 = pd.DataFrame(y_10000,index=range(len(y_10000)),columns=['Value'])
cal_autocorr(df2.values,lags,'Autocorrelation - MA(2) process - N = 10000')
plt.show()

e_100000 = np.random.normal(0, 1, 100000)
y_100000 = np.zeros(len(e_100000))
T_100000 = len(y_100000) - 2 - 1
for i in range(len(y_100000)):
    if i == 0:
        y_100000[0] = e_100000[0]

    elif i == 1:
        y_100000[i] = 0.5 * e_100000[i - 1] + e_100000[i]

    else:
        y_100000[i] = 0.5 * e_100000[i - 1] + 0.2 * e_100000[i - 2] + e_100000[i]

df3 = pd.DataFrame(y_100000,index=range(len(y_100000)),columns=['Value'])
cal_autocorr(df3.values,lags,'Autocorrelation - MA(2) process - N = 100000')
plt.show()

# d) Printing the first 5 values of this process on the console, rounded up to 2 digits.
y_format = [round(i,2) for i in y]
print(f'The first 5 values of MA(2) process using for loop are: {y_format[:5]}')

# e) Checking for stationarity
y_Series = pd.Series(y,name='MA(2)')
Cal_rolling_mean_var(y_Series)
ADF_Cal(y_Series)
kpss_test(y_Series)

# Q2. Creating the same process through DLSIM method
num = [1, 0.5, 0.2]
den = [1, 0, 0]
system = (num, den, 1)
_, y_dlsim = signal.dlsim(system, e)

# a) Plotting the first 5 values of this process
y_dlsim = [round(i,2) for i in list(y_dlsim.reshape(len(y_dlsim)))]
print(f'The first 5 values of MA(2) process using dlsim function: {y_dlsim[:5]}')

# b) To show that the first 5 elements of both processes are same, please see the print statements below
y_format = [round(i,2) for i in y]
print(f'The first 5 values of MA(2) process using for loop are: {y_format[:5]}')
print(f'The first 5 values of MA(2) process using dlsim function: {y_dlsim[:5]}')

# Observations noted in the report.




