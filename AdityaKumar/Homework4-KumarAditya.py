import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from toolbox import cal_autocorr
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")
# Creating a common white noise with different samples for all problems
np.random.seed(6313)
N = 100
mean_e = 2
variance_e = 1
e1 = np.random.normal(mean_e, variance_e, N)

N1 = 1000
e2 = np.random.normal(mean_e, variance_e, N1)

N2 = 10000
e3 = np.random.normal(mean_e, variance_e, N2)

# Q1: AR(2)
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)
# Q1 b)
num_ar = [1,0,0]
den_ar = [1,-0.5,-0.2]
system_ar = (num_ar, den_ar, 1)
t, y1_ar = signal.dlsim(system_ar, e1)

# Q1 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y1_ar):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y1_ar):.4f}\n")

# Q1 d)
# e2 = 1000 samples
t, y2_ar = signal.dlsim(system_ar, e2)

print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y2_ar):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y2_ar):.4f}\n")

# e3 = 10,000 samples
t, y3_ar = signal.dlsim(system_ar, e3)

print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y3_ar):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y3_ar):.4f}\n")

# Q1 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","6.6667",round(np.mean(y1_ar),4),"1.7094",round(np.var(y1_ar),4)],["1000","6.6667",round(np.mean(y2_ar),4),"1.7094",round(np.var(y2_ar),4)],["10,000","6.6667",round(np.mean(y3_ar),4),"1.7094",round(np.var(y3_ar),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q1 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1_ar,lag1,"AR(2)")
plt.show()
cal_autocorr(y1_ar,lag2,"AR(2)")
plt.show()
cal_autocorr(y1_ar,lag3,"AR(2)")
plt.show()

# Q2: MA(2)
# 𝑦(𝑡)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
# Q2 b)
num_ma = [1,0.1,0.4]
den_ma = [1,0,0]
system_ma = (num_ma, den_ma, 1)
t, y1_ma = signal.dlsim(system_ma, e1)

# Q2 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y1_ma):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y1_ma):.4f}\n")

# Q2 d)
# e2 is 1000 samples
t, y2_ma = signal.dlsim(system_ma, e2)

print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y2_ma):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y2_ma):.4f}\n")

# e3 is 10,000 samples
t, y3_ma = signal.dlsim(system_ma, e3)

print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y3_ma):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y3_ma):.4f}\n")

# Q2 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","3.0000",round(np.mean(y1_ma),4),"1.1700",round(np.var(y1_ma),4)],["1000","3.0000",round(np.mean(y2_ma),4),"1.1700",round(np.var(y2_ma),4)],["10,000","3.0000",round(np.mean(y3_ma),4),"1.1700",round(np.var(y3_ma),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q2 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1_ma,lag1,"MA(2)")
plt.show()
cal_autocorr(y1_ma,lag2,"MA(2)")
plt.show()
cal_autocorr(y1_ma,lag3,"MA(2)")
plt.show()


# Q3: ARMA(2,2)
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
# Q3 b)
num_arma = [1,0.1,0.4]
den_arma = [1,-0.5,-0.2]
system_arma = (num_arma, den_arma, 1)
t, y1_arma = signal.dlsim(system_arma, e1)

# Q3 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y1_arma):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y1_arma):.4f}\n")

# Q3 d)
# e2 is 1000 samples
t, y2_arma = signal.dlsim(system_arma, e2)
print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y2_arma):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y2_arma):.4f}\n")

# e3 is 10,000 samples
t, y3_arma = signal.dlsim(system_arma, e3)
print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y3_arma):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y3_arma):.4f}\n")

# Q3 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","10.0000",round(np.mean(y1_arma),4),"3.0000",round(np.var(y1_arma),4)],["1000","10.0000",round(np.mean(y2_arma),4),"3.0000",round(np.var(y2_arma),4)],["10,000","10.0000",round(np.mean(y3_arma),4),"3.0000",round(np.var(y3_arma),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q3 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1_arma,lag1,"ARMA(2,2)")
plt.show()
cal_autocorr(y1_arma,lag2,"ARMA(2,2)")
plt.show()
cal_autocorr(y1_arma,lag3,"ARMA(2,2)")
plt.show()
