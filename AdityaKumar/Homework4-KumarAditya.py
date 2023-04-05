import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from toolbox import cal_autocorr
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")
# Creating a common white noise for all problems
np.random.seed(6313)
N = 100
mean_e = 2
variance_e = 1
e1 = np.random.normal(mean_e, variance_e, N)

# Q1: AR(2)
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)
# Q1 b)
num = [1,0,0]
den = [1,-0.5,-0.2]
system = (num, den, 1)
t, y1 = signal.dlsim(system, e1)

# Q1 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y1):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y1):.4f}\n")

# Q1 d)
np.random.seed(6313)
N1 = 1000
e2 = np.random.normal(mean_e, variance_e, N1)
t, y2 = signal.dlsim(system, e2)
print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y2):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y2):.4f}\n")

np.random.seed(6313)
N2 = 10000
e3 = np.random.normal(mean_e, variance_e, N2)
t, y3 = signal.dlsim(system, e3)
print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.mean(y3):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) is:{np.var(y3):.4f}\n")

# Q1 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","6.6667",round(np.mean(y1),4),"1.7094",round(np.var(y1),4)],["1000","6.6667",round(np.mean(y2),4),"1.7094",round(np.var(y2),4)],["10,000","6.6667",round(np.mean(y3),4),"1.7094",round(np.var(y3),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q1 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1,lag1,"AR(2)")
plt.show()
cal_autocorr(y1,lag2,"AR(2)")
plt.show()
cal_autocorr(y1,lag3,"AR(2)")
plt.show()

np.random.seed(6313)
N = 100
mean_e = 2
variance_e = 1
e1 = np.random.normal(mean_e, variance_e, N)

# Q2: MA(2)
# 𝑦(𝑡)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
# Q2 b)
num = [1,0.1,0.4]
den = [1,0,0]
system = (num, den, 1)
t, y1 = signal.dlsim(system, e1)

# Q2 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y1):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y1):.4f}\n")

# Q2 d)
np.random.seed(6313)
N1 = 1000
e2 = np.random.normal(mean_e, variance_e, N1)
t, y2 = signal.dlsim(system, e2)
print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y2):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y2):.4f}\n")

np.random.seed(6313)
N2 = 10000
e3 = np.random.normal(mean_e, variance_e, N2)
t, y3 = signal.dlsim(system, e3)
print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y3):.4f}")
print(f"The Experimental Variance of y(t) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y3):.4f}\n")

# Q2 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","6.6667",round(np.mean(y1),4),"1.7094",round(np.var(y1),4)],["1000","6.6667",round(np.mean(y2),4),"1.7094",round(np.var(y2),4)],["10,000","6.6667",round(np.mean(y3),4),"1.7094",round(np.var(y3),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q2 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1,lag1,"MA(2)")
plt.show()
cal_autocorr(y1,lag2,"MA(2)")
plt.show()
cal_autocorr(y1,lag3,"MA(2)")
plt.show()

np.random.seed(6313)
N = 100
mean_e = 2
variance_e = 1
e1 = np.random.normal(mean_e, variance_e, N)

# Q3: ARMA(2,2)
# 𝑦(𝑡)−0.5𝑦(𝑡−1)−0.2𝑦(𝑡−2)=𝑒(𝑡)+0.1𝑒(𝑡−1)+0.4𝑒(𝑡−2)
# Q3 b)
num = [1,0.1,0.4]
den = [1,-0.5,-0.2]
system = (num, den, 1)
t, y1 = signal.dlsim(system, e1)

# Q3 c)
print(f"For 100 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y1):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y1):.4f}\n")

# Q3 d)
np.random.seed(6313)
N1 = 1000
e2 = np.random.normal(mean_e, variance_e, N1)
t, y2 = signal.dlsim(system, e2)
print(f"For 1000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y2):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y2):.4f}\n")

np.random.seed(6313)
N2 = 10000
e3 = np.random.normal(mean_e, variance_e, N2)
t, y3 = signal.dlsim(system, e3)
print(f"For 10000 Samples:")
print(f"The Experimental Mean of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.mean(y3):.4f}")
print(f"The Experimental Variance of y(t) - 0.5 y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2) is:{np.var(y3):.4f}\n")

# Q3 e)

table = [["Number of Samples","Theoretical Mean","Experimental Mean","Theoretical Variance","Experimental  Variance"],["100","6.6667",round(np.mean(y1),4),"1.7094",round(np.var(y1),4)],["1000","6.6667",round(np.mean(y2),4),"1.7094",round(np.var(y2),4)],["10,000","6.6667",round(np.mean(y3),4),"1.7094",round(np.var(y3),4)]]

print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q3 f)
lag1 = 20
lag2 = 40
lag3 = 80
cal_autocorr(y1,lag1,"ARMA(2,2)")
plt.show()
cal_autocorr(y1,lag2,"ARMA(2,2)")
plt.show()
cal_autocorr(y1,lag3,"ARMA(2,2)")
plt.show()
