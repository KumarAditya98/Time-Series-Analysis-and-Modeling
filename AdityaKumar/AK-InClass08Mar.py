# For loop has been used to create the AR random process.

import pandas as pd
import numpy as np

np.random.seed(6313)
# =========================================
# y(t) + 0.5y(t-1) + 0.25y(t-2) = e(t)
# ===========================================
e = np.random.normal(1,1,1000)

# Method 1:
y = np.zeros(len(e))
for i in range(len(e)):
    if i == 0:
        y[0] = e[0]
    elif i == 1:
        y[i] = -0.5*y[i-1] + e[i]
    else:
        y[i] = -0.5 * y[i - 1] - 0.25*y[i-2] + e[i]

print(f'For loop method generation: {y[:3]}')

# Dlsim method:
from scipy import signal
num = [1,0,0]
den = [1,0.5,0.25] # Coefficients of the polynomial form of the AR, when the negative power are removed by multiplying by the highest power
system = (num,den,1)
t,y_dlsim = signal.dlsim(system,e)

print(f"dlsim method: {y_dlsim[:3]}")
print(f"The experimental mean of y is: {np.mean(y_dlsim)}")

