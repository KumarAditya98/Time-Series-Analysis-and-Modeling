import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from toolbox import Cal_rolling_mean_var, cal_autocorr

import warnings
warnings.filterwarnings("ignore")
# Creating a common white noise for all problems
np.random.seed(6313)
N = 100
mean_e = 2
variance_e = 1
e1 = np.random.normal(mean_e, variance_e, N)

# Q1.
# b)

