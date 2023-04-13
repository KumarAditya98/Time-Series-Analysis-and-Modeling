import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from toolbox import ARMA_process, Cal_GPAC, ACF_PACF_Plot

# Q1. Creating a function for process generation and GPAC

y, ry = ARMA_process()
Cal_GPAC(ry,7,7)

ACF_PACF_Plot(y,60)