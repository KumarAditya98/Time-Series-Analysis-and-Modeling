import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Lab1_Kumar_Aditya import Cal_rolling_mean_var

np.random.seed(123)
x = np.random.normal(0,1,1000)
date = pd.date_range(start = '1/1/2000',
                     end = '12/31/2000',
                     periods = len(x))

t = np.linspace(-np.pi,np.pi,len(x))
y = 5*np.sin(t) + x

df = pd.DataFrame(y, columns = ['temp'])

df.index = date

df.plot()
plt.grid()
plt.title('dummy data')
plt.show()

Cal_rolling_mean_var(df.temp)