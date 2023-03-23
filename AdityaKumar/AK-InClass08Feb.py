import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodel
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/AirPassengers.csv'

df = pd.read_csv(url,index_col='Month',parse_dates=True)
print(df.head())
y=df['#Passengers']

yt,yf = train_test_split(y,shuffle = False, test_size = 0.2)

# SES Method
holtt = ets.ExponentialSmoothing(yt,trend=None,damped=False,seasonal=None).fit()
holtf = holtt.forecast(steps=len(yf))
holtf = pd.DataFrame(holtf).set_index(yf.index)

MSE = np.square(np.subtract(yf.values,np.ndarray.flatten(holtf.values))).mean()
print(MSE)

fig,ax = plt.subplots()
