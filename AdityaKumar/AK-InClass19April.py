import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Look at in class file.

ax = plt.subplot(111)
durations = [5,6,6,2.5,4,4]
event = [1,0,0,1,1,1]
kmf = KaplanMeierFitter()
kmf.fit(durations,event, label = "Number of users that stay on a website")
kmf.plot_survival_function(ax = ax)
plt.show()

