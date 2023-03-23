import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from toolbox import cal_autocorr

# Creating the datasets first
train_list = pd.Series([112,118,132,129,121,135,148,136,119],name='value')
train_index = np.arange(1,10)

test_list = pd.Series([104,118,115,126,141],name='value')
test_index = np.arange(10,15)

df_train = pd.DataFrame(train_list).set_index(train_index)
df_test = pd.DataFrame(test_list).set_index(test_index)

#============
# Average Forecast Method
#============

# Q2. Calculating the 1-step prediction for training set and h-step forecast for test set using the Average Forecast Method

# Defining a function for train set average forecast method prediction
def avg_forecast_train(x):
    new = []
    for i in range(1,len(x)):
        new.append(np.mean(x[0:i]))
    return new
# Applying this running average to get predicted values
y_pred = avg_forecast_train(df_train.value.values)
# Inserting blank at first index to match train set size
y_pred.insert(0,np.nan)
# Creating columns in the train dataset, similar to Q1
df_train['Avg_forecast_method'] = y_pred
df_train['Avg_forecast_error'] = df_train['value'] - df_train['Avg_forecast_method']
df_train['Avg_forecast_error_squ'] = df_train['Avg_forecast_error']**2
# Printing the train set table as done in Q1
print(df_train)

# Now performing Average forecast method for test set. No need to define a function since its a single static value
y_pred = np.mean(df_train.value.values)
# Now entering this forecasted value in the test data set, along with error and error^2
df_test['Avg_forecast_method'] = y_pred
df_test['Avg_forecast_error'] = df_test['value'] - df_test['Avg_forecast_method']
df_test['Avg_forecast_error_squ'] = df_test['Avg_forecast_error']**2

# Printing the test set table as done in Q1
print(df_test)

# Plotting the test data, train data, and h step forecasted data onto a single graph
fig, ax = plt.subplots()
ax.plot(df_train.value,label="Train Data")
ax.plot(df_test.value,label="Test Data")
ax.plot(df_test.Avg_forecast_method,label="Average Forecast Method",ls="--")
plt.legend(loc='upper left')
plt.grid()
plt.title('Average Forecast Method')
plt.xlabel('Time (t)')
plt.ylabel('Sample Value, y(t)')
plt.show()

# Q3. Calculating the MSE of the train and test errors that were already calculated previously

mse_prediction = []
mse_forecast = []
mse_prediction.append(np.mean(df_train.Avg_forecast_error_squ.values[2:])) # Since asked to remove first 2 observations
mse_forecast.append(np.mean(df_test.Avg_forecast_error_squ.values))

# Displaying the MSE in a table format
table = [['MSE for Prediction', 'MSE for Forecast'],
         [mse_prediction[0], mse_forecast[0]]]
print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q4. Calculating the variance for prediction and forecast error
var_prediction = []
var_forecast = []
var_prediction.append(np.var(df_train.Avg_forecast_error.values[2:]))
var_forecast.append(np.var(df_test.Avg_forecast_error.values))
print(f'The Variance for Prediction Error is: {round(var_prediction[0],2)}\nThe Variance for Forecast Error is: {round(var_forecast[0],2)}')

# Q5. First modifying the autocorrelation function to retrieve the list of autocorrelation values instead
def autocorrelation(array,lag):
    mean = np.mean(array)
    denominator = 0
    for x in range(len(array)):
        denominator = (array[x] - mean)**2 + denominator
    ry = []
    for tau in range(lag+1):
        numerator = 0
        for t in range(tau,len(array)):
            numerator = (array[t]-mean)*(array[t-tau]-mean) + numerator
        val = numerator/denominator
        ry.append(val)
    return ry
# Next creating a function to calculate the Q value for a given error array, T value and h value
def q_val(array,T,h):
    val = 0
    for i in range(1,h+1):
        val = array[i]**2 + val
    q = T*val
    return q
# Calculating the autocorrelation values for train set and storing in a list for lag = 5
lag = 5
autocorr = autocorrelation(df_train.Avg_forecast_error.values[2:],lag)
# Now calculating the Q value using the custom function in a list for Q_values of all methods
Q_value = []
T = 7 # Since we're skipping first 2 observations from train set for all methods
h = 5
Q_value.append(q_val(autocorr,T,h))
# Displaying output on the console
print(f'The Q Value for the train set for Average Forecast Method is: {Q_value[0]:.3f}')

#=============
# Naive method
#=============

# Creating a copy of df_train and df_test to work on
df_train1 = df_train.iloc[:,0:1].copy()
df_test1 = df_test.iloc[:,0:1].copy()

# Q2. Calculating the 1-step prediction for training set and h-step forecast for test set using the Naive Method

# Defining a function for train set naive method prediction
def naive_forecast_train(x):
    new = []
    for i in range(len(x)-1):
        new.append(x[i])
    return new
# Applying this method to get predicted values
y_pred = naive_forecast_train(df_train1.value.values)
# Inserting blank at first index to match train set size
y_pred.insert(0,np.nan)
# Creating columns in the train dataset, similar to Q1
df_train1['Naive_forecast_method'] = y_pred
df_train1['Naive_forecast_error'] = df_train1['value'] - df_train1['Naive_forecast_method']
df_train1['Naive_forecast_error_squ'] = df_train1['Naive_forecast_error']**2
# Printing the train set table as done in Q1
print(df_train1)

# Now performing Naive method for test set. No need to define a function since its a single static value
y_pred = df_train1.value.values[-1]
# Now entering this forecasted value in the test data set, along with error and error^2
df_test1['Naive_forecast_method'] = y_pred
df_test1['Naive_forecast_error'] = df_test1['value'] - df_test1['Naive_forecast_method']
df_test1['Naive_forecast_error_squ'] = df_test1['Naive_forecast_error']**2

# Printing the test set table as done in Q1
print(df_test1.to_string())

# Plotting the test data, train data, and h step forecasted data onto a single graph
fig, ax = plt.subplots()
ax.plot(df_train1.value,label="Train Data")
ax.plot(df_test1.value,label="Test Data")
ax.plot(df_test1.Naive_forecast_method,label="Naive Forecast Method",ls="--")
plt.legend(loc='upper left')
plt.grid()
plt.title('Naive Forecast Method')
plt.xlabel('Time (t)')
plt.ylabel('Sample Value, y(t)')
plt.show()

# Q3. Calculating the MSE of the train and test errors that were already calculated previously

mse_prediction.append(np.mean(df_train1.Naive_forecast_error_squ.values[2:]))
mse_forecast.append(np.mean(df_test1.Naive_forecast_error_squ.values))

# Displaying the MSE in a table format
table = [['MSE for Prediction', 'MSE for Forecast'],
         [mse_prediction[1], mse_forecast[1]]]
print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q4. Calculating the variance for prediction and forecast error

var_prediction.append(np.var(df_train1.Naive_forecast_error.values[2:]))
var_forecast.append(np.var(df_test1.Naive_forecast_error.values))
print(f'The Variance for Prediction Error is: {round(var_prediction[1],2)}\nThe Variance for Forecast Error is: {round(var_forecast[1],2)}')

# Q5. Calculating the autocorrelation values for train set and storing in a list for lag = 5
lag = 5
autocorr = autocorrelation(df_train1.Naive_forecast_error.values[2:],lag)
# Now calculating the Q value using the custom function in a list for Q_values of all methods
T = 7 # Since we're skipping first 2 observations from train set for all methods
h = 5
Q_value.append(q_val(autocorr,T,h))
# Displaying output on the console
print(f'The Q Value for the train set for Naive Forecast Method is: {Q_value[1]:.3f}')

#===========
# Drift Method
#===========

# Creating a copy of df_train and df_test to work on
df_train2 = df_train.iloc[:,0:1].copy()
df_test2 = df_test.iloc[:,0:1].copy()

# Q2. Calculating the 1-step prediction for training set and h-step forecast for test set using the Drift Method

# Defining a function for train set Drift method prediction
def drift_forecast_train(x):
    new = []
    for i in range(1,len(x)-1):
        new.append((x[i]+((x[i]-x[0])/(i))))
    return new
# Applying this method to get predicted values
y_pred = drift_forecast_train(df_train2.value.values)
# Inserting 2 blanks at first, second index to match train set size
y_pred.insert(0,np.nan)
y_pred.insert(0,np.nan)
# Creating columns in the train dataset, similar to Q1
df_train2['Drift_forecast_method'] = y_pred
df_train2['Drift_forecast_error'] = df_train2['value'] - df_train2['Drift_forecast_method']
df_train2['Drift_forecast_error_squ'] = df_train2['Drift_forecast_error']**2
# Printing the train set table as done in Q1
print(df_train2)

# Now defining a function for Drift method for test set.
def drift_forecast_test(x_train,h):
    new = []
    for i in range(1,h+1):
        new.append((x_train[-1]+i*((x_train[-1]-x_train[0])/(len(x_train)-1))))
    return new
y_pred = drift_forecast_test(df_train2.value.values,5)
# Now entering this forecasted value in the test data set, along with error and error^2
df_test2['Drift_forecast_method'] = y_pred
df_test2['Drift_forecast_error'] = df_test2['value'] - df_test2['Drift_forecast_method']
df_test2['Drift_forecast_error_squ'] = df_test2['Drift_forecast_error']**2

# Printing the test set table as done in Q1
print(df_test2.to_string())

# Plotting the test data, train data, and h step forecasted data onto a single graph
fig, ax = plt.subplots()
ax.plot(df_train2.value,label="Train Data")
ax.plot(df_test2.value,label="Test Data")
ax.plot(df_test2.Drift_forecast_method,label="Drift Forecast Method",ls="--")
plt.legend(loc='upper left')
plt.grid()
plt.title('Drift Forecast Method')
plt.xlabel('Time (t)')
plt.ylabel('Sample Value, y(t)')
plt.show()

# Q3. Calculating the MSE of the train and test errors that were already calculated previously

mse_prediction.append(np.mean(df_train2.Drift_forecast_error_squ.values[2:]))
mse_forecast.append(np.mean(df_test2.Drift_forecast_error_squ.values))

# Displaying the MSE in a table format
table = [['MSE for Prediction', 'MSE for Forecast'],
         [mse_prediction[2], mse_forecast[2]]]
print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q4. Calculating the variance for prediction and forecast error

var_prediction.append(np.var(df_train2.Drift_forecast_error.values[2:]))
var_forecast.append(np.var(df_test2.Drift_forecast_error.values))
print(f'The Variance for Prediction Error is: {round(var_prediction[2],2)}\nThe Variance for Forecast Error is: {round(var_forecast[2],2)}')

# Q5. Calculating the autocorrelation values for train set and storing in a list for lag = 5
lag = 5
autocorr = autocorrelation(df_train2.Drift_forecast_error.values[2:],lag)
# Now calculating the Q value using the custom function in a list for Q_values of all methods
T = 7 # Since we're skipping first 2 observations from train set for all methods
h = 5
Q_value.append(q_val(autocorr,T,h))
# Displaying output on the console
print(f'The Q Value for the train set for Drift Forecast Method is: {Q_value[2]:.3f}')

#===========
# Simple Exponential Smoothing Method
#===========

# Creating a copy of df_train and df_test to work on
df_train3 = df_train.iloc[:,0:1].copy()
df_test3 = df_test.iloc[:,0:1].copy()

# Q2. Calculating the 1-step prediction for training set and h-step forecast for test set using the SES Method

# Defining a function for train set SES method prediction
def SES_forecast_train(x,alpha,ic):
    new = [ic]
    for i in range(len(x)-1):
        new.append((alpha*x[i]+(1-alpha)*new[i]))
    return new
# Applying this method to get predicted values
alpha = 0.5
ic = 112.0
y_pred = SES_forecast_train(df_train3.value.values,alpha,ic)

# Creating columns in the train dataset, similar to Q1
df_train3['SES_forecast_method'] = y_pred
df_train3['SES_forecast_error'] = df_train3['value'] - df_train3['SES_forecast_method']
df_train3['SES_forecast_error_squ'] = df_train3['SES_forecast_error']**2
# Printing the train set table as done in Q1
print(df_train3)

# Now defining a function for SES method for test set for different alpha
def SES_forecast_test(x_train,x_predicted,alpha):
    new = alpha*x_train[-1] + (1-alpha)*x_predicted[-1]
    return new
# Retrieving the SES Forecast method for alpha = 0.5
alpha = 0.5
y_pred = SES_forecast_test(df_train3.value.values,df_train3.SES_forecast_method.values,alpha)
# Now entering this forecasted value in the test data set, along with error and error^2
df_test3['SES_forecast_method'] = y_pred
df_test3['SES_forecast_error'] = df_test3['value'] - df_test3['SES_forecast_method']
df_test3['SES_forecast_error_squ'] = df_test3['SES_forecast_error']**2

# Printing the test set table as done in Q1
print(df_test3.to_string())

# Plotting the test data, train data, and h step forecasted data onto a single graph
fig, ax = plt.subplots()
ax.plot(df_train3.value,label="Train Data")
ax.plot(df_test3.value,label="Test Data")
ax.plot(df_test3.SES_forecast_method,label="SES Forecast Method",ls="--")
plt.legend(loc='upper left')
plt.grid()
plt.title('Simple Exponential Smoothing Forecast Method')
plt.xlabel('Time (t)')
plt.ylabel('Sample Value, y(t)')
plt.show()

# Q3. Calculating the MSE of the train and test errors that were already calculated previously

mse_prediction.append(np.mean(df_train3.SES_forecast_error_squ.values[2:]))
mse_forecast.append(np.mean(df_test3.SES_forecast_error_squ.values))

# Displaying the MSE in a table format
table = [['MSE for Prediction', 'MSE for Forecast'],
         [mse_prediction[3], mse_forecast[3]]]
print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q4. Calculating the variance for prediction and forecast error

var_prediction.append(np.var(df_train3.SES_forecast_error.values[2:]))
var_forecast.append(np.var(df_test3.SES_forecast_error.values))
print(f'The Variance for Prediction Error is: {round(var_prediction[3],2)}\nThe Variance for Forecast Error is: {round(var_forecast[3],2)}')

# Q5. Calculating the autocorrelation values for train set and storing in a list for lag = 5
lag = 5
autocorr = autocorrelation(df_train3.SES_forecast_error.values[2:],lag)
# Now calculating the Q value using the custom function in a list for Q_values of all methods
T = 7 # Since we're skipping first 2 observations from train set for all methods
h = 5
Q_value.append(q_val(autocorr,T,h))
# Displaying output on the console
print(f'The Q Value for the train set for Drift Forecast Method is: {Q_value[3]:.3f}')

# Q9. First calculating the forecasted value for SES Method for different alphas
alpha1 = 0
alpha2 = 0.25
alpha3 = 0.75
alpha4 = 0.99
y_pred1 = SES_forecast_test(df_train3.value.values,df_train3.SES_forecast_method.values,alpha1)
y_pred2 = SES_forecast_test(df_train3.value.values,df_train3.SES_forecast_method.values,alpha2)
y_pred3 = SES_forecast_test(df_train3.value.values,df_train3.SES_forecast_method.values,alpha3)
y_pred4 = SES_forecast_test(df_train3.value.values,df_train3.SES_forecast_method.values,alpha4)

# Saving this in new dataframe for test set
df_test4 = df_test.iloc[:,0:1].copy()
df_train4 = df_train.iloc[:,0:1].copy()
df_test4['y_pred1'] = y_pred1
df_test4['y_pred2'] = y_pred2
df_test4['y_pred3'] = y_pred3
df_test4['y_pred4'] = y_pred4

# Now plotting on the same plot
fig, ax = plt.subplots(2,2,figsize = (16,8))
ax[0,0].plot(df_test4.y_pred1,label = "SES Forecast")
ax[0,0].plot(df_test4.value,label = "Test Data")
ax[0,0].plot(df_train4.value,label = "Train Data")
ax[0,0].grid()
ax[0,0].set_title('SES Forecast: Alpha = 0')
ax[0,0].set_xlabel('Time (t)')
ax[0,0].set_ylabel('Sample Value, y(t)')
ax[0,0].legend(loc='upper left')
ax[0,1].plot(df_test4.y_pred2,label = "SES Forecast")
ax[0,1].plot(df_test4.value,label = "Test Data")
ax[0,1].plot(df_train4.value,label = "Train Data")
ax[0,1].grid()
ax[0,1].set_title('SES Forecast: Alpha = 0.25')
ax[0,1].set_xlabel('Time (t)')
ax[0,1].set_ylabel('Sample Value, y(t)')
ax[0,1].legend(loc='upper left')
ax[1,0].plot(df_test4.y_pred3,label = "SES Forecast")
ax[1,0].plot(df_test4.value,label = "Test Data")
ax[1,0].plot(df_train4.value,label = "Train Data")
ax[1,0].grid()
ax[1,0].set_title('SES Forecast: Alpha = 0.75')
ax[1,0].set_xlabel('Time (t)')
ax[1,0].set_ylabel('Sample Value, y(t)')
ax[1,0].legend(loc='upper left')
ax[1,1].plot(df_test4.y_pred4,label = "SES Forecast")
ax[1,1].plot(df_test4.value,label = "Test Data")
ax[1,1].plot(df_train4.value,label = "Train Data")
ax[1,1].grid()
ax[1,1].set_title('SES Forecast: Alpha = 0.99')
ax[1,1].set_xlabel('Time (t)')
ax[1,1].set_ylabel('Sample Value, y(t)')
ax[1,1].legend(loc='upper left')
plt.tight_layout()
plt.show()

# Q10. Creating a table to compare the 4 forecast methods that were developed
table = [['Forecasting Method', 'Q-Value','MSE (Training Set)','Mean of Residuals','Variance of Prediction Errors'],
         ["Average Forecast Method",Q_value[0],mse_prediction[0], np.mean(df_train.Avg_forecast_error.values[2:]),var_prediction[0]],
         ["Naive Forecast Method",Q_value[1],mse_prediction[1], np.mean(df_train1.Naive_forecast_error.values[2:]),var_prediction[1]],
          ["Drift Forecast Method",Q_value[2],mse_prediction[2], np.mean(df_train.Avg_forecast_error.values[2:]),var_prediction[2]],
           ["SES Forecast Method (Alpha=0.5)",Q_value[3],mse_prediction[3], np.mean(df_train.Avg_forecast_error.values[2:]),var_prediction[3]]]
print(tabulate(table,headers='firstrow', tablefmt = 'fancy_grid'))

# Q11. Creating the ACF plot for all 4 residual errors that we've obtained

lags = 5
fig, axs = plt.subplots(2,2,figsize=(16,8))
cal_autocorr(df_train.Avg_forecast_error.values[2:],lags,'Average Method Residual Errors',ax=axs[0,0])
cal_autocorr(df_train1.Naive_forecast_error.values[2:],lags,'Naive Method Residual Errors',ax=axs[0,1])
cal_autocorr(df_train2.Drift_forecast_error.values[2:],lags,'Drift Method Residual Errors',ax=axs[1,0])
cal_autocorr(df_train3.SES_forecast_error.values[2:],lags,'SES Method Residual Errors',ax=axs[1,1])
plt.tight_layout()
plt.show()