import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from numpy import linalg as la

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/autos.clean.csv'

df = pd.read_csv(url)
df = df[['normalized-losses', 'wheel-base', 'length', 'width','height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg','price']]
X = df[['normalized-losses', 'wheel-base', 'length', 'width','height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg','highway-mpg']]
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=6313)

# Q1
print(f"The X train shape is: {X_train.shape}\nThe Y train shape is: {y_train.shape}\nThe X test shape is: {X_test.shape}\nThe Y test shape is: {y_test.shape}")

# Q2
plt.figure(figsize=(13, 13))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.show()

# Q3a
# X.insert(loc=0, column='Bias', value=np.ones(len(X),dtype=int))
X_matrix = X.values
# X_matrix = np.insert(X_matrix, 0, int(1), axis=1)
H = np.dot(X_matrix.T,X_matrix)
s, d, v = np.linalg.svd(H)
print("SingularValues = ", d)

# Q3b
Cond = la.cond(X)
print("The Condition Number of the feature space is: ",Cond)

# Q3c
from itertools import combinations
col = X.columns
Count = 13
Switch = 0
for i in reversed(range(1,Count+1)):
    comb = combinations(col,i)
    for cols in comb:
        columns = list(cols)
        tempdf = X[columns]
        cond_num = la.cond(tempdf)
        if cond_num < 100:
            Switch = 1
            break;
        continue;
    if Switch == 1:
        print(f"The number of features that need to be removed to eliminate collinearity is: {14-i}\nThe Feature set that helps eliminate collinearity is: {columns}\nAnd the Condition number that it produces is: {cond_num} ")
        break;
    continue;

# Q4
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# tempdf = X_train.copy()
# tempdf['target'] = y_train
scaled_xtrain = scaler.fit_transform(X_train)
Xtrain_scaled = pd.DataFrame(scaled_xtrain,columns=X.columns)
# Xtrain_scaled = tempdf_scaled1.iloc[:,:-1]
# Ytrain_scaled = tempdf_scaled1.iloc[:,-1]
# tempdf1 = X_test.copy()
# tempdf1['target'] = y_test
scaled_xtest = scaler.transform(X_test)
Xtest_scaled = pd.DataFrame(scaled_xtest,columns=X.columns)
# Xtest_scaled = tempdf1_scaled1.iloc[:,:-1]
# Ytest_scaled = tempdf1_scaled1.iloc[:,-1]
# Xtrain_scaled = scaler.fit_transform(X_train)
# Xtest_scaled = scaler.transform(X_test)

# Q5
Xtrain_scaledd = Xtrain_scaled.copy()
Xtrain_scaledd.insert(loc=0, column='Bias', value=np.ones(len(Xtrain_scaled),dtype=int))
X_mat = Xtrain_scaledd.values
Y_mat = y_train.values
# y_train = y_train.to_numpy().reshape(len(y_train),1)
# Ytrain_scaled = Ytrain_scaled.to_numpy().reshape(len(Ytrain_scaled),1)
beta_hat = np.dot(la.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, Y_mat))
# beta_hat = np.dot(la.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, Ytrain_scaled))
print(f"The Coefficients from the Normal equation are: {beta_hat.reshape(1,-1)}")

# Q6
Xtrain_scaled = sm.add_constant(Xtrain_scaled) # To add constant intercept
y_train = y_train.reset_index()
y_train = y_train.drop(columns='index',axis=1)
model = sm.OLS(y_train,Xtrain_scaled).fit()
print(model.params)

# Q7
# Looking at the summary with all variables first
print(model.summary())
# Noting down the Adjusted R-squared, AIC and BIC values of this model
# Bias has the highest p-value and appears extremely insignificant. However, the value is extremely small and can be ignored in this model.
# Next highest p-value is normalized-losses. Removing that and checking the model performance
Xtrain_scaled1 = Xtrain_scaled.drop(columns=['normalized-losses'],axis=1)
model1 = sm.OLS(y_train,Xtrain_scaled1).fit()
print(model1.summary())
# Model performance gone up. We'll drop that feature
# Next highest p-value is of bore. Removing that and checking the model performance
Xtrain_scaled2 = Xtrain_scaled1.drop(columns=['bore'],axis=1)
model2 = sm.OLS(y_train,Xtrain_scaled2).fit()
print(model2.summary())
# Model performance gone up. We'll drop that feature
# Next highest p-value is of wheel-base. Removing that and checking the model performance
Xtrain_scaled3 = Xtrain_scaled2.drop(columns=['wheel-base'],axis=1)
model3 = sm.OLS(y_train,Xtrain_scaled3).fit()
print(model3.summary())
# Model performance gone up. We'll drop that feature
# Next highest p-value is of highway-mpg. Removing that and checking the model performance
Xtrain_scaled4 = Xtrain_scaled3.drop(columns=['highway-mpg'],axis=1)
model4 = sm.OLS(y_train,Xtrain_scaled4).fit()
print(model4.summary())
# Model performance has gone up. We'll drop that feature
# Next highest p-value is of city-mpg. Removing that and checking the model performance
Xtrain_scaled5 = Xtrain_scaled4.drop(columns=['city-mpg'],axis=1)
model5 = sm.OLS(y_train,Xtrain_scaled5).fit()
print(model5.summary())
# Model performance has gone up. We'll drop that feature
# Next highest p-value is of length. Removing that and checking the model performance
Xtrain_scaled6 = Xtrain_scaled5.drop(columns=['length'],axis=1)
model6 = sm.OLS(y_train,Xtrain_scaled6).fit()
print(model6.summary())
# Model performance has gone up. We'll drop that feature
# Next highest p-value is of height. Removing that and checking the model performance
Xtrain_scaled7 = Xtrain_scaled6.drop(columns=['height'],axis=1)
model7 = sm.OLS(y_train,Xtrain_scaled7).fit()
print(model7.summary())
# Model performance has gone up. We'll drop that feature
# Next highest p-value is of curb-weight. Removing that and checking the model performance
Xtrain_scaled8 = Xtrain_scaled7.drop(columns=['curb-weight'],axis=1)
model8 = sm.OLS(y_train,Xtrain_scaled8).fit()
print(model8.summary())
# Model performance has gone up/not changed. We'll drop that feature
# Next highest p-value is of horsepower. Removing that and checking the model performance
Xtrain_scaled9 = Xtrain_scaled8.drop(columns=['horsepower'],axis=1)
model9 = sm.OLS(y_train,Xtrain_scaled9).fit()
print(model9.summary())
# The model performance has decreased by removing this feature. This is where I will stop removing features
# Model8 is my final model.

# Q10.
# Developing the final regression model
print(model8.params)
Xtrain_scaled_final = Xtrain_scaled.drop(columns=['normalized-losses','bore','wheel-base','highway-mpg','city-mpg','length','height','curb-weight'],axis=1)
model_final = sm.OLS(y_train,Xtrain_scaled_final).fit()
print(model_final.summary())

# Q11
Xtest_scaled = sm.add_constant(Xtest_scaled)
# Dropping the columns based on above analysis in test set as well.
Xtest_scaled_final = Xtest_scaled.drop(columns=['normalized-losses','bore','wheel-base','highway-mpg','city-mpg','length','height','curb-weight'],axis=1)
predictions = model_final.predict(Xtest_scaled_final)

predictions.index = np.arange(160,201,1)
y_test.index = np.arange(160,201,1)
fig, ax = plt.subplots(figsize = (16,8))
ax.plot(predictions,label="Predicted Price")
ax.plot(y_test,label="Observed Price",alpha = 0.5)
ax.plot(y_train,label="Training Price")
plt.legend(loc='upper left')
plt.grid()
plt.title('Price Graph - Train, Test, Predicted')
plt.xlabel('Index')
plt.ylabel('Price')
plt.show()

# Q12
Error = np.array(y_test-predictions)
lags = 20
from toolbox import cal_autocorr
cal_autocorr(Error,lags,'Prediction Error')

plt.show()

# MSE of the dataset
MSE = np.square(np.subtract(y_test,predictions)).mean()

# Q13
model_final.summary()