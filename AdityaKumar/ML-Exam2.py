import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

raw_data = pd.read_csv('ML_train.csv',index_col="Unnamed: 0")

# Preprocessing
print(raw_data.head())
print(raw_data.info())
# Junction_details is completely blank. Will drop this column
raw_data.drop(columns="Junction_Detail",axis=1,inplace=True)
print(raw_data.shape)
print(raw_data.columns)
# raw_data = raw_data.rename(columns={'Unnamed: 0':"id"})

# Treating missing values
print(raw_data.isnull().sum())
# Dropping 11 rows of missing time and other insignificant number of rows of missing data since we have a lot of data.

raw_data1 = raw_data.copy()
raw_data1 = raw_data1[raw_data1["Time"].notna()]
raw_data1 = raw_data1[raw_data1["Special_Conditions_at_Site"].notna()]
raw_data1 = raw_data1[raw_data1["Carriageway_Hazards"].notna()]
raw_data1 = raw_data1[raw_data1["Did_Police_Officer_Attend_Scene_of_Accident"].notna()]
raw_data1 = raw_data1[raw_data1["Road_Surface_Conditions"].notna()]

raw_data1['Date'] = pd.to_datetime(raw_data1['Date'],dayfirst=True)
# Splitting date into multiple features
raw_data1['Month'] = raw_data1.Date.dt.month
raw_data1['Day'] = raw_data1.Date.dt.day
# Don't need date column anymore
raw_data1.drop(columns="Date",axis=1,inplace=True)
# I'll drop time column as well as i don't have time to feature engineer it
raw_data1.drop(columns="Time",axis=1,inplace=True)

print(raw_data1.isnull().sum())
raw_data1.shape

# At this point I'll perform the train test split to avoid any leakage problems.

from sklearn.model_selection import train_test_split

# Target column is Accident_Severity
X_raw = raw_data1[['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
       'Longitude', 'Latitude', 'Police_Force',
       'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Local_Authority_(District)', 'Local_Authority_(Highway)',
       '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
       'Junction_Control', '2nd_Road_Class', '2nd_Road_Number',
       'Pedestrian_Crossing-Human_Control',
       'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions',
       'Special_Conditions_at_Site', 'Carriageway_Hazards',
       'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
       'LSOA_of_Accident_Location', 'Month','Day', 'Year']]
Y_raw = raw_data1[['Accident_Severity']]
X_raw.shape
Y_raw.shape

Y_raw = Y_raw.replace(1,0)
Y_raw = Y_raw.replace(2,0)
Y_raw = Y_raw.replace(3,1)
Y_raw.head()

Y_raw.Accident_Severity.nunique()
Y_raw.Accident_Severity.value_counts()
# the imbalance is clear with this value counts

X_train, X_test, y_train, y_test = train_test_split(X_raw, Y_raw, train_size=0.75, shuffle=True)

# I'll perform the remaining pre-processing on X_train only
X_train.isnull().sum()
# Data imputation
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(strategy='most_frequent')
mode_imputer.fit(X_train[['Junction_Control']])
X_train[['Junction_Control']] = mode_imputer.transform(
       X_train[['Junction_Control']])
X_test[['Junction_Control']] = mode_imputer.transform(
       X_test[['Junction_Control']])
# I'll do a similar imputation for the other categorical variable since i don't have time. Although this may not be correct approach.
mode_imputer = SimpleImputer(strategy='most_frequent')
mode_imputer.fit(X_train[['LSOA_of_Accident_Location']])
X_train[['LSOA_of_Accident_Location']] = mode_imputer.transform(
       X_train[['LSOA_of_Accident_Location']])
X_test[['LSOA_of_Accident_Location']] = mode_imputer.transform(
       X_test[['LSOA_of_Accident_Location']])

# I'll perform one hot encoding for all the categorical variables ---- Have to perform this for test set separately
objList = X_train.select_dtypes(include = "object").columns
print (objList)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for feat in objList:
    X_train[feat] = le.fit_transform(X_train[feat].astype(str))

objList = X_test.select_dtypes(include = "object").columns
print (objList)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for feat in objList:
    X_test[feat] = le.fit_transform(X_test[feat].astype(str))

print(X_test.info())

#Creating validation set
X_trainn, X_val, y_trainn, y_val = train_test_split(X_train,y_train, test_size=0.2)

# Now standardizing my dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_xtrain = scaler.fit_transform(X_trainn)
scaled_xval = scaler.transform(X_val)
scaled_xtest = scaler.transform(X_test)
# outlier removal
scaled_xtrain = np.clip(scaled_xtrain, -5, 5)
scaled_xval = np.clip(scaled_xval, -5, 5)
scaled_xtest = np.clip(scaled_xtest, -5, 5)
scaled_xtrain = pd.DataFrame(scaled_xtrain,columns=X_train.columns)
scaled_xval = pd.DataFrame(scaled_xval,columns=X_train.columns)
scaled_xtest = pd.DataFrame(scaled_xtest,columns=X_train.columns)

# Now looking for collinearity in the features of train set
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain, i) for i in range(scaled_xtrain.shape[1])]
vif["features"] = scaled_xtrain.columns
# print VIF values
print(vif)

# as we can see, latitude and longitude have high correlation, we'll remove on and test it again
scaled_xtrain.drop(columns='Latitude',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain, i) for i in range(scaled_xtrain.shape[1])]
vif["features"] = scaled_xtrain.columns
# print VIF values
print(vif)

# removing longtitude
scaled_xtrain.drop(columns='Longitude',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain, i) for i in range(scaled_xtrain.shape[1])]
vif["features"] = scaled_xtrain.columns
# print VIF values
print(vif)
# removing local authority district feature
scaled_xtrain.drop(columns='Local_Authority_(District)',axis=1,inplace=True)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(scaled_xtrain, i) for i in range(scaled_xtrain.shape[1])]
vif["features"] = scaled_xtrain.columns
# print VIF values
print(vif)

# All multicollinearity has been removed. I'll remove these columns for my val and test set
scaled_xval.drop(columns=['Local_Authority_(District)','Longitude','Latitude'],axis=1,inplace=True)
scaled_xtest.drop(columns=['Local_Authority_(District)','Longitude','Latitude'],axis=1,inplace=True)

# ------------------- preprocessing complete. building models

# Assumption: 1 - accident, 0 is no accident. Will tune model to have higher recall over precision.
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier


model = Sequential()
model.add(Dense(512, input_dim=28, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
    # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class_weight = {0: 6,1: 1.} # for imbalanced data handling
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

My_model = model.fit(scaled_xtrain, y_trainn, batch_size = 32,validation_data=(scaled_xval, y_val), epochs=4000, verbose=1, callbacks=[es],class_weight=class_weight)

test = pd.read_csv('Test_submission_netid_Ver_X.csv')
test.drop(columns="Junction_Detail",axis=1,inplace=True)
test['Date'] = pd.to_datetime(raw_data1['Date'],dayfirst=True)
# Splitting date into multiple features
test['Month'] = raw_data1.Date.dt.month
test['Day'] = raw_data1.Date.dt.day
# Don't need date column anymore
test.drop(columns="Date",axis=1,inplace=True)
# I'll drop time column as well as i don't have time to feature engineer it
test.drop(columns="Time",axis=1,inplace=True)
from sklearn.impute import SimpleImputer
mode_imputer = SimpleImputer(strategy='most_frequent')
mode_imputer.fit(X_train[['Junction_Control']])
test[['Junction_Control']] = mode_imputer.transform(
      test[['Junction_Control']])
objList = test.select_dtypes(include = "object").columns
print (objList)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for feat in objList:
       test[feat] = le.fit_transform(test[feat].astype(str))
test.drop(columns=['Local_Authority_(District)','Longitude','Latitude'],axis=1,inplace=True)
My_model.predict(test)

# wasn't able to train model. hence cannot give output file.
# However, everything is ready in the python file



# References:
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
