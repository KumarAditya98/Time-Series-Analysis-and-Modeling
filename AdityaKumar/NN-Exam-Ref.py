# data pre-processing
# 1. Check range of each variable - if range is too large for some - then take log of that variable.
# 2. using box plots - identify outliers and clip them.
Q1 = df['ph'].quantile(0.05)
Q2 = df['ph'].quantile(0.95)
df['ph'] = df['ph'].clip(lower = Q1, upper = Q2)

# 3.
#---------------------------------------
# Step - Check for feature multi collineairy - VIF correlation - to be done after standardscaler.
from statsmodels.stats.outliers_influence import variance_inflation_factor
# calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
# print VIF values
print(vif)

# Drop the feature with the highest and perform again till VIF is reduced
#----------------------------------------
# Train-test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
# Further split X_train into validation and actual training data

# Covert categorical to keras categorical - refer sirs python file on keras
# Types of optimizers and metric available in keras documentation


#--------------------------------------------
# Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_xtrain = scaler.transform(X_train)
scaled_xtest = scaler.transform(X_test)
#.clip after this
#----------------------------------------------
# Metrics - Classification reports
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

# Metrics - Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(predicted, labels_test)

# Class weights
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
# Pass it as a parameter while fitting the model.

#------------------------------------------------
# Implementing Kfold technique to increase accuracy of the model
# Early stopping and model accuracy in link on edge browser.
# Hyper parameter tuning in link on edge browser.

# KerasClassifier wrapper = then implement cvfold
# Implement mlp classifier with sklearn as well
# And then do sklearn voting classifier