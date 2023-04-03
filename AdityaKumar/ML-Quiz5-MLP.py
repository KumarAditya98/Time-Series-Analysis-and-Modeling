import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np

# Q1.
mnist = load_digits()
x = pd.DataFrame(mnist.data)
y = pd.DataFrame(mnist.target)

# Q2.
fig, axes = plt.subplots(2, 2, figsize=(8,6))
num = 0
for i in range(2):
    for j in range(2):
        axes[i, j].imshow(mnist.images[num], cmap='gray');
        axes[i, j].axis('off')
        axes[i, j].set_title(f"target: {mnist.target[num]}")
        num = num+1
plt.tight_layout()
plt.show()

# Q3.
# x.shape
# y.shape
# df = pd.concat((x,y),axis=1)
# df.shape

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=6202)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000,shuffle=True)
mlp.fit(X_train, y_train.values.ravel())

# Q4.
predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Q5.
X, y = load_digits(return_X_y=True)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(mlp, X, y, cv=30,return_times=True)
plt.plot(train_sizes,np.mean(train_scores,axis=1))
