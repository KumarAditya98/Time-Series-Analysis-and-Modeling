import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2 ],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1 , 1],
              [-1, -1, 1 ,1, -1, -1, 1, 1]])
# -------------------------------------------------

# Answer 1:
# Plotting the Input Patterns and Targets
input_x = p[0]
input_y = p[1]
target_x = t[0]
target_y = t[1]
fig, ax = plt.subplots()
ax.plot(input_x,input_y,'bo',label='Input')
ax.plot(target_x,target_y,'ro',label='Target')
plt.grid()
ax.axhline(y = 0,color='black')
ax.axvline(x = 0,color='black')
ax.set_title("Input Vector Space With the Targets")
ax.set_xlabel("P1")
ax.set_ylabel("P2")
plt.tight_layout()
plt.legend()
plt.show()

# -------------------------------------------------
# Answer 2:
# Creating a network architecture of 2 neuron - single layer to solve this problem
# Initializing with W = [[1,0],[0,1]] and bias = 0
W = np.array([1,0,0,1]).reshape(2,2)
alpha = 0.001
bias = np.array([1,1]).reshape(2,1)

# This is created to store the P1 and P2 errors
total_error_p1 = np.array([])
total_error_p2 = np.array([])

# Applying the LMS learning rule for 1000 epochs
for i in range(1000):
    error_p1 = np.array([])
    error_p2 = np.array([])
    for j in range(0,8):
        temp = p[:,j].reshape(2,1)
        a = np.dot(W,temp) + bias
        e = t[:, j].reshape(2, 1) - a
        W = W + 2*alpha*np.dot(e,temp.T)
        bias = bias + 2*alpha*e
        error_p1 = np.append(error_p1,e[0])
        error_p2 = np.append(error_p2,e[1])
    # Appending the Sum Square Error for every epoch in the ndarray
    total_error_p1 = np.append(total_error_p1,np.sum(error_p1**2))
    total_error_p2 = np.append(total_error_p2,np.sum(error_p2**2))

# Weights and biases should've converged by now.
# -------------------------------------------------
# Answer 3:
# Plotting the Sum Square Error that was stored for each epoch. Using a log scale for x-axis and y-axis
x_tick = np.arange(0,len(total_error_p1))
series1 = pd.Series(total_error_p1,index=x_tick)
series2 = pd.Series(total_error_p2,index=x_tick)
fig, ax = plt.subplots()
ax.plot(x_tick,series1,label='Error for P1')
ax.plot(x_tick,series2,label='Error for P2')
ax.set_title("SSE Error Plot")
ax.set_xlabel("Log Scale for SSE Error")
ax.set_ylabel("Log Scale for Epochs")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
# The sum square error has seemed to converge after ~ 80 epochs

# -------------------------------------------------
# Additional Check -
# Checking the final weights given in the book solution - Very high errors still
# error_p1 = np.array([])
# error_p2 = np.array([])
# W = np.array([-0.5948,-0.0523,0.1667,-0.6667]).reshape(2,2)
# bias = np.array([0.0131,0.1667]).reshape(2,1)
# for i in range(0,8):
#     temp = p[:, i].reshape(2, 1)
#     a = np.dot(W, temp) + bias
#     e = t[:, i].reshape(2, 1) - a
#     error_p1 = np.append(error_p1, e[0])
#     error_p2 = np.append(error_p2, e[1])
# n = np.sum(error_p1**2)
# m = np.sum(error_p2**2)

# -------------------------------------------------
# Answer 4:
# Plotting the decision boundary for LMS Algorithm along with input patterns
input_x = p[0]
input_y = p[1]
target_x = t[0]
target_y = t[1]
fig, ax = plt.subplots()
ax.plot(input_x,input_y,'bo',label='Input')
plt.grid()
ax.axline((0,0.2600874861383471),(0.02232481107900362,0),color='blue',label="Decision Boundary 1")
ax.axline((0,0.2522632466931575),(-0.9998363222965161,0),color='red',label="Decision Boundary 2")
ax.set_title("Decision Boundary for LMS Algorithm")
ax.set_xlabel("P1")
ax.set_ylabel("P2")
plt.legend()
plt.show()

# ------------------------------------------------------
# Q4. To make a comparison, creating a perceptron learning rule.
# With the same initializations:
W1 = np.array([1,0,0,1]).reshape(2,2)
bias1 = np.array([1,1]).reshape(2,1)
total_error_p1 = np.array([])
total_error_p2 = np.array([])

# Defining the hardlims function for perceptron architecture
def hardlims(x):
    sample = []
    for i in range(len(x)):
        if x[i]>=0:
            sample.append(1)
        else:
            sample.append(-1)
    final = np.array(sample).reshape(len(x),1)
    return final

# Applying the Perceptron learning rule for 1000 epochs.
for i in range(1000):
    error_p1 = np.array([])
    error_p2 = np.array([])
    for j in range(0,8):
        temp = p[:,j].reshape(2,1)
        a = hardlims(np.dot(W1,temp) + bias1)
        e = t[:, j].reshape(2, 1) - a
        W1 = W1 + np.dot(e,temp.T)
        bias1 = bias1 + e
        error_p1 = np.append(error_p1,e[0])
        error_p2 = np.append(error_p2,e[1])
    total_error_p1 = np.append(total_error_p1,np.sum(error_p1**2))
    total_error_p2 = np.append(total_error_p2,np.sum(error_p2**2))

# Plotting the decision boundary for perceptron
input_x = p[0]
input_y = p[1]
target_x = t[0]
target_y = t[1]
fig, ax = plt.subplots()
ax.plot(input_x,input_y,'bo',label='Input')
plt.grid()
ax.axline((-0.2,0),(-0.2,2),color='blue',label="Decision Boundary 1")
ax.axline((0,-0.14285714285714285),(0.5,0),color='red',label="Decision Boundary 2")
ax.set_title("Decision Boundary for LMS Algorithm")
ax.set_xlabel("P1")
ax.set_ylabel("P2")
plt.legend()
plt.show()

# Plotting both the decision boundaries for comparison:
input_x = p[0]
input_y = p[1]
target_x = t[0]
target_y = t[1]
fig, ax = plt.subplots(2,1,figsize=(16,8))
ax[0].plot(input_x,input_y,'bo',label='Input')
ax[0].grid()
ax[0].set_title("Perceptron Decision Boundary")
ax[1].plot(input_x,input_y,'bo',label='Input')
ax[1].grid()
ax[0].axline((-0.2,0),(-0.2,2),color='black',label="Decision Boundary 1")
ax[0].axline((0,-0.14285714285714285),(0.5,0),color='black',label="Decision Boundary 2")
ax[1].axline((0,0.2600874861383471),(0.02232481107900362,0),color='black',label="Decision Boundary")
ax[1].axline((0,0.2522632466931575),(-0.9998363222965161,0),color='black',label="Decision Boundary")
fig.suptitle("Decision Boundary comparison b/w LMS vs Perceptron")
ax[0].set_xlabel("P1")
ax[0].set_ylabel("P2")
ax[1].set_xlabel("P1")
ax[1].set_ylabel("P2")
ax[1].set_title("LMS Algorithm Decision Boundary")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# Explanation of difference: The key difference between the LMS algorithm and Perceptron is that LMS algorithm uses the performance index, and optimizes/minimizes it that results in a decision boundary that is the farthest from the input prototype patterns, thus addressing the problem of extra noise that the future input patterns may come with, resulting in better classification. With perceptron, the learning rule is such that for the given input patterns, the weights and bias are adjusted in each iteration until the input patterns are correctly classified. This results in decision boundaries that may be close to the input patterns as long as it is being classified correctly. This may lead to issues for future unseen data that may come up with a little noise.
# As we can see from the plot above, this difference is evident. Decision boundary 1 and Decision boundary 2 for perceptron are a lot closer to the input patterns, whereas the Decision boundary 1 and Decision boundary 2 for LMS has been adjusted such that it is farthest away from all the input patterns belonging to the classes.