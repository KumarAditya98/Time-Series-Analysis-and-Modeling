import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork_Backpropagation:
    """
    A neural network with 1 - S^1 - 1 architecture
    Default: Sigmoid function in Hidden Layer and Linear function in output layer
    """
    def __init__(self,neurons,learning_rate,seed=):
        self.S1 = neurons
        self.alpha = learning_rate
        self.seed = seed
        np.random.seed(self.seed)
        self.w1 = np.random.uniform(low=-0.5,high=0.5,size=(self.S1,1))
        self.b1 = np.random.uniform(-0.5,0.5,(self.S1,1))
        self.w2 = np.random.uniform(-0.5,0.5,(self.S1,1))
        self.b2 = np.random.uniform(-0.5,0.5,(1,1))

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))

    def deriv_sigmoid(self,x):
        fx = self.sigmoid(x)
        return (1-fx)*fx

    def feedforward(self,p):
        n1 = np.dot(self.w1,p) + self.b1
        a1 = self.sigmoid(n1)
        a2 = np.dot(self.w2,a1) + self.b2
        return a2

