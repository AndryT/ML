# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:06:30 2017

@author: Andrea
"""

''' 
Database from:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

Code from:
"Pythoon Machine Learning"
Foreword by Dr. Randal S. Olson
PACKT Publishing
'''

import numpy as np

class Perceptron(object):
    """ Perceptron classifier.
    
    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
        
    Attributes:
    -------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        -------------
        X : {array-like}, shape = [n_sample, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_sample]
            Target values.
            
        Returns
        ---------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1]) # 1 given to w0 (w0 = threshold)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y): # tuple
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        """ Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
""" Adaptive Linear Neuron classifier (Adaline) - Batch Gradient Descent """

class AdalineGD(object):
    """ ADAptive LInear NEuron classifier 
    
    Paramters
    -------------
    eta: float
        Learning rate (between 0.0 and 1.0).    
    n_iter: int
        Passes over the training dataset.
    
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Number of missclassifications in every epoch.
    
    """
    def __init__(self, eta = 0.1, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
            
    
    def fit(self, X, y):
        """ Fit training data
        
        Parameters 
        -------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples and n_feature is the 
            number of features.
        y : array-like, shape[n_samples]
            Target values.
        
        Returns
        ------------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        """ Calculate Net-Input  """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """ Compute Linear Activation """
        return self.net_input(X)
    
    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)
        
""" Adaptive Linear Neuron classifier - Stochastic Gradient Descent """
from numpy.random import seed
class AdalineSGD(object):
    """ ADAptive LInear NEuron classifier
    
    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset.
        
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Number of missclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training data at every epoch if True to prevent cycles.
    random_state : int (default : None)
        Set random state for shuffling and initilizing the weights.

    """        
    def __init__(self, eta = 0.01, n_iter = 10, \
                shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
    
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        -------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features,
        y : array-like, shape = [n_samples]
            Target values
        
        Returns
        ------------
        self : object
        
        """
            
        
        
""" 
    1) Import Iris dataset into a DataFrame using Pandas
    2) Check the last 5 lines of the import dataset
    3) Extract data only for "Iris-Setosa" and "Iris-Versicolor" -> First 100 rows
    4) Visualise data point using matplotlib
    5) Use perceptron learning algorithm to classify dataset
    6) Visualise results
"""

import pandas as pd
import py
import matplotlib.pyplot as plt

df = pd.read_csv(py._path.local.LocalPath() + '\iris_data.csv', header = None, \
            names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Name'])
df.tail() # Show last 5 lines of table

# Extract only data for "Iris-Setosa" and "Iris-Versicolor"
df1 = pd.concat([df.loc[df.Name == "Iris-setosa",:], \
                df.loc[df.Name == "Iris-versicolor",:]])
y = df1.loc[:,'Name'].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Plotting Training data
X1 = df1.loc[df1.Name == 'Iris-setosa', ['SepalLength','PetalLength']].values
X2 = df1.loc[df1.Name == 'Iris-versicolor', ['SepalLength','PetalLength']].values
plt.scatter(X1[:, 0], X1[:, 1], color = 'r', marker = 'o', label = 'setosa')
plt.scatter(X2[:, 0], X2[:, 1], color = 'b', marker = 'x', label = 'versicolor')
#X = df1.iloc[:, [0,2]].values
#plt.scatter(X[:50, 0], X[:50, 1], color = 'r', marker = 'o', label = 'setosa')
#plt.scatter(X[50:, 0], X[50:, 1], color = 'b', marker = 'x', label = 'versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
plt.close()

''' Plotting Missclassification Error '''
ppn = Perceptron(eta = 0.1, n_iter = 10)
X = np.concatenate((X1,X2), axis = 0)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Ephocs')
plt.ylabel('Number of missclassifications')
plt.show()
plt.close()

''' Convenience function to visualize the decision boundaries '''
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) # i.e. 2
    
    # Plot the decision surface (2D)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), \
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # ravel = reshape(-1)
    Z = Z.reshape(xx1.shape)
    
    # contourf(). Filled contours without polygon edge --> for polygon edge call contour()
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha = 0.8, \
                c=cmap(idx), marker = markers[idx], label=cl)
    
plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
plt.close()


""" Testing the Adaline algorithm """
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4))
ada1 = AdalineGD(eta = 0.01, n_iter = 10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(eta = 0.0001, n_iter = 10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
plt.close()

""" Standardization of Data """
X_std = np.copy(X)
for i in range(X_std.shape[1]):
    X_std[:,i] = (X[:,i]-X[:,i].mean())/X[:,i].std()
#X_std[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()
#X_std[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()
ada = AdalineGD(eta = 0.01, n_iter = 15)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier = ada)
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.title('Adaline - Gradient Descent')
plt.legend(loc = 'upper left')
plt.show()
plt.figure()
plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
