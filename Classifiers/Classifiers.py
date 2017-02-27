# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:45:20 2017

@author: Andrea

Reference:
"Pythoon Machine Learning"
Author: Sebastian Raschka
PACKT Publishing
"""

""" 
The 5 steps for training a machine learning algorithm:
    1) Select the FEATURES
    2) Choosing the PERFORMANCE METRIC
    3) Select the CLASSIFIER and OPTIMIZATION algorithm
    4) Evaluate the PERFORMANCE
    5) TUNING the algorithm
"""

""" Import database and select features """
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
# select only 2 features
X = iris.data[:,[2,3]]
y = iris.target

""" Create a training and a test datasets """
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                    test_size = 0.3, random_state = 0)

""" Standardize the features for further optimization """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
# For comparability purpose the test data are scaled using the same 
# scaling parameters (mean, std) used for training data
X_test_std = sc.transform(X_test)

""" Train a Classifier algorithm - Perceptron - and test it """
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train_std, y_train)
# Prediction
y_pred = ppn.predict(X_test_std)

""" Evaluate the PERFORMANCE of the classifier using 2 performance metric:
    1) Missclassification Error
    2) Accuracy = 1 - Missclassification error
"""
print('Missclassified flowers: %d' % (y_test != y_pred).sum())
print('Missclassification Error : %.2f' % ((y_test != y_pred).sum()/len(y_test)))
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

""" Investigate the results: plot the Decision Regions """
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
    # Setup Marker generator and color map
    markers = ['s','x','o','^','v']
    colors = ['red','blue','lightgreen','gray','cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
  
    
    # Plot the decision surface - for 2 Features only !!!
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), \
                np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl ,0], y=X[ y == cl, 1], \
            alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl )
    
    # Highlight test Samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c = '', \
            alpha = 1.0, linewidths = 1, marker='o', s=55, label = 'test set')
            
""" Plotting the decision regions """
# Create unique array containing training & test data - stack them Vertically
X_combined_std = np.vstack((X_train_std, X_test_std)) # tuple argument
y_combined = np.hstack((y_train, y_test)) # tuple argument
plot_decision_regions(X_combined_std, y_combined, classifier = ppn, \
        test_idx = range(len(X_train_std), len(X_combined_std)))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.close()

""" From the plot it is clear that there is not a perfectly linear distinction
between the 3 datasets. Increasing number of epochs or decreasing the learning 
rate will not make the algorithm converge 

Instead of using the Perceptron classifier let's use a Logistic regression model
""" 
# Plot the sigmoid function in order to visualize the activation function that 
# will be used during the training
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(x=0, color = 'k')
plt.axhspan(0.0, 1.0, facecolor = '1.0', alpha = 1.0, ls='dotted')
plt.axhline(y=0.5, ls = 'dotted', color = 'k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()
plt.close()

""" Cost function for logistic regression """
phi_z = np.arange(0.0, 1.0, 0.02)
Jw_y0 = -np.log(1-phi_z)
Jw_y1 = -np.log(phi_z)
plt.plot(phi_z, Jw_y0, ls = 'dashed', label = 'J(w) if y=0')
plt.plot(phi_z, Jw_y1, color = 'blue', label = 'J(w) if y=1')
plt.xlabel('$\phi (z)$')
plt.ylabel('J(w)')
plt.legend(loc = 'upper center')
plt.show()
plt.close()

""" Training logistic regression model and predict class """
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 1000.0, random_state = 0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier = lr, \
    test_idx = range(len(X_train_std), len(X_combined_std)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.close()

""" Understanding the regularization parameter C used in the LogisticRegression """
weights, params = [], []
for c in range(-5, 5):
    lr = LogisticRegression(C = 10**c, random_state = 0)
    lr.fit(X_combined_std, y_combined)
    weights.append(lr.coef_[1]) # Attribute
    params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:,0], label = 'petal length')
plt.plot(params, weights[:,1], label = 'petal width', ls = '--')
plt.ylabel('Weight Coefficient')
plt.xlabel('Regularization Parameter C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()
plt.close()

""" Support Vector Machine algorithm - linear classifier """
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(X_combined_std, y_combined)
plot_decision_regions(X_combined_std, y_combined, classifier = svm,\
    test_idx = range(len(X_train_std), len(X_combined_std)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.close()

""" Using linear classifier for large amount of data 
    => Stochastic Gradient Descent algorithm
"""
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss = 'perceptron') # perceptron
lr = SGDClassifier(loss = 'log') # logistic regression
svm = SGDClassifier(loss = 'hinge') # Support Vector Machine

""" Solving non-linear problems using Kernel SVM """
# Create a simple dataset
np.random.seed(0)
X_xor = np.random.randn(200, 2) # sample from standard normal distribution
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], \
        c = 'b', marker = 'x', label = '1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], \
        c = 'r', marker = 's', label = '-1')
plt.ylim(-3.0)
plt.legend()
plt.show()
plt.close()
# using the Kernel to identify the decision boundaries - Radial Basis Function RBF
#from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C = 10.0, gamma = 0.1, random_state = 0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.show()
plt.close()

""" Test the Kernel function to the Iris dataset 
    Check sensitivity of gamma parameter
"""
# gamma = 0.2 --> Small gamma = soft boundaries
gamma_values = [0.1, 1, 10, 100]
for g in gamma_values:
    svm = SVC(kernel = 'rbf', random_state = 0, C = 10.0, gamma = g)
    svm.fit(X_combined_std, y_combined)
    plt.figure()    
    plot_decision_regions(X_combined_std, y_combined, classifier = svm, \
                          test_idx = range(len(X_train_std),len(X_combined_std)))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc = 'upper left')
    plt.show()
plt.close('all')

""" Decision Tree - Impurity models for 1 class """
# Gini impurity:
def gini(p):
    return (p)*(1-p)+(1-p)*(1-(1-p))
# Entropy:
def entropy(p):
    return -(p)*np.log2(p)-(1-p)*np.log2(1-p)
# Missclassification Error:
def error(p):
    return 1-np.max([p, 1-p])
x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p!=0 else None for p in x]
sc_ent = [0.5*e if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], \
    ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Missclassification Error'],\
    ['-','-','--','-.'], ['black','lightgray','red','green','cyan']):
        line = ax.plot(x, i, label=lab, linestyle = ls, lw=2, color = c)
ax.legend(loc='upper center', bbox_to_anchor = (0.5, 1.15), \
    ncol = 3, fancybox = True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

""" Building a Decision Tree """
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3,\
                random_state = 0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
plot_decision_regions(X_combined, y_combined, classifier = tree,\
        test_idx = range(len(X_train),len(X_combined)))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
plt.close()
# Export the decision tree to graphviz:
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = 'tree.dot', \
    feature_names = ['petal length', 'petal width'])
    
""" Random Forest to ensemble different tree classifiers """
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10,\
        random_state = 1, n_jobs = 2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier = forest,\
                      test_idx = range(len(X_train),len(X_combined)))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
plt.close()

""" K-Nearest Neighbors classifier """
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p =2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier = knn, \
    test_idx = range(len(X_train_std), len(X_combined_std)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()