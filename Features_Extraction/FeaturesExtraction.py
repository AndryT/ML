# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:03:05 2017

@author: Andrea

Reference:
"Pythoon Machine Learning"
Author: Sebastian Raschka
PACKT Publishing

"""
""" Feature Extraction - Unsupervised data compression
Principal Component Analysis (PCA)
"""
# 1) Upload database
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.head()
# 2) Split data in Training and Test datasets
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size = 0.3, random_state = 0)
# 3) Standardize the datasets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# 4) Create the covariance matrix
import numpy as np
cov_mat = np.cov(X_train_std.T)
# 5) Evaluate the Eigenvectors and eigenvalues for the covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
# Show variance explained ratio
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pylab as plt
plt.bar(range(1,X_train_std.shape[1]+1), var_exp, color = 'lightblue', \
    alpha = 0.5, label = 'Individual explained variance', align = 'center')
plt.step(range(1,X_train_std.shape[1]+1), cum_var_exp, where='mid', \
    label = 'Cumulative variance explained')
plt.xlabel('Principal Components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()
plt.close()

# 6) Create Transformation matrix W from covariance matrix selcting only 2 features
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) \
    for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse = True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis], eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W: \n', w)
# 7) Feature tranformation X' = Xw
X_train_pca = X_train_std.dot(w)
# Show the transformed dataset onto the new 2-dimensional subspace
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(np.ma.masked_array(X_train_pca[:, 0], y_train!=l), \
                np.ma.masked_array(X_train_pca[:, 1], y_train!=l), \
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

""" Re-run same analysis using in-built scikit-learning libraries """
# Create a method that will allow user to visualize the decision regions
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot decision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),\
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha =0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=np.ma.masked_array(X[:, 0], y != cl),\
            y=np.ma.masked_array(X[:, 1], y != cl), alpha = 0.8, \
            c = cmap(idx), marker = markers[idx], label= cl)

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
lr = LogisticRegression()
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier = lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
plt.close()
# Check performace of algorithm with test data
plot_decision_regions(X_test_pca, y_test, classifier = lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()
plt.close()
# Explained variance ratios from sklearn
pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_