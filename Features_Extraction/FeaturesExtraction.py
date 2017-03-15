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

""" Linear Discriminant Analysis (LDA) - Supervised data compression """
# Load database and prepare training and test data
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# Calculate mean vectors
np.set_printoptions(precision=4)
mean_vecs = []
for label in np.unique(y_train):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label,mean_vecs[label-1]))
# Calculate Within-class scatter matrix:
d = X_train_std.shape[1]
S_W = np.zeros((d,d))
for label, mv in zip(np.unique(y_test), mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X_train[y_train == label]:
        row, mv = row.reshape(d,1), mv.reshape(d,1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# Check wheteher the labels in the training set are uniformly disgtributed
print("Class label distribution: %s" % np.bincount(y_train)[1:])
# Normalize the Within-class scatter matrix
# d = 13 number of features
S_W = np.zeros((d,d))
for label, mv in zip(np.unique(y_test), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# Compute between-class scatter matrix
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)
# d = 13 number of features
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
#    mean_overall = mean_overall.reshape(d, 1)
    S_B += n*(mean_vec - mean_overall).dot((mean_vec-mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
""" Slelecting the linear discriminants for the new feature subspace """
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in \
    range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True) # k[0] = eigen_value
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
# Class-discriminatory information (discriminability)
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr  = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha = 0.5, align = 'center',\
     label='individual "discriminability"')
plt.step(range(1,14), cum_discr, where = 'mid', \
    label = 'cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc = 'best')
plt.show()
# Create the tramsformation matrix W using the first 2 most discriminative eigenvectors columns
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, \
                eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
# Projecting samples onto the new feature space
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0]*(-1), X_train_lda[y_train==l, 1]*(-1),\
        color=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()
""" LDA using the scikit learning tools """
from sklearn.lda import LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier = lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
plt.show()
# Plot the test set
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc = 'lower left')
plt.show()

""" Kernel Principal Component Analysis (Kernel-PCA) - Radial Basis Function (RBF)"""
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
#import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF Keernel PCA implementation.
    
    Parameters
    --------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
    
    gamma: float
        Tuning parameter of the RBF kernel
    
    n_components: int
        Number of principal components to return
     
    Returns
    -------------
    X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
        Projected dataset
    """
    # Calculate pairwise squared Euclidian distances in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')
        
    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)
    
    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)
    
    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh return them in sorted order
    eigvals, eigvecs = eigh(K)
    
    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1, n_components+1)))
    
    return X_pc

""" Separating half-moon shapes """
from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 100, random_state = 123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
plt.show()

# from sklearn.decomposition import PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0,0], X_spca[y==0,1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1,0], X_spca[y==1,1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0,0], np.zeros((50,1))+0.02,\
                color='r', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1,0], np.zeros((50,1))-0.02,\
                color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.show()
# Use Kernel-PCA (RBF)
from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma = 15, n_components = 2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0,0], np.zeros((50,1))+0.02, \
    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1,0], np.zeros((50,1))-0.02, \
    color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_xlabel('PC 1')
ax[1].set_yticks([])
ax[1].set_ylim([-1, 1])
ax[1].set_xlabel('PC 1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

""" Separating concentric circles """
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1], color='r', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', alpha=0.5)
plt.show()

scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1], color='r', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1], color='b', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0,0], np.zeros((500,1))+0.02, \
    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1,0], np.zeros((500,1))-0.02, \
    color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.show()

X_kpca = rbf_kernel_pca(X, gamma=15, n_components = 2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3)) 
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], \
    color='r', marker='^', alpha = 0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], \
    color='b', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, \
    color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, \
    color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_xlabel('PC 1')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
plt.show()

""" Projecting New data Point """
# Modify the RBF Kernel PCA algorithm in order tp return both eigenvectors and
# its eigenvalues. This is done to be able to normalize the eigenvectors (alpha)
# by its eigenvalues (lambda)

def rbf_kernel_pca(X, gamma, n_components):
    """ Similar to the previous method """
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    # Sorted eigenvectors and eigenvalues   
    eigvals, eigvecs = eigh(K)
    # Eigenvectors
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1, n_components+1)))
    # Eigenvalues    
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    
    return alphas, lambdas
    
# Testing modified rbf_kernel_pca algorithm
X, y = make_moons(n_samples = 100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma = 15, n_components=1)
x_new = X[25]
print(x_new)
x_proj = alphas[25] # original projection
print(x_proj)

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/lambdas)

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj

plt.scatter(alphas[y==0, 0], np.zeros((50)), color='r', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color='b', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='k', label='original projection of point X[25]',\
    marker='^', s=100)
plt.scatter(x_reproj, 0, color='g', label='remapped point X[25]', \
    marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()

""" Kernel PCA in scikit learning """
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples = 100, random_state=123)
scikit_kpca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], \
    color='r', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], \
    color='b', marker = 'o', alpha = 0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()