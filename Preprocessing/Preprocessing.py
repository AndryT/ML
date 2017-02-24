# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:53:30 2017

@author: Andrea

Excercises taken from:
"Python Machine Learning"
Autor: S. Raschka
"""

""" Data preprocessing """
""" Missing data - Drop data """
# create a small database
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
    1.0, 2.0, 3.0, 4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
# show dataframe
df
# Quatify how many missing data per column:
df.isnull().sum()
# from dataframe to numpy array:
df_array = df.values
# Remove rows with missing data
df.dropna()
# Remove columns with missing data
df.dropna(axis=1)
# Remove rows where all columns are NAN
df.dropna(how='all')
# Remove rows that have less more than 4 columns with NAN
df.dropna(thresh=4)
# Remove all rows where NAN appear in specific column (e.g. 'C')
df.dropna(subset=['C']) 

""" Missing Data - Imputing Data """
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
# axis = 0 => mean of values in column
# strategy = mean, median, most_frequent
imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data

""" Handling CATEGORICAL data """
df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue','XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
df
""" Mapping Ordinal Features """
# Size is an ordinal categorical feature: M < L < XL
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
df
inv_mapping = {v:k for k, v in size_mapping.items()}

""" Encoding a class label """
import numpy as np
class_mapping = {label:idx for idx, label in \
                enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
df
inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df.classlabel.map(inv_class_mapping)
df
# Alternative:
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y
class_le.inverse_transform(y)

""" One-hot encoding on Nominal feature """
X = df[['color','size','price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X
# In this way we are giving Ordinal values to a nominal feature -> not good
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
#
# Alternative in Pandas
pd.get_dummies(df[['price','color','size']])

""" Partitioning a dataset in Traning and Test """
# import a dataset to partition
df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', \
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', \
    'Nonflavanoids phenols','Proanthocyanins',\
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
print('Class label', np.unique(df_wine['Class label']))
df_wine.head()
# Partition of dataset
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, \
                            random_state = 0)
# Bring features onto a same scale
# Normalise
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
# Standardise
from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
X_train_std = stsc.fit_transform(X_train)
X_test_std = stsc.transform(X_test)

""" reduce overfitting by feature reduction via regularization L1 """
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.1) # C: regularization strength
lr.fit(X_train_std, y_train)
print('Training Accuracy: ', lr.score(X_train_std, y_train))
print('Test Accuracy: ', lr.score(X_test_std, y_test))
# Print to screen intercepts
lr.intercept_
# Print to screen (weights) coefficients
lr.coef_ # sparse solution
# Plot the regularization path - sensitivity on the regualrization parameter C
import matplotlib.pyplot as plt
colors = ['blue', 'green', 'gray', 'red', 'cyan', 'yellow', 'lightgreen', 'pink',\
    'black', 'lightblue', 'orange', 'magenta', 'indigo', ]
plt.figure()
ax = plt.subplot(111)
reg_coef = np.arange(-4,6)
weights, params =  [], []
for c in reg_coef:
    lr = LogisticRegression(penalty = 'l1', C = 10**c)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label = df_wine.columns[column+1],\
        color=color)
plt.axhline(y = 0, linestyle = '--', linewidth = 3, color = 'k')
plt.xlabel('C')
plt.ylabel('Weight coefficient')
plt.xlim([10**(-5), 10**5])
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03),ncol=1, fancybox=True)
plt.show()
plt.close()

""" Code for a Sequential Backward Selection (SBS) 
algorithm for feature selection """
from sklearn.base import clone
from itertools import combinations
# import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring = accuracy_score, \
        test_size = 0.25, random_state = 1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = self.test_size, random_state = self.random_state)
        
        dim = X_train.shape[1]            
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test,\
            self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r = dim-1): # permutations for remaining features
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]        
        return self            
        
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        
# Use SBS for KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(estimator = knn, k_features = 1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()        
plt.close()
# The graph shows that only 5 features are suffficient from an accuracy point of view
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
# Compare the accuracy between the knn classifier applied to the dataset including
# all the features and the knn classifier applied to the dataset that inlcude 
# only the 5 features detected with SBS selection feature algorithm
knn.fit(X_train_std, y_train)
print('Training accuracy: ', knn.score(X_train_std, y_train))
print('Test accuracy: ', knn.score(X_test_std, y_test))
knn.fit(X_train_std[:,k5], y_train)
print('Training accuracy: ', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy: ', knn.score(X_test_std[:, k5], y_test))      

""" Feature selection through RandomForest """
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, \
    n_jobs = 1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
# Show this information in graphical format
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color = 'lightblue',\
    align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
plt.close()
# Selection of best 15% features
#X_selected = forest.transform(X_train, threshold=0.15)
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(estimator = forest, threshold = 0.15, prefit = True)
X_selected = sfm.transform(X_train)
X_selected.shape