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
X, y = df_wine.iloc[:, 1:], df_wine[:, 0:]
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
