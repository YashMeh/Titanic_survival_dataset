#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:53:07 2018

@author: yash
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('train.csv')
y=dataset.iloc[:,1:2].values
X=dataset.iloc[:,[2,4,5,6,7,9,11]].values

#Encoding the sex column
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#Filling in missing age column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
#Fillung in missing values of place column
places=['Q','S','C']
for i in range(len(X[:,6])):
    if X[:,6][i] not in places:
        X[:,6][i]='S'
    
#Encoding the place column
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Random Forest Classifier -83.79
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print('The accuracy score of this model is ',score*100)

#SVC -81.56
from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print('The accuracy score of this model is ',score*100)
