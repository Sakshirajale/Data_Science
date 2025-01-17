# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:21:35 2024

@author: ACER
"""
################# RANDOM FOREST ALGORITHM ######################

import pandas as pd
df=pd.read_csv("C:/12-Decision_Tree/movies_classification.csv")
df.info()
#movies_classification dataset contains two columns which are object
#hence convert into dummies
df=pd.get_dummies(df, columns=["3D_available","Genre"],drop_first=True)
#let us assign input and output variables
predictors=df.loc[:,df.columns!="Start_Tech_Oscar"]
#Except start_Tech_Oscar, rest all columns are assigned as predictors
#predictor has got 20 columns
target=df["Start_Tech_Oscar"]
#############################
#let us partition the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(predictors, target, test_size=0.2)

#############################
#model selection
from sklearn.ensemble import RandomForestClassifier
rand_for=RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
#n_estimators: It is number of trees in the forest, always in range 500 to 1000
#n_jobs=1 means number of jobs running paralle=1, if is -1 then multiplication
#random_state=controls randomness in bootstraping
#Bootstraping is getting sample replaced
rand_for.fit(X_train,y_train)
pred_X_train=rand_for.predict(X_train)
pred_X_test=rand_for.predict(X_test)
#####
#let us check the performance of the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(pred_X_test,y_test)
confusion_matrix(pred_X_test,y_test)

################
#for training dataset
accuracy_score(pred_X_train,y_train)
confusion_matrix(pred_X_train,y_train)
#################################

