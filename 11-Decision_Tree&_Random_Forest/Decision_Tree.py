# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:25:16 2024

@author: ACER
"""
#################### DECISION TREE ##########################
'''Decision tree is always overfitted'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("C:/DATASCIENCE/12-Decision_Tree/credit.csv")
data
##data preparation
#check for null values
data.isnull().sum()
'''
o/p-->
checking_balance        0
months_loan_duration    0
credit_history          0
purpose                 0
amount                  0
savings_balance         0
employment_duration     0
percent_of_income       0
years_at_residence      0
age                     0
other_credit            0
housing                 0
existing_loans_count    0
job                     0
dependents              0
phone                   0
default                 0
dtype: int64
'''

data.dropna()
'''o/p-->
checking_balance  months_loan_duration  ... phone default
0             < 0 DM                     6  ...   yes      no
1         1 - 200 DM                    48  ...    no     yes
2            unknown                    12  ...    no      no
3             < 0 DM                    42  ...    no      no
4             < 0 DM                    24  ...    no     yes
..               ...                   ...  ...   ...     ...
995          unknown                    12  ...    no      no
996           < 0 DM                    30  ...   yes      no
997          unknown                    12  ...    no      no
998           < 0 DM                    45  ...   yes     yes
999       1 - 200 DM                    45  ...    no      no

[1000 rows x 17 columns]
'''

data.columns
'''
o/p-->
Index(['checking_balance', 'months_loan_duration', 'credit_history', 'purpose',
       'amount', 'savings_balance', 'employment_duration', 'percent_of_income',
       'years_at_residence', 'age', 'other_credit', 'housing',
       'existing_loans_count', 'job', 'dependents', 'phone', 'default'],
      dtype='object')
'''
#There are 9 columns having non numeric values, let us tree
#There is one column called phone which is not useful,
data=data.drop(["phone"],axis=1)
#Now there are 16 columns
lb=LabelEncoder()
data["Checking_balance"]=lb.fit_transform(data["checking_balance"])
data["credit_history"]=lb.fit_transform(data["credit_history"])
data["purpose"]=lb.fit_transform(data["purpose"])
data["savings_balance"]=lb.fit_transform(data["savings_balance"])
data["employment_duration"]=lb.fit_transform(data["employment_duration"])
data["other_credit"]=lb.fit_transform(data["other_credit"])
data["housing"]=lb.fit_transform(data["housing"])
data["job"]=lb.fit_transform(data["job"])

#Check for non-numeric columns
non_numeric_cols= data.select_dtypes(include=['object']).columns
print(non_numeric_cols)
data["checking_balance"]=lb.fit_transform(data["checking_balance"])
data["default"]=lb.fit_transform(data["default"])

##now let us check how many unique values are there in target column
data["default"].unique()  # array([0, 1])
data["default"].value_counts()
'''
o/p-->
default
0    700
1    300
Name: count, dtype: int64'''
##Now we want to spli tree, we need all feature columns
colnames=list(data.columns)
#Now let us assign these columns to variable predictor
predictors=colnames[:15]
target=colnames[15]

#Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT
help(DT)
model = DT(criterion='entropy')
model.fit(train[predictors],train[target])


#Prediction on the Test Data
preds=model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['predictions'])
'''
o/p-->
predictions    0   1
Actual              
0            166  39
1             52  43
'''
np.mean(preds == test[target])  #Test Data Accuracy-->0.6966666666666667

#Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Predictors'])
'''
o/p=
Predictors    0    1
Actual              
0           495    0
1             0  205
'''
np.mean(preds==train[target])  #Train Data Accuracy= 1.0
##########################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("C:/DATASCIENCE/12-Decision_Tree/salaries.csv")

inputs=df.drop('salary_more_than_100k', axis='columns')
target=df['salary_more_than_100k']

#Now there are 16 columns
lb=LabelEncoder()
data["company"]=lb.fit_transform(data["company"])
data["job"]=lb.fit_transform(data["job"])
data["degree"]=lb.fit_transform(data["degree"])
data["savings_balance"]=lb.fit_transform(data["savings_balance"])

#Check for non-numeric columns
non_numeric_cols= data.select_dtypes(include=['object']).columns
print(non_numeric_cols)
data["company"]=lb.fit_transform(data["company"])
data["salary_more_than_100k"]=lb.fit_transform(data["salary_more_than_100k"])

##now let us check how many unique values are there in target column
data["salary_more_than_100k"].unique()
data["salary_more_than_100k"].value_counts()
##Now we want to spli tree, we need all feature columns
colnames=list(data.columns)
#Now let us assign these columns to variable predictor
predictors=colnames[:15]
target=colnames[3]

#Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT
help(DT)
model = DT(criterion='entropy')
model.fit(train[predictors],train[target])


#Prediction on the Test Data
preds=model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['predictions'])

np.mean(preds == test[target])  #Test Data Accuracy

#Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Predictors'])

np.mean(preds==train[target])  #Train Data Accuracy

model.predict([[2,1,0]])

model.predict([[2,1,1]])

##############################################################################