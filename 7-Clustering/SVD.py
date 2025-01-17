# -*- coding: utf-8 -*-
"""
SVD==> Singular Value Decomposition
"""
import numpy as np
import pandas as pd
from scipy.linalg import svd
A=np.array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))
#SVD Applying to a dataset
import pandas as pd
data=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:]  #remove non numeric data
data
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()
#Scatter Diagram
import matplotlib.pylab as plt
plt.scatter(x=result.pc0,y=result.pc1)
####################################################
#####24-09-2024##########
##################################################################
import pandas as pd
import numpy as np
Univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
Univ1.describe()#Generates descriptive statistics like count, mean, std, etc., for numerical columns.
Univ1.info()#Provides a concise summary of the DataFrame, including data types, non-null counts, and memory usage.
Univ=Univ1.drop(["State"],axis=1)#Drops the "State" column from the DataFrame.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
Univ.data=Univ.iloc[:,1:]
#normilizing numerical data
uni_normal=scale(Univ.data)
uni_normal
##################################
'''
ASSOCIATION RULES==>
'''
#####26-09-2024########
#Impored required libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#Sample dataset
transactions = [
    ['Milk','Bread','Butter'],
    ['Bread','Eggs'],
    ['Milk','Bread','Eggs','Butter'],
    ['Bread','Eggs','Butter'],
    ['Milk','Bread','Eggs']
]
#step 1: Convert the dataset into a format suitable for Apriori
te = TransactionEncoder()
te_ary= te.fit(transactions).transform(transactions)
df= pd.DataFrame(te_ary, columns=te.columns_)

#step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets= apriori(df, min_support=0.5, use_colnames=True)

#step 3: Generate association rules from the frequent itemsets
rules= association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#step 4: Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])
#################################################
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#step 1: simulating healthcare transactions (symptoms/diseases/tratement)
healthcare_data=[
    ['Fever','Cough','COVID-19'],
    ['Cough','Sore Throat','Flu'],
    ['Fever','Cough','Shortness of Breath','COVID-19'],
    ['Cough','Sore Threat','FLu','Headache'],
    ['Fever','Body Ache','FLu'],
    ['Fever','Cough','COVID-19','Shortness of Breath'],
    ['Sore Threat','Headche','Cough'],
    ['Body Ache','Fatigue','Flu']
]

#step 1: Convert the dataset into a format suitable for Apriori
te = TransactionEncoder()
te_ary= te.fit(healthcare_data).transform(healthcare_data)
df= pd.DataFrame(te_ary, columns=te.columns_)

#step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets= apriori(df, min_support=0.3, use_colnames=True)

#step 3: Generate association rules from the frequent itemsets
rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#Apriori: The Apriori algorithm is used to find frequent itemsets
#with a support threshold of 0.3(i.e. patterns that occur in at least)

#step 4: Generate association rules Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])
##########################################################
#step 1: Simulate e-commerce transaction(product purchase per)
transactions=[
    ['Laptop','Mouse','Keyboard'],
    ['Smartphone','Headphones'],
    ['Laptop','Mouse','Headphones'],
    ['Smartphones','Charger','Phone Case'],
    ['Laptop','Mouse','Monitor'],
    ['Headphones','Smartwatch'],
    ['Laptop','Keyboard','Monitor'],
    ['Smartphone','Charger','Phone case','Screen Protector'],
    ['Smartphone','Headphones','Smartwatch']
]
te = TransactionEncoder()
te_ary= te.fit(transactions).transform(transactions)
df= pd.DataFrame(te_ary, columns=te.columns_)

#step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets= apriori(df, min_support=0.3, use_colnames=True)

#step 3: Generate association rules from the frequent itemsets
rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#Apriori: The Apriori algorithm is used to find frequent itemsets
#with a support threshold of 0.3(i.e. patterns that occur in at least)

#step 4: Generate association rules Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])  
#####################################################################
              
  

 





