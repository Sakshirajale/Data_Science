# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:13:07 2024

@author: ACER
"""
###06-08-2024###
'''
Feature->unsupervised
feature+label/target->supervised
unsupervised->clustering
'''
############09-08-2024##########
import pandas as pd
import matplotlib.pyplot as plt
#Now import file from data set and create a dataframe
univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
univ1
a=univ1.describe()
a
#We have one column "state" which really not useful we will drop it
univ=univ1.drop(["State"],axis=1)
univ
#We know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to univ dataframe
#for all the rows and columns from 1 until end
#since 0th column has university name hence skipprd
df_norm=norm_func(univ.iloc[:,1:])
df_norm
#you can check the df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new datafFrame
b=df_norm.describe()
b
#Before you apply clustering,you need to plot dendrograme first
#Now to create dendogram, we need to measure distance, we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierachical or agloramative clustering
#ref the help for linkage
z=linkage
###########################
############10-08-2024##########
import pandas as pd
import matplotlib.pyplot as plt
#Now import file from data set and create a dataframe
Univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
Univ1
a=Univ1.describe()
a
#We have one column "state" which really not useful we will drop it
Univ=Univ1.drop(["State"],axis=1)
Univ
#We know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to univ dataframe
#for all the rows and columns from 1 until end
#since 0th column has university name hence skipprd
df_norm=norm_func(Univ.iloc[:,1:])
df_norm
#you can check the df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new datafFrame
b=df_norm.describe()
b
#Before you apply clustering,you need to plot dendrograme first
#Now to create dendogram, we need to measure distance, we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierachical or agloramative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendogram")
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show
#applying agglomerative clustering choosing 5 as cluster from dendogram
#whatever has been 
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
#apply labels to the cluster
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to univ DataFrame as column and name the column
Univ['clust']=cluster_labels
#we want to relocate the column 7 to 0th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#Now check the univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from the o/p cluster 2 has got highest top 10
##########13-08-2024##############################
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("C:/7-Clustering/income.csv.xls")
df.head()
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_
#There is data is not there are cluster
df1=df[df.cluster==0]
#
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()

#Preprocessing using min max scaler
scaler=MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
df.head()

plt.scatter(df.Age,df['Income($)'])
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
#################################
##########14-08-2024########
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
#Let us try to understand first how k means works for two dimensional data
#for that, generate random numbers in the range 0 to 1
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=["X","Y"])
#Assign the values of X and Y to these coluns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind="scatter")
model1=KMeans(n_clusters=3).fit(df_xy)
'''
with data X and Y, apply KMeans model,
generate scatter plot
with scale/font=10
cmap=plt.cm.coolwarm:cool color combination
'''
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)
####################################
Univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
#we know that there is scale difference among the columns,which we 
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to Univ dataframe for all the
df_norm=norm_func(Univ.iloc[:,1:])
df_norm
'''
What will be ideal cluster number, will it be 1,2 or 3
'''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    #Total within sum of square
    TWSS.append(kmeans.inertia_) 

'''
KMeans inertia, also known as Sum of Squares Errors
(or SSE), calculates the sum of the distances
of all points within a cluster from the centroid of the 
point. It is the difference between the observed value and the
predicted value.
'''

TWSS
#As k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-');
plt.xlabel("No_of_clusters");
plt.ylabel("Total_within_SS")

'''
How to select value of k from elbow curve
when k changes from 2 to 3, then decrease
in twss is higher than
When k changes from 3 to 4
When k changes from 5 to 6 decreases
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ['clust']=mb
Univ.head()
Univ=Univ.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv("kmeans_university.csv",encoding="utf-8")
import os
os.getcwd() 
####################################################################





