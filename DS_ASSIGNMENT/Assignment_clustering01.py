# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:09:20 2024

@author: ACER
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 23:57:53 2024

@author: ACER
"""
'''
1.Perform K means clustering on the airlines dataset to obtain
 optimum number of clusters. Draw the inferences from the 
clusters obtained. Refer to EastWestAirlines.xlsx dataset.

'''
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
univ1=pd.read_excel("C:/7-Clustering/EastWestAirlines.xlsx")
univ1
a=univ1.describe()
a

univ=univ1.drop(["ID#"],axis=1)
univ
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(univ.iloc[1:16,0:12])
df_norm
#################################################################
'''
2.Perform clustering for the crime data and identify the number of clusters            
formed and draw inferences. Refer to crime_data.csv dataset.
'''
import pandas as pd
df=pd.read_csv("C:/7-Clustering/crime_data.csv.xls")
df
df.head()
df.rename(columns={df.columns[0]: 'City'}, inplace=True)
df.head()
# Display all columns and exactly 13 rows
df.iloc[:14, :]


#######################################################################
'''
3.Analyze the information given in the following ‘Insurance 
Policy dataset’ to create clusters of persons falling in the 
same type. Refer to Insurance Dataset.csv
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/7-Clustering/Insurance Dataset.csv.xls")
df.head()

data = df.dropna()

# Encode categorical variables if necessary (not needed here since all are numeric)
data_encoded = data 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)
# Determine the number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)
print(data.groupby('Cluster').mean())

sns.scatterplot(x=data['Age'], y=data['Income'], hue=data['Cluster'], palette="deep")
plt.title('Clusters of Insurance Policyholders')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
##################################################
'''
4.Perform clustering analysis on the telecom dataset. The data 
is a mixture of both categorical and numerical data. It consists
of the number of customers who churn. Derive insights and get 
possible information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load the dataset
df = pd.read_excel('Telco_customer_churn.xlsx')

# Identify categorical and numerical columns
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Perform KMeans clustering with the optimal number of clusters
optimal_k = 4  # Replace with the optimal number of clusters found from the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Analyze clusters
cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary (Numerical Features):")
print(cluster_summary)

# For categorical features, get the most common value in each cluster
categorical_summary = df.groupby('Cluster')[categorical_features].agg(lambda x: x.value_counts().index[0])
print("\nCluster Summary (Categorical Features):")
print(categorical_summary)

# Visualize clusters
sns.pairplot(df, hue='Cluster', vars=numerical_features)
plt.show()

# Optionally, calculate silhouette score for the chosen number of clusters
silhouette_avg = silhouette_score(X, df['Cluster'])
print(f'\nSilhouette Score: {silhouette_avg}')
####################################################################

'''
5.Perform clustering on mixed data.Convert the categorical 
variables to numeric by using dummies or label encoding and 
perform normalization techniques. The dataset has the details 
of customers related to their auto insurance. Refer to 
Autoinsurance.csv dataset.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('Autoinsurance.csv')

# Identify categorical and numerical columns
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Perform KMeans clustering with the optimal number of clusters
optimal_k = 4  # Replace with the optimal number of clusters found from the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Analyze clusters
cluster_summary = df.groupby('Cluster').mean()
print("Cluster Summary (Numerical Features):")
print(cluster_summary)

# For categorical features, get the most common value in each cluster
categorical_summary = df.groupby('Cluster')[categorical_features].agg(lambda x: x.value_counts().index[0])
print("\nCluster Summary (Categorical Features):")
print(categorical_summary)

# Visualize clusters
sns.pairplot(df, hue='Cluster', vars=numerical_features)
plt.show()
#########################################################
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:/7-Clustering/crime_data.csv.xls")

# Display the first few rows
print(df.head())
# Drop any rows with missing values (if applicable)
data = df.dropna()

# If the first column is just the state names, you may want to exclude it from clustering
data_clustering = data.iloc[:, 1:]  # Assuming the first column is not needed

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clustering)
from sklearn.cluster import KMeans

# Use the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow graph, choose the number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Analyze the clusters
print(df.groupby('Cluster').mean())

import seaborn as sns

# Visualizing the clusters (example with two features, you can change these as needed)
sns.scatterplot(x=data['Murder'], y=data['Assault'], hue=df['Cluster'], palette='viridis')
plt.title('Clusters of Crime Data')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.show()

###################################
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/7-Clustering/crime_data.csv")  # Adjust file extension if necessary

# Display the first few rows and data types
print(df.head())
print(df.dtypes)

# Drop any rows with missing values
data = df.dropna()

# If the first column is not needed for clustering
data_clustering = data.iloc[:, 1:]

# Convert all columns to numeric where possible
for col in data_clustering.columns:
    data_clustering[col] = pd.to_numeric(data_clustering[col], errors='coerce')

# Drop rows with NaN values after conversion
data_clustering = data_clustering.dropna()

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clustering)

# Use the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the elbow graph, choose the number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled)

# Analyze the clusters
print(df.groupby('Cluster').mean(numeric_only=True))  # Use numeric_only to avoid issues with non-numeric data

# Visualizing the clusters (example with two features, adjust as needed)
# Replace 'Feature1' and 'Feature2' with actual column names you want to visualize
sns.scatterplot(x=df['Murder'], y=df['Rape'], hue=df['Cluster'], palette='viridis')
plt.title('Clusters of Crime Data')
plt.xlabel('Murder')
plt.ylabel('Rape')
plt.show()
##########################
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/7-Clustering/crime_data.csv.xls")

# Removing the State column since it's categorical
X = df.drop('State', axis=1)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, let's assume the optimal number of clusters is 3
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
sns.pairplot(df, hue='Cluster', palette='Set2', diag_kind='kde')
plt.show()

# Displaying the clustered data
print(df)



