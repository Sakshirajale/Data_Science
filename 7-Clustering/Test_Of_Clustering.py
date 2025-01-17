# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:14:46 2024

@author: ACER
"""
'''
1. You are given a dataset with two numerical features Height and Weight. 
Your goal is to cluster these people into 3 groups using K-Means clustering. 
After clustering, you will visualize the clusters and their centroids. 
 Load the dataset (or generate random data for practice). 
 Apply K-Means clustering with k = 3. 
 Visualize the clusters and centroids. 
 Experiment with different values of k and see how the clustering changes.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:/7-Clustering/HeightWeight.csv")
df

# Print column names to check what is available in the dataset
print("Columns in the dataset:", df.columns)


X= df[['Height(Inches)', 'Weight(Pounds)']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering with k = 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels and centroids
df['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

# Inverse transform centroids back to original scale
centroids_original = scaler.inverse_transform(centroids)

# Visualize the clusters and centroids
plt.figure(figsize=(10, 6))
plt.scatter(df['Height(Inches)'], df['Weight(Pounds)'], c=df['Cluster'], cmap='viridis', marker='o', s=100, label='Data points')
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering (k=3)')
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
plt.legend()
plt.show()



'''
2. You have a dataset of  customers with features Age, Annual Income, and 
Spending Score. You need to apply hierarchical clustering to segment these 
customers. Plot a dendrogram to decide the optimal number of clusters and 
compare it with K-Means clustering results. 
Steps: 
 Load the dataset. 
 Apply hierarchical clustering. 
 Plot a dendrogram and choose the number of clusters. 
 Apply K-Means clustering with the same number of clusters. 
 Compare the results.
'''
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:/7-Clustering/Mall_Customers.csv")
df


# Check the first few rows to understand the structure
print(df.head())

# Select the features for clustering (Age, Annual Income, Spending Score)
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot the dendrogram for hierarchical clustering
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# From the dendrogram, select the optimal number of clusters (e.g., k=5)
n_clusters = 5

# Apply Agglomerative (Hierarchical) Clustering
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
hc_labels = hc.fit_predict(X_scaled)

# Add the labels to the dataframe
df['Hierarchical Cluster'] = hc_labels

# Apply K-Means clustering with the same number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add the K-Means labels to the dataframe
df['KMeans Cluster'] = kmeans_labels

# Compare the clustering results
print(df[['Hierarchical Cluster', 'KMeans Cluster']].head())

# Visualize the clusters (for simplicity, let's plot only Annual Income and Spending Score)
plt.figure(figsize=(10, 7))

# Scatter plot for Hierarchical Clustering
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=hc_labels, cmap='rainbow')
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

# Scatter plot for K-Means Clustering
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=kmeans_labels, cmap='rainbow')
plt.title('K-Means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.show()
##########################################################################