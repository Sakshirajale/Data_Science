# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 00:00:07 2024

@author: ACER
"""
'''
1.	Business Problem
1.1.What is the business objective?
==>The objective is to support a pharmaceutical company studying a new 
heart disease medication by providing analytical insights through patient data 
segmentation.

1.2.Are there any constraints?
==>Potential constraints for the project include:
    *Data Quality
    *Interpretability
    *Computational Limitations
    *Optimal Cluster Selection
    *Scalability
    
2.Work on each feature of the dataset to create a data dictionary as displayed in the below image:
 2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.
==>Steps for Completing the Data Dictionary
  *Identify Data Types: Determine if each feature is numeric, categorical, or text.
  *Describe Features: Provide meaningful context for each feature.
  *Evaluate Relevance: Assess if each feature contributes valuable information to model building.
  *Document Irrelevance: Justify why any feature might not be suitable for inclusion.
  
3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.
==>-*Data Cleaning:
      #Handling Missing Values:Use imputation techniques like mean, median, or mode for numeric features.

      #Identify outliers using methods like the Z-score, IQR, or visualization tools (box plots).

     #Noise Reduction:Eliminate noisy data points that may negatively impact clustering performance.
   
    -*Data Transformation:
      #Normalization/Standardization:Scale numeric features using techniques such as Min-Max Scaling or StandardScaler to ensure balanced clustering performance.

      #Encoding Categorical Data:Apply one-hot encoding for categorical features if necessary for model compatibility.

    -*Feature Engineering:
     #Derived Features:Create meaningful features that may better represent the data, such as "Age Group" derived from Age.

     #Dimensionality Reduction:Use PCA to reduce feature dimensions while retaining maximum variance.

    -*Data Balancing:Ensure balanced data distribution if clustering may be biased due to imbalanced sample sizes.
    
    -*Handling Duplicate Records:Identify and remove duplicate rows to avoid redundant data points.


4.	Exploratory Data Analysis (EDA):
4.1 Summary
Overview Statistics:
--Mean, median, mode, standard deviation, minimum, and maximum values for numeric features.
--Frequency distribution for categorical features.
Missing Values Analysis:
--Count and percentage of missing values per feature.
Correlation Matrix:
--Visualize correlations between numerical features to identify potential multicollinearity or redundant features.

4.2 Univariate Analysis
Numeric Features:
--Histograms and Density Plots: To understand the distribution of each variable.
--Box Plots: For detecting outliers and visualizing spread.

Categorical Features:
--Bar Plots: For frequency distribution.
--Pie Charts: To visualize the proportion of different categories.

4.3 Bivariate Analysis
Numeric-Numeric Relationships:
--Scatter plots to explore relationships between pairs of numeric features (e.g., Age vs. Cholesterol).
--Heatmaps to visualize correlation between features.

Numeric-Categorical Relationships:
--Box plots grouped by categorical features (e.g., Cholesterol distribution across different Age Groups).

Categorical-Categorical Relationships:
--Stacked bar charts to show relationships between categorical features.

5.	Model Building
This section details the steps to build, evaluate, and explain models for clustering and dimensionality reduction using PCA.

5.1 Build the Model on Scaled Data (Multiple Options)
--Scaling: Standardize features to zero mean and unit variance using StandardScaler or MinMaxScaler.

Modeling Options:
--K-Means Clustering:
   *Specify a range for k to test optimal cluster numbers.
   *Evaluate using metrics like inertia (within-cluster sum of squares) or silhouette score.

--Hierarchical Clustering:
   *Use dendrograms to visualize the clustering process and determine the number of clusters.
   *Test linkage methods such as single, complete, and average linkage.
   

5. Model Building
This section details the steps to build, evaluate, and explain models for clustering and dimensionality reduction using PCA.

5.1 Build the Model on Scaled Data (Multiple Options)
Scaling: Standardize features to zero mean and unit variance using StandardScaler or MinMaxScaler.
Modeling Options:
K-Means Clustering:

Specify a range for k to test optimal cluster numbers.
Evaluate using metrics like inertia (within-cluster sum of squares) or silhouette score.
Hierarchical Clustering:

Use dendrograms to visualize the clustering process and determine the number of clusters.
Test linkage methods such as single, complete, and average linkage.
5.2 Perform PCA Analysis and Get Maximum Variance Between Components
--Apply PCA:
    *Extract the first three principal components while retaining maximum variance.
    *Analyze the explained variance ratio to assess the contribution of each component.

5.3 Perform Clustering Before and After PCA
 1)Before PCA:
   *Run clustering algorithms on the original scaled dataset.
   *Evaluate performance metrics and visualize clusters.
 2)After PCA:
   *Perform clustering on the PCA-transformed dataset.
   *Compare cluster centroids and visualize results using scatter plots in 2D or 3D.
Optimum Clusters Determination:
 --Elbow Method: Plot inertia values against k to find the point where the curve bends.
 --Silhouette Analysis: Evaluate how well data points fit within their assigned clusters.

5.4 Model Output Documentation
Cluster Analysis Results:
--Describe the number of clusters formed before and after PCA.
S--ummarize patterns or insights (e.g., patient groups with higher cholesterol levels or specific age groups).

Variance Contribution:
--Highlight how much variance is explained by the principal components.

Performance Comparison:
--Discuss clustering effectiveness before and after PCA.
'''
################################################################################
'''
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. After that,
perform PCA on the dataset and extract the first 3 principal components 
and make a new dataset with these 3 principal components as the columns. 
Now, on this new dataset, perform hierarchical and K-means clustering. 
Compare the results of clustering on the original dataset and clustering 
on the principal components dataset (use the scree plot technique to obtain 
the optimum number of clusters in K-means clustering and check if you’re 
getting similar results with and without PCA).
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('C:/9-PVD_PCA/wine.csv.xls') 
data_features = data.drop('Type', axis=1)  
data
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_features)

#Step 1
# Perform hierarchical clustering
linkage_matrix = linkage(scaled_data, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

#Step 2:K-Means Clustering
#determine the optimal number of clusters
inertia = []
range_clusters = range(1, 10)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters (Original Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Choose optimal k
optimal_k = 3
kmeans_original = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_original.fit(scaled_data)
labels_original = kmeans_original.labels_

#Step 3: PCA Analysis
# Apply PCA and extract the first 3 principal components
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# Scree Plot 
plt.figure(figsize=(8, 4))
plt.plot(range(1, 4 + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

print("Explained Variance Ratios for First 3 Principal Components:", pca.explained_variance_ratio_)

#Step 4: Clustering on PCA Data 
# Perform K-means clustering on PCA-transformed data
kmeans_pca = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_pca.fit(pca_data)
labels_pca = kmeans_pca.labels_

#Step 5: Silhouette Scores for Comparison 
silhouette_original = silhouette_score(scaled_data, labels_original)
silhouette_pca = silhouette_score(pca_data, labels_pca)

print(f"Silhouette Score (Original Data): {silhouette_original:.3f}")
print(f"Silhouette Score (PCA Data): {silhouette_pca:.3f}")

#Step 6: Visualizing Clusters
# Scatter plot of clusters on PCA components
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels_pca, cmap='viridis', marker='o')
plt.title('Clusters on PCA Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

#########################################################################################
'''
Problem Statement: -
A pharmaceuticals manufacturing company is conducting a study on a new 
medicine to treat heart diseases. The company has gathered data from its 
secondary sources and would like you to provide high level analytical insights 
on the data. Its aim is to segregate patients depending on their age group and 
other factors given in the data. Perform PCA and clustering algorithms on the 
dataset and check if the clusters formed before and after PCA are the same and 
provide a brief report on your model. You can also explore more ways to improve 
your model. 
Note: This is just a snapshot of the data. The datasets can be downloaded from 
AiSpry LMS in the Hands-On Material section.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('C:/9-PVD_PCA/heart disease.csv.xls')

#Data Preprocessing
print("Missing values in each column:\n", data.isnull().sum())

# Standardize the data 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#PCA Analysis 
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# Explained Variance Ratio Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Explained Variance Ratio by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

#K-Means Clustering Before PCA
kmeans_original = KMeans(n_clusters=3, random_state=42)
kmeans_original.fit(scaled_data)
labels_original = kmeans_original.labels_

#K-Means Clustering After PCA
kmeans_pca = KMeans(n_clusters=3, random_state=42)
kmeans_pca.fit(pca_data)
labels_pca = kmeans_pca.labels_

#Score Comparison 
silhouette_original = silhouette_score(scaled_data, labels_original)
silhouette_pca = silhouette_score(pca_data, labels_pca)

print(f'Silhouette Score (Original Data): {silhouette_original:.3f}')
print(f'Silhouette Score (PCA Data): {silhouette_pca:.3f}')

#Cluster Visualization After PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels_pca, cmap='viridis', marker='o')
plt.title('Clusters After PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()


