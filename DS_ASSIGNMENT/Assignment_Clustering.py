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
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
df = pd.read_excel("C:/7-Clustering/EastWestAirlines.xlsx")

# Step 3: Preprocessing 
# Assuming the first column is an identifier and we drop it
df = df.drop(columns=['ID#'])

# Step 4: Normalize the data 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 5: Finding the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)  # We will try K from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

#Step 6: Plot the curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-', color='blue')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 7: Apply KMeans clustering using the optimal number of clusters (e.g., K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 8: Analyze and print the clusters
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels:\n", df['Cluster'].value_counts())

# Save the results to a new Excel file
df.to_excel("Airlines_Clustered.xlsx", index=False)
print("Clustering completed. Results saved to 'Airlines_Clustered.xlsx'.")
#################################################################
'''
2.Perform clustering for the crime data and identify the number of clusters            
formed and draw inferences. Refer to crime_data.csv dataset.
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("C:/7-Clustering/crime_data.csv.xls")
df

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])  
#first fits the model to the data and then immediately returns the cluster labels for each data point.

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange',label='Cluster 0')
plt.scatter(df2.Murder, df2['Assault'], color='blue', label='Cluster 1')
plt.scatter(df3.Murder, df3['Assault'], color='green',label='Cluster 2')


# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Murder']])
df['Murder']=scaler.transform(df[['Murder']])

scaler.fit(df[['Assault']])
df['Assault']=scaler.transform(df[['Assault']])

df.head()

plt.scatter(df.Murder,df['Assault'])

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange',label='Cluster 0')
plt.scatter(df2.Murder, df2['Assault'], color='blue', label='Cluster 1')
plt.scatter(df3.Murder, df3['Assault'], color='green',label='Cluster 2')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
#plt.title('K-Means Clustering')
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()


####### ELBOW CURVE=> NO. OF CLUSTERS .....

scaler = MinMaxScaler()
df['Murder'] = scaler.fit_transform(df[['Murder']])
df['Assault'] = scaler.fit_transform(df[['Assault']])

# Step 1: Use the Elbow Method to find the optimal number of clusters
TWSS = []

k = list (range(2,8))
for i in k:
    km = KMeans(n_clusters=i)
    km.fit(df[['Murder', 'Assault']])
    TWSS.append(km.inertia_)

# Step 2: Plot the Elbow graph
plt.figure(figsize=(10,6))
plt.plot(k, TWSS, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method For Optimal K')
plt.show()


#######################################################################
'''
3.Analyze the information given in the following ‘Insurance 
Policy dataset’ to create clusters of persons falling in the 
same type. Refer to Insurance Dataset.csv
'''
# Step 1: Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
file_path = "C:/7-Clustering/Insurance Dataset.csv.xls"  
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Step 3: Preprocess the data
print(df.isnull().sum())  # Check for missing values

# If missing values exist, you can fill or drop them. Example:
# df = df.dropna()  # or df.fillna(method='ffill')

# Step 4: Normalize the data (since K-Means is sensitive to scale)
scaler = StandardScaler()

# Assuming all columns except the ID column are features, adjust if needed
df_scaled = scaler.fit_transform(df.drop(columns=['Premiums Paid']))  # Drop ID column if present

# Step 5: Use the Elbow Method to find the optimal number of clusters
inertia = []
K = range(1, 11)  # Trying K from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Step 6: Plot the Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-', color='blue')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 7: Choose the optimal K (based on the Elbow curve) and fit KMeans
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 8: Analyze the clusters
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels:\n", df['Cluster'].value_counts())

# Step 9: Save the clustered data to a new CSV file
df.to_csv("Insurance_Clustered.csv", index=False)



##################################################
'''
4.Perform clustering analysis on the telecom dataset. The data 
is a mixture of both categorical and numerical data. It consists
of the number of customers who churn. Derive insights and get 
possible information on factors that may affect the churn decision. Refer to Telco_customer_churn.xlsx dataset.
'''
# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Step 2: Load the dataset
df = pd.read_excel("C:/7-Clustering/Telco_customer_churn.xlsx" )
df.head()

# Step 3: Preprocess the data
df.isnull().sum()
# Drop rows with missing values (if any)
df = df.dropna()

# Separate features into numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Step 4: Encode categorical variables using OneHotEncoding and standardize numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)])

# Step 5: Apply K-Means clustering using the preprocessed data
kmeans_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('kmeans', KMeans(n_clusters=3, random_state=42))])
kmeans_pipeline.fit(df)

# Predict clusters and add them to the dataset
df['Cluster'] = kmeans_pipeline['kmeans'].labels_

# Step 6: Analyze the clusters
print("Cluster Centers:\n", kmeans_pipeline['kmeans'].cluster_centers_)
print("Cluster Labels Distribution:\n", df['Cluster'].value_counts())

# Step 7: Visualize the Elbow Curve to find the optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(preprocessor.fit_transform(df))
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-', color='blue')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 8: Save the clustered data to a new Excel file
df.to_excel('Telco_Customer_Clustered.xlsx', index=False)


####################################################################

'''
5.Perform clustering on mixed data.Convert the categorical 
variables to numeric by using dummies or label encoding and 
perform normalization techniques. The dataset has the details 
of customers related to their auto insurance. Refer to 
Autoinsurance.csv dataset.
'''
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 2: Load the dataset
file_path = "C:/7-Clustering/Autoinsurence.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Step 3: Handle missing values (if any)
print(df.isnull().sum())  # Check for missing values
df = df.dropna()  # Remove missing values

# Step 4: Encode categorical variables
# If you want to use Label Encoding
label_encoder = LabelEncoder()

# Assuming 'Gender' and 'Policy_Type' are categorical columns (replace with actual column names)
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Policy_Type'] = label_encoder.fit_transform(df['Policy_Type'])

# Alternatively, you can use One-Hot Encoding for categorical variables
# transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), ['Gender', 'Policy_Type'])], remainder='passthrough')
# df_encoded = pd.DataFrame(transformer.fit_transform(df))

# Step 5: Normalize the numerical features
numerical_columns = ['Age', 'Annual_Premium', 'Policy_Years']  # Replace with actual numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Step 6: Apply K-Means Clustering
# Finding the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-', color='blue')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 7: Choose the optimal K (from the Elbow curve) and apply KMeans
optimal_k = 3  # Choose the optimal K based on the Elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Step 8: Analyze the clusters
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Cluster Labels Distribution:\n", df['Cluster'].value_counts())

# Step 9: Save the clustered data to a CSV file
df.to_csv('Autoinsurance_Clustered.csv', index=False)
print("Clustering completed. Results saved to 'Autoinsurance_Clustered.csv'.")
######################################





























