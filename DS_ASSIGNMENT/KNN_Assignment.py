# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:40:30 2025

@author: ACER
"""
'''    #######K-Nearest Neighbors#########
K-Nearest Neighbors (KNN) is a straightforward, non-parametric algorithm 
used for classification and regression tasks.
1)Business Objective:
--The primary goal of employing KNN in business is to enhance decision-making 
by accurately classifying or predicting outcomes based on the similarity to 
known data points.


2)Key Considerations:
--Data Types: Ensure each feature's data type is correctly identified 
to apply appropriate preprocessing steps.

--Relevance Assessment: Evaluate each feature's importance. Features 
like 'CustomerID' are identifiers and don't contribute to the predictive power of the model, so they can be excluded.

--Categorical Variables: Encode categorical features into numerical 
formats (e.g., one-hot encoding) for effective utilization in KNN models.

--Normalization: KNN relies on distance calculations; therefore, numerical 
features should be normalized or standardized to prevent features with larger ranges 
from dominating the distance metric.

3)Data Pre-processing
Data preprocessing is essential for K-Nearest Neighbors (KNN) to ensure 
accurate and efficient model performance. Key steps include:

*Data Cleaning:
--Handle Missing Values: Impute or remove missing data to maintain dataset 
integrity.
--Remove Duplicates: Eliminate duplicate records to prevent bias.
--Address Outliers: Identify and manage outliers that could skew results.

*Feature Engineering:
--Encode Categorical Variables: Convert categories into numerical values, 
such as through one-hot encoding, for model compatibility.
--Feature Scaling: Normalize or standardize numerical features to ensure 
equal weight in distance calculations, crucial for KNN performance.
--Dimensionality Reduction: Apply techniques like Principal Component 
Analysis (PCA) to reduce feature space dimensionality, enhancing efficiency.

4)Exploratory Data Analysis (EDA):
4.1 Summary
Summarize the dataset to grasp its structure and key characteristics:
--Descriptive Statistics: Calculate measures such as mean, median, mode, 
standard deviation, and range to understand central tendencies and variability.
--Data Structure: Review the number of observations, features, and data 
types to comprehend the dataset's composition.

4.2 Univariate Analysis
Examine individual variables to understand their distributions:
--Numerical Variables:Histograms: Visualize frequency distributions.
--Box Plots: Identify central values, dispersion, and outliers.

Categorical Variables:
--Bar Charts: Display category frequencies.
--Pie Charts: Show proportionate distributions.

4.3 Bivariate Analysis
Explore relationships between two variables to detect associations and 
correlations:
*Numerical vs. Numerical:
--Scatter Plots: Assess correlations and trends.
--Correlation Coefficients: Quantify the strength and direction of linear 
relationships.

*Categorical vs. Numerical:
--Box Plots: Compare distributions across categories.
--Violin Plots: Combine box plot and density plot features for detailed distribution insights.

Categorical vs. Categorical:
--Contingency Tables: Summarize frequency distributions across categories.
--Heatmaps: Visualize the intensity of relationships between categories.

5.Model Building
To build a K-Nearest Neighbors (KNN) model:
--Scale Data: Standardize features to ensure equal weight in distance 
calculations.
--Determine Optimal K: Use cross-validation to test various K values and 
select the one with the highest accuracy.
--Train and Test Model: Split data into training and testing sets, train
the KNN model with the optimal K, and evaluate performance using metrics 
like accuracy, precision, and recall.


6.Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
*Implementing a K-Nearest Neighbors (KNN) model offers several business benefits:
--Simplicity and Ease of Implementation: KNN is straightforward to understand 
and implement, requiring minimal coding expertise. 

--Versatility: Applicable to both classification and regression tasks, 
making it suitable for various business applications. 

--No Training Phase: KNN doesn't require a separate training phase, making
 it suitable for real-time applications. 

--Interpretability: The results are easy to interpret, allowing businesses
 to understand the rationale behind predictions. 
'''
###############################################################################
'''
1.A glass manufacturing plant uses different earth elements to design new 
glass materials based on customer requirements. For that, they would like 
to automate the process of classification as it’s a tedious job to 
manually classify them. Help the company achieve its objective by 
correctly classifying the glass type based on the other features using 
KNN algorithm.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("C:/DataScience_Dataset/Datasets/KNN Datasets/glass.csv")
print(df.head())
'''
O/P-->
 RI     Na    Mg    Al     Si     K    Ca   Ba   Fe  Type
0  1.52101  13.64  4.49  1.10  71.78  0.06  8.75  0.0  0.0     1
1  1.51761  13.89  3.60  1.36  72.73  0.48  7.83  0.0  0.0     1
2  1.51618  13.53  3.55  1.54  72.99  0.39  7.78  0.0  0.0     1
3  1.51766  13.21  3.69  1.29  72.61  0.57  8.22  0.0  0.0     1
4  1.51742  13.27  3.62  1.24  73.08  0.55  8.07  0.0  0.0     1
'''
print(df.info())
'''
O/P-->
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 214 entries, 0 to 213
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   RI      214 non-null    float64
 1   Na      214 non-null    float64
 2   Mg      214 non-null    float64
 3   Al      214 non-null    float64
 4   Si      214 non-null    float64
 5   K       214 non-null    float64
 6   Ca      214 non-null    float64
 7   Ba      214 non-null    float64
 8   Fe      214 non-null    float64
 9   Type    214 non-null    int64  
dtypes: float64(9), int64(1)
memory usage: 16.8 KB
None
'''
print(df.describe())
'''
O/P-->
RI          Na          Mg  ...          Ba          Fe        Type
count  214.000000  214.000000  214.000000  ...  214.000000  214.000000  214.000000
mean     1.518365   13.407850    2.684533  ...    0.175047    0.057009    2.780374
std      0.003037    0.816604    1.442408  ...    0.497219    0.097439    2.103739
min      1.511150   10.730000    0.000000  ...    0.000000    0.000000    1.000000
25%      1.516522   12.907500    2.115000  ...    0.000000    0.000000    1.000000
50%      1.517680   13.300000    3.480000  ...    0.000000    0.000000    2.000000
75%      1.519157   13.825000    3.600000  ...    0.000000    0.100000    3.000000
max      1.533930   17.380000    4.490000  ...    3.150000    0.510000    7.000000

[8 rows x 10 columns]
'''

X = df.drop('Type', axis=1)  
y = df['Type']               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train) #KNeighborsClassifier()

y_pred = knn.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
'''
O/P-->
[[10  1  0  0  0  0]
 [ 5  8  0  1  0  0]
 [ 1  2  0  0  0  0]
 [ 0  2  0  1  0  1]
 [ 0  0  0  0  3  0]
 [ 0  0  0  0  0  8]]
'''

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
'''
O/P-->
 precision    recall  f1-score   support

           1       0.62      0.91      0.74        11
           2       0.62      0.57      0.59        14
           3       0.00      0.00      0.00         3
           5       0.50      0.25      0.33         4
           6       1.00      1.00      1.00         3
           7       0.89      1.00      0.94         8

    accuracy                           0.70        43
   macro avg       0.60      0.62      0.60        43
weighted avg       0.64      0.70      0.66        43
'''

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
'''
O/P-->0.6976744186046512
'''
#############################################################################
'''
2.A National Zoopark in India is dealing with the problem of segregation 
of the animals based on the different attributes they have. Build a KNN 
model to automatically classify the animals. Explain any inferences you 
draw in the documentation.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("C:/DataScience_Dataset/Datasets/KNN Datasets/Zoo.csv")
print(df.head())
'''
O/P-->
animal name  hair  feathers  eggs  milk  ...  legs  tail  domestic  catsize  type
0    aardvark     1         0     0     1  ...     4     0         0        1     1
1    antelope     1         0     0     1  ...     4     1         0        1     1
2        bass     0         0     1     0  ...     0     1         0        0     4
3        bear     1         0     0     1  ...     4     0         0        1     1
4        boar     1         0     0     1  ...     4     1         0        1     1

[5 rows x 18 columns]
'''

print(df.info())
'''
O/P-->
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 101 entries, 0 to 100
Data columns (total 18 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   animal name  101 non-null    object
 1   hair         101 non-null    int64 
 2   feathers     101 non-null    int64 
 3   eggs         101 non-null    int64 
 4   milk         101 non-null    int64 
 5   airborne     101 non-null    int64 
 6   aquatic      101 non-null    int64 
 7   predator     101 non-null    int64 
 8   toothed      101 non-null    int64 
 9   backbone     101 non-null    int64 
 10  breathes     101 non-null    int64 
 11  venomous     101 non-null    int64 
 12  fins         101 non-null    int64 
 13  legs         101 non-null    int64 
 14  tail         101 non-null    int64 
 15  domestic     101 non-null    int64 
 16  catsize      101 non-null    int64 
 17  type         101 non-null    int64 
dtypes: int64(17), object(1)
memory usage: 14.3+ KB
None
'''

print(df.describe())
'''
O/P-->
 hair    feathers        eggs  ...    domestic     catsize        type
count  101.000000  101.000000  101.000000  ...  101.000000  101.000000  101.000000
mean     0.425743    0.198020    0.584158  ...    0.128713    0.435644    2.831683
std      0.496921    0.400495    0.495325  ...    0.336552    0.498314    2.102709
min      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
25%      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
50%      0.000000    0.000000    1.000000  ...    0.000000    0.000000    2.000000
75%      1.000000    0.000000    1.000000  ...    0.000000    1.000000    4.000000
max      1.000000    1.000000    1.000000  ...    1.000000    1.000000    7.000000

[8 rows x 17 columns]
'''

#Feature Selection: Separate features and target variable
X = df.drop(['animal name', 'type'], axis=1)  
y = df['type'] 

#Train-Test Split: Divide the data into training and testing sets.                            
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling: Standardize the feature values.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Initialize and Train the Model: Choose an appropriate value for K (e.g., K=5).
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train) #KNeighborsClassifier()

y_pred = knn.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
'''
[[12  0  0  0  0  0]
 [ 0  2  0  0  0  0]
 [ 0  0  0  1  0  0]
 [ 0  0  0  2  0  0]
 [ 0  0  0  0  3  0]
 [ 0  0  0  0  0  1]]
'''

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
'''
precision    recall  f1-score   support

           1       1.00      1.00      1.00        12
           2       1.00      1.00      1.00         2
           3       0.00      0.00      0.00         1
           4       0.67      1.00      0.80         2
           6       1.00      1.00      1.00         3
           7       1.00      1.00      1.00         1

    accuracy                           0.95        21
   macro avg       0.78      0.83      0.80        21
weighted avg       0.92      0.95      0.93        21
'''

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
#Accuracy Score:0.9523809523809523


















