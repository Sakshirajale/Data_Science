# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:15:09 2025

@author: ACER
"""
'''
1. Business Problem
1.1 Objective:
    -Predict outcomes like customer loan defaults, movie success, or employee salaries using decision-making models.
    -Provide insights for better classification and pattern identification.
1.2 Constraints:
    -Data Constraints: Handle categorical features, missing values, and outliers.
    -Model Constraints: Decision Trees risk overfitting; Random Forest can be computationally expensive.
    -Interpretability vs Accuracy: Decision Trees are simpler but less robust than Random Forests.
2. Modeling Steps
Data Preparation: Encode categorical features, handle missing data, and split data into training/testing sets.

Modeling:
 -Decision Tree: Prone to overfitting; hyperparameters like max_depth and criterion need tuning.
 -Random Forest: More accurate; tune n_estimators, max_features, and other parameters.

3. Evaluation
  -Use accuracy, confusion matrix, and F1-score to assess performance.
  -Compare clustering and predictions before and after dimensionality reduction using PCA.
  
2.Work on each feature of the dataset to create a data dictionary as displayed in the below image:
2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.
Here’s a concise Data Dictionary format to summarize the features for a Decision Tree or Random Forest model, based on common datasets like heart disease classification, movie 
classification, or salary prediction.

3.Data Pre-processing
3.1 Data Cleaning 
--Handle missing values: Impute with mean/median for numerical, mode for categorical.
--Remove irrelevant features like IDs or redundant columns.
3.2Feature Engineering
--Encode categorical variables using LabelEncoder or OneHotEncoding.
--Scale numerical features with StandardScaler or MinMaxScaler.
--Select important features using Recursive Feature Elimination (RFE) or feature importance from models.

4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
--Statistical Overview: Compute mean, median, min, max, standard deviation.
--Missing Values: Identify missing data using data.isnull().sum().
--Data Types: Check feature types using data.info()

4.2.Univariate analysis.
--Numerical Features: Use histograms and box plots to visualize distribution and outliers.
--Categorical Features: Use bar plots to visualize category frequency.

4.3.Bivariate analysis.
--Numerical-Numerical Relationships: Use scatter plots and correlation heatmaps.
--Categorical-Numerical Relationships: Use box plots to analyze value distribution by category.
--Categorical-Categorical Relationships: Use cross-tabulations and stacked bar charts.


5.	Model Building
5.1	Build the model on the scaled data (try multiple options).
--Split data into train and test sets using train_test_split.
--Scale data using StandardScaler if required.
--Experiment with models such as Decision Tree and Random Forest.

5.2	Perform Decision Tree and Random Forest on the given datasets.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Decision Tree Model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
--Use cross-validation with cross_val_score.
--Evaluate using accuracy_score, precision, recall, and confusion_matrix.

5.4	Briefly explain the model output in the documentation. 
--Decision Tree: Simple and interpretable but prone to overfitting.
--Random Forest: More accurate and robust due to reduced overfitting.
--Performance Comparison: Highlight metrics such as:
   *Accuracy improvement from Random Forest over Decision Tree.
   *Precision/recall trade-offs based on business requirements.

6. Benefits/Impact of the Solution
--Better Decision-Making: Identifies key factors for sales, risk, or customer behavior.
--Higher Accuracy: Random Forest improves predictions over simpler models.
--Risk Management: Early detection of high-risk customers minimizes losses.
--Efficiency: Automates decision-making with clear feature insights from Decision Trees.
'''
############################################################################
'''
Problem Statements:

1.	A cloth manufacturing company is interested to know about the different attributes contributing to high sales. Build a decision tree & random forest model with Sales as target variable (first convert it into categorical variable).
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/DataScience_Dataset/Datasets/DT_RF Datsets/Company_Data.csv")  

# Convert Sales to a categorical variable
threshold = df["Sales"].median()  
df["Sales_Category"] = np.where(df["Sales"] > threshold, "High", "Low")

# Drop original Sales column
df.drop("Sales", axis=1, inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
categorical_cols = ["ShelveLoc", "Urban", "US", "Sales_Category"]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Split the data into predictors and target variable
X = df.drop("Sales_Category", axis=1)
y = df["Sales_Category"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Model
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
decision_tree.fit(X_train, y_train)
'''
O/P-->
DecisionTreeClassifier(criterion='entropy', random_state=42)
'''

# Predictions using Decision Tree
dt_predictions = decision_tree.predict(X_test)

# Random Forest Model
random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)
'''
O/P-->
RandomForestClassifier(n_estimators=500, random_state=42)
'''

# Predictions using Random Forest
rf_predictions = random_forest.predict(X_test)


importances = pd.Series(random_forest.feature_importances_, index=X.columns)
print("\nFeature Importance from Random Forest:\n", importances.sort_values(ascending=False))
'''
O/P-->
Feature Importance from Random Forest:
 Price          0.259829
Age            0.149914
CompPrice      0.122139
ShelveLoc      0.100841
Income         0.099454
Population     0.095308
Advertising    0.090593
Education      0.054233
Urban          0.015777
US             0.011911
dtype: float64
'''
###################################################################################
'''
2.	 Divide the diabetes data into train and test datasets and build a Random Forest and Decision Tree model with Outcome as the output variable. 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset 
df = pd.read_csv("C:\DataScience_Dataset\Datasets\DT_RF Datsets\Diabetes.csv")  

# Splitting features and target variable
X = df.drop(" Class variable", axis=1)
y = df[" Class variable"]

# Split into train and test datasets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

############Decision Tree Model#########
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
decision_tree.fit(X_train, y_train) 
'''
O/P-->DecisionTreeClassifier(criterion='entropy', random_state=42)
'''

# Predictions for Decision Tree
dt_predictions = decision_tree.predict(X_test)

#############Random Forest Model ###########
random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
random_forest.fit(X_train, y_train)
'''
O/P-->
RandomForestClassifier(n_estimators=500, random_state=42)
'''

# Predictions for Random Forest
rf_predictions = random_forest.predict(X_test)

# Model Evaluation
print("Decision Tree Model Performance:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
#O/P-->Accuracy: 0.7272727272727273

print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
'''
O/P-->
Confusion Matrix:
 [[118  33]
 [ 30  50]]
'''

print("Classification Report:\n", classification_report(y_test, dt_predictions))
'''
O/P-->
Classification Report:
               precision    recall  f1-score   support

          NO       0.80      0.78      0.79       151
         YES       0.60      0.62      0.61        80

    accuracy                           0.73       231
   macro avg       0.70      0.70      0.70       231
weighted avg       0.73      0.73      0.73       231
'''

print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
'''
O/P-->Accuracy: 0.7532467532467533
'''


print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
'''
O/P-->
Confusion Matrix:
 [[123  28]
 [ 29  51]]
'''

print("Classification Report:\n", classification_report(y_test, rf_predictions))
'''
O/P-->
Classification Report:
               precision    recall  f1-score   support

          NO       0.81      0.81      0.81       151
         YES       0.65      0.64      0.64        80

    accuracy                           0.75       231
   macro avg       0.73      0.73      0.73       231
weighted avg       0.75      0.75      0.75       231
'''

# Feature Importance from Random Forest
importances = pd.Series(random_forest.feature_importances_, index=X.columns)
print("\nFeature Importances from Random Forest:\n", importances.sort_values(ascending=False))
'''
Feature Importances from Random Forest:
  Plasma glucose concentration    0.270482
 Body mass index                 0.165697
 Age (years)                     0.142781
 Diabetes pedigree function      0.115331
 Diastolic blood pressure        0.087556
 Number of times pregnant        0.077021
 2-Hour serum insulin            0.070935
 Triceps skin fold thickness     0.070197
dtype: float64
'''
####################################################################################
'''
3.	Build a Decision Tree & Random Forest model on the fraud data. Treat those who have taxable_income <= 30000 as Risky and others as Good (discretize the taxable 
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv("C:/DataScience_Dataset/Datasets/DT_RF Datsets/Fraud_check.csv")

# Display the first few rows
print(data.head())
'''
O/P-->
Undergrad Marital.Status  ...  Work.Experience  Urban
0        NO         Single  ...               10    YES
1       YES       Divorced  ...               18    YES
2        NO        Married  ...               30    YES
3       YES         Single  ...               15    YES
4        NO        Married  ...               28     NO

[5 rows x 6 columns]
'''

# Check for missing values
print(data.isnull().sum())
'''
O/P-->
Undergrad          0
Marital.Status     0
Taxable.Income     0
City.Population    0
Work.Experience    0
Urban              0
dtype: int64
'''

# Display data types of each column
print(data.dtypes)
'''
O/P-->

Undergrad          object
Marital.Status     object
Taxable.Income      int64
City.Population     int64
Work.Experience     int64
Urban              object
dtype: object
'''

# Create a new column 'Tax_Status' based on 'Taxable_Income'
data['Tax_Status'] = np.where(data['Taxable.Income'] <= 30000, 'Risky', 'Good')

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
data['Undergrad'] = le.fit_transform(data['Undergrad'])
data['Marital.Status'] = le.fit_transform(data['Marital.Status'])
data['Urban'] = le.fit_transform(data['Urban'])

# Encode the target variable
data['Tax_Status'] = le.fit_transform(data['Tax_Status'])

# Drop 'Taxable_Income' and define features (X) and target (y)
X = data.drop(columns=['Taxable.Income', 'Tax_Status'])
y = data['Tax_Status']

# Split the data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)
#O/P-->DecisionTreeClassifier(criterion='entropy', random_state=42)


# Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
#O/P-->Decision Tree Accuracy: 66.11%

print("Confusion Matrix:\n", conf_matrix_dt)
'''
O/P-->
Confusion Matrix:
 [[110  33]
 [ 28   9]]
'''

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
#o/p-->RandomForestClassifier(n_estimators=500, random_state=42)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
#o/p-->Random Forest Accuracy: 75.00%
print("Confusion Matrix:\n", conf_matrix_rf)
'''
O/P-->
Confusion Matrix:
 [[135   8]
 [ 37   0]]
'''

################################################################################
'''
4.	In the recruitment domain, HR faces the challenge of predicting if the candidate is faking their salary or not. For example, a candidate claims to have 5 years of experience and earns 70,000 per month working as a regional manager. The candidate expects more money than his previous CTC. We need a way to verify their claims (is 70,000 a month working as a regional manager with an experience of 5 years a genuine claim or does he/she make less than that?) Build a Decision Tree and Random Forest model with monthly income as the target variable. 
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("C:/DataScience_Dataset/Datasets/DT_RF Datsets/HR_DT.csv")

# Display the first few rows
print(data.head())
'''
o/p-->
 Position of the employee  ...   monthly income of employee
0         Business Analyst  ...                        39343
1        Junior Consultant  ...                        46205
2        Senior Consultant  ...                        37731
3                  Manager  ...                        43525
4          Country Manager  ...                        39891

[5 rows x 3 columns]
'''

# Check for missing values
print(data.isnull().sum())
'''
o/p-->
Position of the employee                 0
no of Years of Experience of employee    0
 monthly income of employee              0
dtype: int64
'''

# Display data types of each column
print(data.dtypes)
'''
Position of the employee                  object
no of Years of Experience of employee    float64
 monthly income of employee                int64
dtype: object
'''

# Initialize LabelEncoder
le = LabelEncoder()

# Encode 'Job_Title' and any other categorical columns
data['Position of the employee'] = le.fit_transform(data['Position of the employee'])

# Repeat encoding for other categorical features if necessary

# Define features (X) and target (y)
X = data.drop(columns=[' monthly income of employee'])  # Features
y = data[' monthly income of employee']                 # Target variable

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
#o/p-->DecisionTreeRegressor(random_state=42)


# Predict on test data
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree MAE: {mae_dt}")
#o/p-->Decision Tree MAE: 1330.1875
print(f"Decision Tree MSE: {mse_dt}")
#o/p-->Decision Tree MSE: 12051515.50625
print(f"Decision Tree R^2 Score: {r2_dt}")
#o/p-->Decision Tree R^2 Score: 0.981709118123554


# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
#o/p-->RandomForestRegressor(n_estimators=500, random_state=42)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MAE: {mae_rf}")
#o/p-->Random Forest MAE: 1715.2789147619053
print(f"Random Forest MSE: {mse_rf}")
#o/p-->Random Forest MSE: 11454061.529712578
print(f"Random Forest R^2 Score: {r2_rf}")
#o/p-->Random Forest R^2 Score: 0.9826158887372408
###############################################################











