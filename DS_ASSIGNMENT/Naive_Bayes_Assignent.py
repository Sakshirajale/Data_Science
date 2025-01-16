# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:12:33 2025

@author: ACER
"""

'''
Problem Statement:
1.) Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train and test datasets are given separately. Use both for model building. 
'''
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("C:/DataScience_Dataset/Datasets/NaiveBayes Datasets/SalaryData_Train.csv")
test_data = pd.read_csv("C:/DataScience_Dataset/Datasets/NaiveBayes Datasets/SalaryData_Test.csv")

train_data = train_data.dropna()
test_data = test_data.dropna()

label_encoders = {}
categorical_columns = train_data.select_dtypes(include=['object']).columns

for column in categorical_columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])
    label_encoders[column] = le

X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']
X_test = test_data.drop('Salary', axis=1)
y_test = test_data['Salary']

model = GaussianNB()
model.fit(X_train, y_train) # GaussianNB()

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Training Accuracy: {train_accuracy}')
#o/p-->Training Accuracy: 0.7953317197705646

test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Testing Accuracy: {test_accuracy}')
#o/p-->Testing Accuracy: 0.7946879150066402

cm = confusion_matrix(y_test, y_pred_test)
print('Confusion Matrix:',cm)
'''
Confusion Matrix: [[10759   601]
                   [ 2491  1209]]
'''
##################################################################
'''
Problem Statement: -
2)This dataset contains information of users in a social network. This social 
network has several business clients which can post ads on it. One of the 
clients has a car company which has just launched a luxury SUV for a ridiculous 
price. Build a Bernoulli Naïve Bayes model using this dataset and classify 
which of the users of the social network are going to purchase this luxury SUV. 
1 implies that there was a purchase and 0 implies there wasn’t a purchase.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv("C:/DataScience_Dataset/Datasets/NaiveBayes Datasets/NB_Car_Ad.csv")
print(dataset.head())
'''
O/P-->
User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0
'''

X = dataset.iloc[:, [2, 3]].values  # Selecting 'Age' and 'EstimatedSalary'
y = dataset.iloc[:, 4].values       # Selecting 'Purchased'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train_bin = np.where(X_train > 0, 1, 0)
X_test_bin = np.where(X_test > 0, 1, 0)

classifier = BernoulliNB()
classifier.fit(X_train_bin, y_train) #BernoulliNB()

y_pred = classifier.predict(X_test_bin)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')
'''
O/P-->Confusion Matrix:
[[63  5]
 [16 16]]
'''

print(f'Accuracy: {accuracy:.2f}')
#o/p-->Accuracy: 0.79

def plot_decision_boundary(X, y, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap([(1, 0, 0), (0, 1, 0)]))  # Red and Green in RGB
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=[(1, 0, 0), (0, 1, 0)][i], label=j)  # Red and Green in RGB
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


plot_decision_boundary(X_train_bin, y_train, 'Bernoulli Naive Bayes (Training set)')
plot_decision_boundary(X_test_bin, y_test, 'Bernoulli Naive Bayes (Test set)')
#################################################################################
'''
3)Problem Statement: -
In this case study, you have been given Twitter data collected from an anonymous twitter handle. With the help of a Naïve Bayes model, predict if a given tweet about a real disaster is real or fake.
1 = real tweet and 0 = fake tweet
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

dataset = pd.read_csv("C:/DataScience_Dataset/Datasets/NaiveBayes Datasets\Disaster_tweets_NB.csv")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

dataset['cleaned_text'] = dataset['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset['cleaned_text'])
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train) #MultinomialNB()

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
#o/p-->Accuracy: 0.79
print('Confusion Matrix:')
print(cm)
'''
O/P-->
[[795  79]
 [235 414]]
'''

print('Classification Report:')
print(report)
'''
O/P-->
precision    recall  f1-score   support

           0       0.77      0.91      0.84       874
           1       0.84      0.64      0.73       649

    accuracy                           0.79      1523
   macro avg       0.81      0.77      0.78      1523
weighted avg       0.80      0.79      0.79      1523
'''
######################################################################################

