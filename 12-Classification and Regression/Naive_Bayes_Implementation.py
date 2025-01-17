# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:30:48 2024

@author: ACER
"""
#####17-10-2024

#trainining accuracy lower and test accuracy higher -> overfitting
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
email_data = pd.read_csv("C:/11-Classification and Regression/sms_raw_NB.csv",encoding ="ISO-8859-1")

###Theses sms are in text form , open the data frame and there are ham or spam
#cleaning data
#the function tokenizes the text and removes words
#with fewer than 4 characters

import re
def cleaning_text(i):
    i = re.sub("[^A-Z a-z""]+"," ",i).lower()
    w=[]
    #every thing else A to Z and a to z is going to convert to space and 
    #we will take each row and tokenize
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return(" ".join(w))
  
#Testing above function wirth sample text
    cleaning_text("Hope you are having good week.just checking")
    cleaning_text("hope i can i understand your feeling12321.hi how are you")
    cleaning_text("Hi how are you")
    
# Note the dataframe size is 5559 , 2, now removing empty spaces
# removing empty rows
email_data = email_data.loc[email_data.text != " ",:]
email_data.shape

#you can use count vectorizer which directly converts a collection of documents
# first we will split data
from sklearn.model_selection import train_test_split
email_train, email_test=train_test_split(email_data,test_size=0.2)

#splits each email into a list of words
## creating matrix of token count for entire dataframe

def split_into_words(i):
    return[word for word in i.split(" ")]

# defining the preparation of email text into word count matrix format
#CountVectorizer : Converts the emails into a matrix of token counts
# .fit(): Learns the vocabulary from the text data
# .transform() : Converts text data into a token count matrix

emails_bow = CountVectorizer(analyzer = split_into_words).fit(email_data.text)
#defing Bow for all data frames
all_emails_matrix = emails_bow.transform(email_data.text)
train_email_matrix = emails_bow.transform(email_train.text)
#for testing messages
test_email_matrix = emails_bow.transform(email_test.text)
#Learning term weighting and normalizing entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
#preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_email_matrix)
train_tfidf.shape

test_tfidf  = tfidf_transformer.transform(test_email_matrix)
test_tfidf.shape

###### Now apply to naive bayes

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb =MB()
classifier_mb.fit(train_tfidf,email_train.type)
#email_train.type: this is the column in the training datset
# (email_train) that contains the target labels , # 
#which specify whether each message is spam or ham (non-spam)
# The .type attributes referes to that specific column
# in the email_train dataframe
# in the email_train dataframe
###training data prepared in terms of tfidf and # training data prepared in terms of tfidf and 
# label of corresponding training data


#evaluation on test data
test_pred_m = classifier_mb.predict(test_tfidf)

##calculating accuracy
accuracy_test_m = np.mean(test_pred_m ==email_test.type)
accuracy_test_m
#Evaluation on Test Data accuracy matrix
from sklearn.metrics import accuracy_score

accuracy_score(test_pred_m,email_test.type)
pd.crosstab(test_pred_m,email_test.type)
#Training data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == email_train.type)
accuracy_train_m
#Test datat (with laplace)
classifier_mb_lap = MB(alpha =3 )
classifier_mb_lap.fit(train_tfidf,email_train.type)

#Accuracy  After Tunning
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == email_test.type)
accuracy_test_lap
accuracy_score(test_pred_lap,email_test.type)

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap,email_test.type)
pd.crosstab(test_pred_lap,email_test.type)

#Trainig Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap=np.mean(train_pred_lap == email_train.type)
accuracy_train_lap
#############################################################################
###### 18-10-2024 #####
'''
Problem Statement:-
This dataset contains information of users in a social network.
This social network has several business clients which can post ads
on it.
One of the clients has a car company which has just luanched a luxury SUV 
for a ridiculous price. Build a Bernoulli Naive Bayes model using this dataset and
classify which of the users of social network are going to purchase this Luxury
SUV. 1 implies that there was a purchase and 0 implies there wasn't a puchase.

1. Business Problem
 1.1. What is the Business objectives?
      1.1.1. This will help you bring those audiences to your website
             who are interested in cars. And, there will be many of those
             who are planning to buy a car in the near future.
             
             
      1.1.2 Communicating withyour with your target audience over social media
            is always a great way to build a good market reputation.
            Try responding to peoples automobile related queries on Twitter
            and Facebook. Pages quickly to be their choice when it comes to buying
            a car.


  1.2 Are there any constraints?
      Not having a clear marketing or social media strategy may result
      in reduced benefits for your business.
      
      Additional resource may be needed to manage your online presence.
      
      Social media is immediate and needs daily monitoring
      
      If you don't actively manage your social media presence,
      you may not see any real benefits.
      
      There is a risk of unwanted or inappropriate behavior on your site
      including bullying and harassment
      
      Greater exposure online has the potential to attract risks.
      Risks can include negative feedback information, Leaks or hacking

'''

'''
#DATA DICTINARY
2.Work on each feature of the dataset to create a data dictinary 
user ID :Integer type which is not contributory
Gender: Object Type need to be label encoding
Age: Integer
EstimatedSalary:Integer
Purchased: Integer Type
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Load dataset
car = pd.read_csv("C:/12-Classification and Regression/NB_Car_Ad.csv")
car

# EDA
print(car.describe())
car.isna().sum()

car.drop(['User ID'], axis=1, inplace=True)

plt.hist(car.Age)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.hist(car.EstimatedSalary)
plt.title('Estimated Salary Distribution')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()

# Data Preprocessing
label_encoder = preprocessing.LabelEncoder()
car['Gender'] = label_encoder.fit_transform(car['Gender'])

# Separate the target variable (assuming 'Purchased' is the target column)
target_column = 'Purchased'
X = car.drop(columns=[target_column])  # Input features (Age, EstimatedSalary, Gender)
y = car[target_column]  # Target variable (Purchased)

# Normalize the input features (excluding target)
def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())

X_norm = norm_func(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

# Naive Bayes Model (BernoulliNB)
classifier_bb = BernoulliNB()
classifier_bb.fit(X_train, y_train)

# Predictions
y_pred_b = classifier_bb.predict(X_test)

# Accuracy
accuracy_test_b = np.mean(y_pred_b == y_test)
print("Accuracy:", accuracy_test_b)
