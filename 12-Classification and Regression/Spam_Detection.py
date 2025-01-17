# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced data visualization

# Load the dataset
Data = pd.read_csv("C:/12-Classification and Regression/sms_raw_NB.csv")

# Display the first few rows of the dataset
print(Data.head())

# Get a concise summary of the dataset (column info, non-null counts, data types)
print(Data.info())

# Display the column names in the dataset
print(Data.columns)

# Analyze the distribution of target classes (spam vs. non-spam emails)
print(Data['label'].value_counts())

# Plot the class distribution (spam vs. non-spam emails)
sns.countplot(x='label', data=Data)
plt.title("Class Distribution")  # Add title to the plot
plt.show()

# Import NLP libraries for text preprocessing
import nltk
from nltk.corpus import stopwords  # To remove common words like 'is', 'the', etc.
from nltk.tokenize import word_tokenize  # To split text into words (tokens)
from nltk.stem import WordNetLemmatizer  # To reduce words to their base form

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for preprocessing text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text into individual words
    tokens = word_tokenize(text)
    # Remove non-alphanumeric tokens (punctuation, special characters, etc.)
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords (e.g., "is", "and", etc.)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize tokens to reduce them to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the text column
Data['processed_text'] = Data['text'].apply(preprocess_text)

# Convert processed text into numerical features using TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Limit features to the top 3000 for simplicity
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(Data['processed_text']).toarray()  # Convert text to numerical form
y = Data['label']  # Target column containing labels (spam or non-spam)

# Display the TF-IDF feature array and target labels
print(X)
print(y)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes model for spam detection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the confusion matrix as a heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")  # Add title to the heatmap
plt.show()

# Generate Word Clouds for spam and non-spam emails
from wordcloud import WordCloud

# Combine all spam email text into one string
spam_words = ' '.join(Data[Data['label'] == 1]['processed_text'])  # Adjust based on spam label value
non_spam_words = ' '.join(Data[Data['label'] == 0]['processed_text'])  # Adjust based on non-spam label value

# Create word clouds for spam and non-spam text
spam_wc = WordCloud(width=800, height=400).generate(spam_words)
non_spam_wc = WordCloud(width=800, height=400).generate(non_spam_words)

# Display the Spam Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.title("Spam Word Cloud")  # Add title to the Word Cloud
plt.axis('off')  # Remove axes for better visualization
plt.show()

# Combine all spam email text for further validation
spam_text = Data[Data['label'] == 'spam']['processed_text'].str.cat(sep=' ')

# Validate if spam_text is empty
if not spam_text:
    print("No spam words found.")
else:
    print(f"Spam text length: {len(spam_text)}")

# Create and display Word Cloud for all spam emails
spam_wc = WordCloud(width=800, height=400).generate(spam_text)
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')  # Remove axes for better visualization
plt.show()
