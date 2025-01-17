# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV file
file_path = "C:/6-Recommendation System/Entertainment.csv.xls"
data = pd.read_csv(file_path)

# Step-1: Preprocess the 'Category' column using TFF-IDF
tfidf = TfidfVectorizer(stop_words = 'english') #Remove common stop_words
tfidf_matrix = tfidf.fit_transform(data['Category']) 
# Fit and transform the category data

# Step-2: Compute the cosine similarity between titles
cosine_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step-3: Create a function to recommend titles based on similarity
def get_recommendations(title,cosine_sim = cosine_sim):
    # Get the index of the title that matches the input title
    idx = data[data['Titles']==title].index[0]
    '''
    data['Titles']==title
    This creates a boolean mask (a series of True and False values)
    indicating which rows in the Titles column
    match the input title.
    '''
    
    # Get the pairwise similarity scores of all titles with that title
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the titles based on the similarity scores in descending order
    sim_scores = sorted(sim_scores,key = lambda x: x[1],reverse = True)
    
    # Get the indices of most similar titles
    sim_indices = [i[0] for i in sim_scores[1:6]]
    
    # Return the top 5 most similar titles
    return data['Titles'].iloc[sim_indices]

# Test the recommendation system with an example title
example_title = "Toy Story (1995)"
recommended_titles = get_recommendations(example_title)

# Print the recommendations
print("Recommended Entertainment for user 3: ")
print(recommended_titles)
'''
Recommended Entertainment for user 3: 
25                        Othello (1995)
16          Sense and Sensibility (1995)
11    Dracula: Dead and Loving It (1995)
10        American President, The (1995)
45          When Night Is Falling (1995)
Name: Titles, dtype: object
'''