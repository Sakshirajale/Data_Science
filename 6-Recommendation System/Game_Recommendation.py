
# Game Recommendations
# Import necessary libraries
import pandas as pd
# Pandas is a library used for data manipulation and analysis
# Useful for handling tabular data like csv file
from sklearn.metrics.pairwise import cosine_similarity
# Cosine similarity is a measure of how similar two vectors are.
# Used in recommendation systems to compare user preferences
import numpy as np
# NumPy is a library for numerical computing in Python.
# It provides support for arrays and various mathematical functions

# Load the CSV file
file_path = "C:/6-Recommendation System/game.csv.xls"
data = pd.read_csv(file_path)
data.head()
'''
o/p-->
userId                                  game  rating
0       3  The Legend of Zelda: Ocarina of Time     4.0
1       6              Tony Hawk's Pro Skater 2     5.0
2       8                   Grand Theft Auto IV     4.0
3      10                           SoulCalibur     4.0
4      11                   Grand Theft Auto IV     4.5
'''
# This reads the CSV file from the specified path into a Pandas DataFrame. 
# The DataFrame data holds the tabular data from the CSV.

# Step-1: Create a user item matrix (rows: users, columns: games)
user_item_matrix = data.pivot_table(index = "userId", columns = "game", values = "rating")
print(user_item_matrix)
'''
o/p-->
game    'Splosion Man  ...  page not found
userId                 ...                
1                 NaN  ...             NaN
2                 NaN  ...             NaN
3                 NaN  ...             NaN
5                 NaN  ...             NaN
6                 NaN  ...             NaN
              ...  ...             ...
7110              NaN  ...             NaN
7116              NaN  ...             NaN
7117              NaN  ...             NaN
7119              NaN  ...             NaN
7120              NaN  ...             NaN

[3261 rows x 3438 columns]
'''
'''
pivot_table: This function reshapes the DataFrame into a matrix where:
    Each row represents a user (identified by userId),
    Each column represents a game (identified by game),
    The values in the matrix represent the ratings that user gave to the games'''

# Step-2: Fill NaN values with 0 (assuming no rating means the game has not)
user_item_matrix_filled = user_item_matrix.fillna(0)
print(user_item_matrix_filled)
'''
o/p-->
game    'Splosion Man  ...  page not found
userId                 ...                
1                 0.0  ...             0.0
2                 0.0  ...             0.0
3                 0.0  ...             0.0
5                 0.0  ...             0.0
6                 0.0  ...             0.0
              ...  ...             ...
7110              0.0  ...             0.0
7116              0.0  ...             0.0
7117              0.0  ...             0.0
7119              0.0  ...             0.0
7120              0.0  ...             0.0

[3261 rows x 3438 columns]
'''
'''
This line replaces any missing values (NaNs)
in the user-item matrix with 0,
indicating that the user did not rate that particular game'''

# Step-3: Compute the cosine similarity between user based on raw ratings
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert similarity matrix to a DataFrame for easy reference
user_similarity_df = pd.DataFrame(user_similarity,index = user_item_matrix.index, columns = user_item_matrix.index)

# Step-4: Function to get game recommendations for a specific user based on
def get_collaborative_recommendations_for_user(user_id, num_recommendations=5):
    
    # Get the similarity scores for the input users for all other users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Get the most similar users (excluding the user themselves)
    similar_users = similar_users.drop(user_id)
    
    # Select the top N similar users to limit noise (e.g. top 50 users)
    top_similar_users = similar_users.head(50)
    # This selects the top 50 most similar users to limit noise in the recommendations
    # Get rating of this similar users, weighted by their similarity scores
    weighted_ratings = np.dot(top_similar_users.values, user_item_matrix_filled.loc[top_similar_users.index])
    
    # np.dot: This computes the dot product between the similarity scores
    # of the top similar users and their corresponding ratings in the
    # user-item matrix.
    # The result is an array of weighted ratings for each game.
    
    # Normalize by the sum of similarities
    sum_of_similarities = top_similar_users.sum()
    
    if sum_of_similarities > 0:
        weighted_ratings /= sum_of_similarities
        
        
        
        # The weighted ratings are normalized by dividing by the sum of
        # similarities to avoid biasing toward users with higher ratings.
        
        
    # Recommend games that the user hasn't rated yet
    user_ratings = user_item_matrix_filled.loc[user_id]
    unrated_games = user_ratings[user_ratings == 0 ]
    
    # identifies games that the target user has not rated (i.e., rated 0)
    
    # Get the weighted scores for unrated games
    game_recommendations = pd.Series(weighted_ratings, index = user_item_matrix_filled.columns).loc[unrated_games.index]
    
    # This creates a pandas Series from the weighted ratings and filters
    # it to include only the unrated games.
    # Finally, it sorts the recommendations in descending order and returs  the top specified number of recommendations.
    
    # Return the top 'num_recommendations' game recommendations
    return game_recommendations.sort_values(ascending = False).head(num_recommendations)

# Example usage: Get recommendations for a user with ID 3
recommended_games = get_collaborative_recommendations_for_user(user_id=3)

# Print the recommended games
print("Recommended games for user 3: ")
print(recommended_games)
'''
o/p-->
Recommended games for user 3:
'Splosion Man         0.0
Resogun               0.0
Resogun: Heroes       0.0
Retro City Rampage    0.0
Retro/Grade           0.0
dtype: float64
'''
