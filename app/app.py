import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based_filtering import ContentBasedFiltering
import pandas as pd
import streamlit as st

# Load data
ratings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_ratings.csv'))
movies_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_movies.csv'))
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Prepare user-item matrix for collaborative filtering
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).values

# Initialize models
collab_model = CollaborativeFiltering()
collab_model.fit(user_item_matrix)

content_model = ContentBasedFiltering()
content_model.fit(movies.set_index('movieId'))

# Function to get movie titles from movie IDs
def get_movie_titles(movie_ids):
    return movies[movies['movieId'].isin(movie_ids)]['title'].values

# Streamlit UI
st.title('MovieLens Recommendation Engine')

# Create a multiselect list for movie titles
movie_titles = movies['title'].tolist()
selected_movie_titles = st.multiselect('Select Movies for Content-Based Recommendation:', movie_titles)

# Get the movie IDs corresponding to the selected titles
selected_movie_ids = movies[movies['title'].isin(selected_movie_titles)]['movieId'].values

# Number of recommendations slider
n_recommendations = st.slider('Number of Recommendations:', min_value=1, max_value=20, step=1)

if st.button('Recommend Content-Based'):
    if len(selected_movie_ids) > 0:
        recommended_ids = []
        for movie_id in selected_movie_ids:
            try:
                recommendations = content_model.recommend(movie_id, n_recommendations)
                st.write(f"Recommendations for Movie ID {movie_id}: {recommendations}")
                recommended_ids.extend(recommendations)
            except ValueError as e:
                st.write(f"Error: {e}")
        # Remove duplicates and the selected movies from the recommendations
        recommended_ids = list(set(recommended_ids) - set(selected_movie_ids))
        # Limit to n_recommendations
        recommended_ids = recommended_ids[:n_recommendations]
        recommended_titles = get_movie_titles(recommended_ids)
        st.write('Recommended Movies:', recommended_titles)
    else:
        st.write('Please select at least one movie.')
