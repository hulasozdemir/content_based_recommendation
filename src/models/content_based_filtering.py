import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedFiltering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.items = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}

    def fit(self, items):
        self.items = items
        self.tfidf_matrix = self.vectorizer.fit_transform(items['genres'])
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(items.index)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}

    def recommend(self, item_id, n_recommendations=10):
        # Ensure item_id is within the valid range
        if item_id not in self.movie_id_to_idx:
            raise ValueError(f"Item ID {item_id} is not in the dataset.")
        
        idx = self.movie_id_to_idx[item_id]
        cosine_similarities = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]  # Exclude the item itself
        similar_movie_ids = [self.idx_to_movie_id[i] for i in similar_indices]
        return similar_movie_ids

if __name__ == "__main__":
    movies_path = '../data/processed/cleaned_movies.csv'
    df = pd.read_csv(movies_path)
    items = df.set_index('movieId')
    model = ContentBasedFiltering()
    model.fit(items)
    recommendations = model.recommend(item_id=1, n_recommendations=10)
    print("Recommended items:", recommendations)

