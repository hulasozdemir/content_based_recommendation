from sklearn.decomposition import TruncatedSVD
import numpy as np

class CollaborativeFiltering:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None

    def fit(self, user_item_matrix):
        n_features = user_item_matrix.shape[1]
        n_components = min(self.n_components, n_features)
        self.svd = TruncatedSVD(n_components=n_components)
        self.user_item_matrix = user_item_matrix
        self.svd.fit(user_item_matrix)
        self.user_factors = self.svd.transform(user_item_matrix)
        self.item_factors = self.svd.components_.T

    def recommend(self, user_id, n_recommendations=10):
        user_vector = self.user_factors[user_id]
        scores = np.dot(self.item_factors, user_vector)
        top_items = np.argsort(scores)[-n_recommendations:][::-1]
        return top_items

if __name__ == "__main__":
    import pandas as pd
    ratings_path = '../data/processed/cleaned_ratings.csv'
    df = pd.read_csv(ratings_path)
    user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0).values
    model = CollaborativeFiltering()
    model.fit(user_item_matrix)
    recommendations = model.recommend(user_id=0, n_recommendations=10)
    print("Recommended items:", recommendations)
