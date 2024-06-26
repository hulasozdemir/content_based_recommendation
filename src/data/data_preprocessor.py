import pandas as pd

class DataPreprocessor:
    def __init__(self, ratings_path, movies_path):
        self.ratings_path = ratings_path
        self.movies_path = movies_path

    def load_data(self):
        ratings = pd.read_csv(self.ratings_path)
        movies = pd.read_csv(self.movies_path)
        return ratings, movies

    def clean_data(self, df):
        # Remove duplicates
        df = df.drop_duplicates()
        # Handle missing values
        df = df.fillna(method='ffill')
        return df

    def preprocess(self):
        ratings, movies = self.load_data()
        ratings = self.clean_data(ratings)
        movies = self.clean_data(movies)
        return ratings, movies

if __name__ == "__main__":
    ratings_path = 'data/ml-latest-small/ratings.csv'
    movies_path = 'data/ml-latest-small/movies.csv'
    preprocessor = DataPreprocessor(ratings_path, movies_path)
    ratings, movies = preprocessor.preprocess()
    ratings.to_csv('data/processed/cleaned_ratings.csv', index=False)
    movies.to_csv('data/processed/cleaned_movies.csv', index=False)

