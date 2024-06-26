# MovieLens Recommendation Engine

## Project Overview
This project involves building a movie recommendation engine using collaborative filtering and content-based filtering techniques. The goal is to recommend movies to users based on their rating history and movie metadata.

## Features
- Collaborative Filtering: Recommends movies based on user-item interactions using Singular Value Decomposition (SVD).
- Content-Based Filtering: Recommends movies based on movie metadata (genres) using TF-IDF and cosine similarity.
- Evaluation Metrics: Precision@K, Recall@K, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- Streamlit Web App for user interaction.

## Directory Structure
```
movielens-recommendation-engine/
│
├── data/
│   ├ml-latest-small/
│   ├── processed/
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   ├── content_based_filtering.py
│   ├── utils/
│   │   ├── evaluation.py
│
├── app/
│   ├── app.py
│
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started
### Prerequisites
- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy

### Installation
1. Navigate to the project directory:
    ```bash
    cd movielens-recommendation-engine
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Preprocess the data:
    ```bash
    python src/data/data_preprocessor.py
    ```
2. Run the models:
    ```bash
    python src/models/collaborative_filtering.py
    python src/models/content_based_filtering.py
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app/app.py
    ```

## Evaluation
The recommendation engine is evaluated using the following metrics:
- **Precision@K**: The proportion of recommended items in the top-K set that are relevant.
- **Recall@K**: The proportion of relevant items found in the top-K recommendations.
- **Mean Absolute Error (MAE)**: The average absolute difference between predicted ratings and true ratings.
- **Root Mean Squared Error (RMSE)**: The square root of the average squared differences between predicted ratings and true ratings.

### Evaluation Example
To evaluate the collaborative filtering model, run the Streamlit app and interact with the recommendations. The app will display evaluation metrics based on the selected movies.
