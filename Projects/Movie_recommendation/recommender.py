'''
import numpy as np
import pandas as pd
from typing import List, Tuple
import os

def load_movies_csv(path: str) -> Tuple[np.ndarray, list[str]]:
    #1. Load the csv into pandas data frame
    df = pd.read_csv(path)

    #2. Identify which columns are features(numbers) to calculate the score
    #we need to identify which columns are features from the CSV
    # apart from movie_id and movie title remaining all should be considered as features

    feature_cols = [col for col in df.columns if col not in ('movie_id', 'title')]
    print(f"featured columns:  '{feature_cols}' " )

    #3. Create Numpy matrix for calculations
    # to_numpy() is converts pandas data to high efficient numpy array
    movie_matrix = df[feature_cols].to_numpy(dtype=float)

    #4. Get our list of movie titles
    movie_titles = df['title'].tolist()

    return movie_matrix, movie_titles

def L2_normalized_rows(mat: np.ndarray) -> np.ndarray:
    # axis=1 tells norm to calculate the length of each row.
    # # keepdims=True keeps the output as a column, so we can divide correctly.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    #print(norms)
    norms[norms == 0] = 1.0
    return mat/norms
'''
'''

matrix , titles = load_movies_csv(path="movies.csv")
print("Movie matrix: \n", matrix)
print("Titles: \n", titles)

Normalized_matrix = l2_norms(mat=matrix)
print("Normalized_matrix: \n", Normalized_matrix)

user_matrix = np.array([
    [9, 8, 2, 1, 0],   # User 1: loves Action + SciFi
    [2, 1, 9, 6, 5]    # User 2: loves Comedy + Romance
])
  # shape (1, num_features)
'''


import numpy as np
import pandas as pd
from typing import List, Tuple
import os
import logging
import yaml
# Logging set up

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


#configuration

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# load the file

CONFIG = load_config()

# Logging set up
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"].upper(), logging.INFO),
    format=CONFIG["logging"]["format"]
)
'''
CONFIG = {
    "movies_path" : "movies.csv",
    "top_k" : 2,
    "metric" : "cosine"
}
'''



# Core functions
def load_movies_csv(path:str) -> Tuple[np.ndarray, list[str]]:
    #logging info
    logging.info(f"Loading movie data from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movie data file not found at path: {path}")

    df = pd.read_csv(path)
    #print(df)

    # RELIABILITY: Check if essential columns exist

    required_cols = {'title'}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns. Must contain 'title'.")

    feature_columns = [col for col in df.columns if col not in ('movie_id', 'title')]

    if not feature_columns:
        raise ValueError("No feature columns found in the CSV.")

    #print(feature_columns)

    movie_matrix = df[feature_columns].to_numpy(dtype=float)

    titles = df['title'].to_list()
    feature_count = len(feature_columns)

    logging.info(f"Loaded {len(titles)} movies with {movie_matrix.shape[1]} features.")
    logging.debug(f"Movie matrix shape: {movie_matrix.shape}")

    return movie_matrix, titles, feature_count


def l2_norms(mat: np.ndarray) -> np.ndarray:

    norms = np.linalg.norm(mat, axis=1, keepdims=True)

    #print("norms \n", norms)

    norms[norms==0] = 1.0

    return mat/norms

def recommend_batch(user_matrix: np.ndarray, movies_matrix: np.ndarray, titles: list[str], k: int, metric: str= 'cosine'):
    #Input validation 
    logging.info(f"Generating top {k} recommendations for {user_matrix.shape[0]} users using '{metric}' metric.")

    # RELIABILITY: Validate all inputs before processing
    if user_matrix.shape[1] != movies_matrix.shape[1]:
        raise ValueError(f"Feature mismatch: User vector has {user_matrix.shape[1]} features, movies have {movies_matrix.shape[1]}.")
    if k <= 0:
        raise ValueError("k (number of recommendations) must be a positive integer.")
    if k > len(titles):
        logging.warning(f"k ({k}) is larger than the number of movies ({len(titles)}). Setting k to {len(titles)}.")
        k = len(titles)
    if metric not in ['cosine', 'dot']:
        raise ValueError(f"Unsupported metric '{metric}'. Choose 'cosine' or 'dot'.")

    if metric == 'cosine':
        user_norm = l2_norms(user_matrix)
        movies_norm = l2_norms(movies_matrix)

        scores_matrix = user_norm @ movies_norm.T

    else: #dot product
        scores_matrix = user_matrix @ movies_matrix.T 
    
    logging.debug(f"Calculated scores matrix with shape: {scores_matrix.shape}")

    #print("Scores matrix:\n", scores_matrix)

    # --- Top-K Ranking ---
    # np.argpartition is much faster than a full sort for finding top-k
    # We use negative scores because it finds the *smallest* elements

    top_k_indices = np.argpartition(-scores_matrix, k, axis=1)[:, :k]

    # --- Formatting Output ---
    batch_results = []
    for user_idx, indices in enumerate(top_k_indices):
        # We need to sort the top-k results themselves
        user_scores = scores_matrix[user_idx, indices]
        sorted_order = np.argsort(-user_scores)
        
        user_recs = [
            (titles[indices[i]], float(scores_matrix[user_idx, indices[i]]))
            for i in sorted_order
        ]
        batch_results.append(user_recs)
        
    return batch_results



if __name__ == "__main__":
    try:
        # Load the movie data
        movie_matrix, titles, feature_count = load_movies_csv(path=CONFIG["movies_path"])

        # Define the batch of users to get recommendations for
        users_batch = np.array([
            [9, 8, 2, 1, 1],   # User 1: loves Action + SciFi
            [1, 2, 9, 8, 3],   # User 2: loves Comedy + Romance
            [3, 5, 3, 5, 1]    # User 3: Average preference
        ])
        
        # Check if user vector features match movie features before running
        if users_batch.shape[1] != feature_count:
             raise ValueError(f"User batch has {users_batch.shape[1]} features, but movie data has {feature_count}. Please check your user definitions.")


        # Get recommendations using the settings from the CONFIG dictionary
        recommendations = recommend_batch(
            user_matrix=users_batch,
            movies_matrix=movie_matrix,
            titles=titles,
            k=CONFIG["top_k"],
            metric=CONFIG["metric"]
        )

        # Print the results in a user-friendly way
        print("\n--- ✅ Recommendation Results ---")
        for i, user_recs in enumerate(recommendations):
            print(f"\nTop {CONFIG['top_k']} recommendations for User #{i+1}:")
            for title, score in user_recs:
                print(f"  - '{title}' (Similarity Score: {score:.4f})")

    except (FileNotFoundError, ValueError) as e:
        # A clear, user-friendly error message
        logging.error(f"A critical error occurred: {e}")
        print(f"\n--- ❌ Error: {e}. Please check the file path and data format. ---")
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print("\n--- ❌ An unexpected error occurred. See logs for details. ---")










