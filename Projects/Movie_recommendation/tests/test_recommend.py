# tests/test_recommend.py

import sys, os
import pytest   # <-- ADD THIS LINE
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recommender import load_movies_csv, l2_norms, recommend_batch

# ---------------------------
# Fixtures (sample test data)
# ---------------------------
@pytest.fixture
def sample_movies(tmp_path):
    # Create a temporary movies.csv file for testing
    file_path = tmp_path / "movies.csv"
    df = pd.DataFrame({
        "title": ["MovieA", "MovieB"],
        "Action": [1, 0],
        "Comedy": [0, 1]
    })
    df.to_csv(file_path, index=False)
    return str(file_path)

# ---------------------------
# Unit Tests
# ---------------------------
def test_load_movies_csv(sample_movies):
    movie_matrix, titles, _ = load_movies_csv(sample_movies)
    assert movie_matrix.shape == (2, 2)
    assert titles == ["MovieA", "MovieB"]

def test_recommend_batch_dot(sample_movies):
    movie_matrix, titles, _ = load_movies_csv(sample_movies)
    users = np.array([[1, 0]])   # Loves Action
    recs = recommend_batch(users, movie_matrix, titles, k=1, metric="dot")
    assert recs[0][0][0] == "MovieA"

def test_recommend_batch_cosine(sample_movies):
    movie_matrix, titles, _ = load_movies_csv(sample_movies)
    users = np.array([[0, 1]])   # Loves Comedy
    recs = recommend_batch(users, movie_matrix, titles, k=1, metric="cosine")
    assert recs[0][0][0] == "MovieB"

def test_feature_mismatch(sample_movies):
    movie_matrix, titles, feature_count = load_movies_csv(sample_movies)
    bad_users = np.array([[1, 2, 3]])   # 3 features, movies have 2
    with pytest.raises(ValueError):
        recommend_batch(bad_users, movie_matrix, titles, k=1)


# ---------------------------
# Edge Case Tests
# ---------------------------
def test_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        load_movies_csv("non_existent.csv")


