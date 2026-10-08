# src/test_dataset.py

import pandas as pd

from dataset import MovieLensDataset

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

dataset = MovieLensDataset(ratings)

print("Length:", len(dataset))

print("First Example:")
print(dataset[0])

print("Second Example:")
print(dataset[1])
