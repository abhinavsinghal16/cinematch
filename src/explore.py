import pandas as pd

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

print("\nFirst 5 rows:")
print(ratings.head())

print("\nDataset shape:")
print(ratings.shape)

print("\nUnique users:")
print(ratings["user_id"].nunique())

print("\nUnique movies:")
print(ratings["movie_id"].nunique())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())

print("\nHighest user id:")
print(ratings["user_id"].max())

print("\nHighest movie id:")
print(ratings["movie_id"].max())
