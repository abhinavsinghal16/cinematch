import pandas as pd
from torch.utils.data import DataLoader

from dataset import MovieLensDataset


ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

dataset = MovieLensDataset(ratings)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

for user_ids, movie_ids, ratings_batch in loader:
    print("Users:", user_ids)
    print("Movies:", movie_ids)
    print("Ratings:", ratings_batch)

    print("User batch shape:", user_ids.shape)
    print("Movie batch shape:", movie_ids.shape)
    print("Rating batch shape:", ratings_batch.shape)

    break
