import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):

    def __init__(self, dataframe):
        self.users = torch.tensor(
            dataframe["user_id"].values,
            dtype=torch.long
        )

        self.movies = torch.tensor(
            dataframe["movie_id"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.movies[idx],
            self.ratings[idx]
        )
