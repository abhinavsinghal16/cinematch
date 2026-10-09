import torch
import torch.nn as nn


class CineMatchModel(nn.Module):

    def __init__(self, num_users, num_movies, embedding_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

    def forward(self, user_ids, movie_ids):

        user_vectors = self.user_embedding(user_ids)

        movie_vectors = self.movie_embedding(movie_ids)

        score = (user_vectors * movie_vectors).sum(dim=1)

        return score
