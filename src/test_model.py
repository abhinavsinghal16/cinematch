import torch

from model import CineMatchModel


model = CineMatchModel(
    num_users=944,
    num_movies=1683,
    embedding_dim=4
)

user_ids = torch.tensor([196, 22, 244])

movie_ids = torch.tensor([242, 377, 51])

scores = model(user_ids, movie_ids)

print(scores)
print(scores.shape)
