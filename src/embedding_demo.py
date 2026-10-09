import torch
import torch.nn as nn

user_embedding = nn.Embedding(
    num_embeddings=943,
    embedding_dim=4
)

print("Embedding table shape:")
print(user_embedding.weight.shape)

print("\nUser 196 vector:")
print(user_embedding(torch.tensor(196)))

print("\nUser 22 vector:")
print(user_embedding(torch.tensor(22)))
