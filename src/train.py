
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MovieLensDataset
from model import CineMatchModel
from pathlib import Path

# 1. Load the ratings dataset
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# 2. Split into training and validation sets
train_df, val_df = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# 3. Create datasets and DataLoaders
train_dataset = MovieLensDataset(train_df)
val_dataset = MovieLensDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

print(f"Training interactions: {len(train_dataset)}")
print(f"Validation interactions: {len(val_dataset)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# 4. Create the model, loss function, and optimizer
model = CineMatchModel(
    num_users=944,
    num_movies=1683,
    embedding_dim=32
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "best_model.pt"

best_val_rmse = float("inf")
best_epoch = 0
epochs_without_improvement = 0
patience = 5

# 5. Train and validate
num_epochs = 50

for epoch in range(num_epochs):

    # Training: update model parameters
    model.train()
    total_train_squared_error = 0.0

    for user_ids, movie_ids, targets in train_loader:
        optimizer.zero_grad()

        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_train_squared_error += (
            (predictions.detach() - targets) ** 2
        ).sum().item()

    train_mse = total_train_squared_error / len(train_dataset)
    train_rmse = train_mse ** 0.5

    # Validation: measure performance without updating parameters
    model.eval()
    total_val_squared_error = 0.0

    with torch.no_grad():
        for user_ids, movie_ids, targets in val_loader:
            predictions = model(user_ids, movie_ids)

            total_val_squared_error += (
                (predictions - targets) ** 2
            ).sum().item()

    val_mse = total_val_squared_error / len(val_dataset)
    val_rmse = val_mse ** 0.5

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} | "
        f"Train RMSE: {train_rmse:.4f} | "
        f"Validation RMSE: {val_rmse:.4f}"
    )

    # Save the model whenever validation RMSE improves
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), checkpoint_path)

        print(f"  Saved best model at epoch {best_epoch}")
    else:
        epochs_without_improvement += 1

    # Stop if validation performance hasn't improved for 5 epochs
    if epochs_without_improvement >= patience:
        print(
            f"Early stopping at epoch {epoch + 1}. "
            f"Best epoch: {best_epoch}, "
            f"best validation RMSE: {best_val_rmse:.4f}"
        )
        break

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_val_rmse:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

