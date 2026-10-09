# CineMatch

CineMatch is a recommendation system project built to learn and implement modern recommendation architectures used in large-scale personalization platforms.

## Motivation

Modern recommendation systems power products such as Netflix, Spotify, YouTube, LinkedIn, and TikTok. CineMatch explores the core architectures behind these systems, including embeddings, collaborative filtering, retrieval, ranking, and personalization.

## Goals

- Implement recommendation architectures with PyTorch
- Build user and movie embeddings
- Implement collaborative filtering
- Train neural recommendation models
- Explore retrieval and ranking architectures
- Evaluate recommendation quality
- Understand production recommendation system design

## Dataset Overview

MovieLens 100K

- Ratings: 100,000
- Users: 943
- Movies: 1,682

Each training example is represented as:

(user_id, movie_id) -> rating

## Current Implementation

### Recommendation Model

CineMatch currently uses embedding-based matrix factorization implemented in PyTorch.

- Learnable user embeddings represent users in a latent vector space.
- Learnable movie embeddings represent movies in the same-dimensional latent vector space.
- The dot product of a user embedding and a movie embedding produces a predicted rating.
- Mean Squared Error (MSE) is used as the training loss.
- The Adam optimizer updates model parameters through backpropagation.

The current implementation predicts explicit user ratings for individual user–movie pairs. Personalized top-K recommendation generation is planned next.

### Training and Evaluation

The training pipeline includes:

- An 80/20 training and validation split
- Mini-batch training using PyTorch DataLoader
- Separate training and validation loops
- RMSE tracking for training and validation data
- Best-model checkpointing based on validation RMSE
- Early stopping after five consecutive epochs without validation improvement

The best model weights are saved locally to `checkpoints/best_model.pt`. Generated checkpoints are excluded from version control.

## Planned Features

- Neural collaborative filtering
- Personalized recommendation generation
- Vector retrieval with FAISS
- Two-tower architecture
- Candidate generation and ranking pipelines
- Production-oriented architecture and system design documentation

## Technology Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn

## Status

### Completed

- Project setup and MovieLens dataset ingestion
- Dataset exploration and statistics
- Custom PyTorch Dataset implementation
- PyTorch DataLoader integration
- User and movie embedding layers
- Matrix factorization rating prediction model
- Training and validation pipeline
- RMSE tracking and evaluation
- Best-model checkpointing and early stopping

### In Progress

- Establishing baseline model performance
- Improving the recommendation model
- Generating personalized top-K recommendations

## Initial Experimental Results

An initial MovieLens 100K experiment using embedding-based matrix factorization achieved the following results:

| Metric | Result |
|---|---:|
| Training interactions | 80,000 |
| Validation interactions | 20,000 |
| Best validation RMSE | 1.6020 |
| Best epoch | 35 |
| Maximum epochs | 50 |
| Early stopping patience | 5 |

These results represent a single experimental run and serve as an initial reference point for future improvements. Model initialization and training configuration may affect the results. Reproducibility and comparison against a simple rating-prediction baseline are planned next steps.

## Future Enhancements

- Two-tower retrieval architecture
- ANN retrieval with FAISS
- Candidate generation and ranking pipelines
- Implicit feedback training
- Sequential recommendation models
- Hybrid recommendation systems
- Online inference service
- Real-time recommendation updates
