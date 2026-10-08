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

## Planned Features

### Phase 1
- Dataset loading
- PyTorch Dataset and DataLoader
- User and movie embeddings
- Rating prediction

### Phase 2
- Neural collaborative filtering
- Training and evaluation pipelines
- Recommendation generation

### Phase 3
- Vector retrieval
- FAISS integration
- Two-tower architecture

### Phase 4
- Retrieval and ranking pipeline
- System design documentation
- Production-oriented architecture

## Technology Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn

## Status

### Completed
- Project setup
- MovieLens dataset ingestion
- Dataset exploration and statistics
- Custom PyTorch Dataset implementation
- PyTorch DataLoader integration

### In Progress
- User and movie embeddings
- Neural recommendation model

## Future Enhancements

- Two-tower retrieval architecture
- ANN retrieval with FAISS
- Candidate generation and ranking pipelines
- Implicit feedback training
- Sequential recommendation models
- Hybrid recommendation systems
- Online inference service
- Real-time recommendation updates
