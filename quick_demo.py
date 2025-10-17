#!/usr/bin/env python3
"""
Quick Demo for Movie Recommendation System
Works with available dependencies to show core functionality
"""

import pandas as pd
import numpy as np
import torch
from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, PopularityRecommender
from neural_models import MatrixFactorization, NeuralRecommender

def main():
    """Run a quick demo of the recommendation system"""
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - QUICK DEMO")
    print("=" * 60)
    
    # Load data
    print("Loading MovieLens 100K dataset...")
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_data()
    
    # Clean and split data
    loader.clean_data(min_ratings=5)
    train_data, test_data = loader.split_data(test_size=0.2, temporal=True)
    
    print(f"Dataset loaded: {len(train_data)} train, {len(test_data)} test ratings")
    print(f"Users: {train_data.user_id.nunique()}, Movies: {train_data.item_id.nunique()}")
    
    # Train baseline models
    print("\n" + "-" * 40)
    print("TRAINING BASELINE MODELS")
    print("-" * 40)
    
    baselines = BaselineModels(train_data, movies)
    baselines.fit_global_average()
    baselines.fit_popular_movies(min_ratings=10)
    
    # Train popularity recommender
    pop_rec = PopularityRecommender(train_data)
    pop_rec.fit()
    
    # Get user-item mappings for neural models
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()
    
    # Train neural model (quick training)
    print("\n" + "-" * 40)
    print("TRAINING NEURAL MODEL")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx, device)
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    mf_model = MatrixFactorization(n_users, n_items, n_factors=32)
    
    print("Training neural matrix factorization...")
    neural_rec.train_model('Matrix_Factorization', mf_model, epochs=5, batch_size=1024)
    
    # Demo recommendations
    print("\n" + "-" * 40)
    print("DEMO RECOMMENDATIONS")
    print("-" * 40)
    
    # Get a sample user
    sample_user = train_data.user_id.iloc[0]
    print(f"Sample user: {sample_user}")
    
    # Show user's ratings
    user_ratings = train_data[train_data.user_id == sample_user]
    print(f"User {sample_user} has rated {len(user_ratings)} movies")
    
    # Show top rated movies by this user
    top_ratings = user_ratings.sort_values('rating', ascending=False).head(5)
    print("\nUser's top rated movies:")
    for _, row in top_ratings.iterrows():
        movie_title = movies[movies.item_id == row.item_id].title.iloc[0]
        print(f"  - {movie_title} ({row.rating} stars)")
    
    # Get recommendations from different models
    print("\nRecommendations from Popularity model:")
    pop_recs = pop_rec.recommend(sample_user, n=5)
    for i, movie_id in enumerate(pop_recs, 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title}")
    
    print("\nRecommendations from Neural MF model:")
    neural_recs = neural_rec.get_recommendations('Matrix_Factorization', sample_user, n=5)
    for i, movie_id in enumerate(neural_recs, 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title}")
    
    # Evaluate models
    print("\n" + "-" * 40)
    print("MODEL EVALUATION")
    print("-" * 40)
    
    # Simple evaluation on test set
    test_sample = test_data.head(100)  # Use first 100 test ratings
    
    # Global average baseline
    global_mean = train_data.rating.mean()
    global_preds = [global_mean] * len(test_sample)
    global_rmse = np.sqrt(np.mean((test_sample.rating.values - global_preds) ** 2))
    
    # Neural model predictions
    neural_preds = []
    neural_rec.models['Matrix_Factorization'].eval()
    with torch.no_grad():
        for _, row in test_sample.iterrows():
            user_idx = user_to_idx[row.user_id]
            item_idx = item_to_idx[row.item_id]
            pred = neural_rec.models['Matrix_Factorization'](
                torch.tensor([user_idx]), torch.tensor([item_idx])
            ).item()
            neural_preds.append(pred)
    
    neural_rmse = np.sqrt(np.mean((test_sample.rating.values - neural_preds) ** 2))
    
    print(f"Global Average RMSE: {global_rmse:.3f}")
    print(f"Neural MF RMSE: {neural_rmse:.3f}")
    
    # Show some popular movies
    print("\n" + "-" * 40)
    print("POPULAR MOVIES")
    print("-" * 40)
    
    movie_stats = train_data.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).round(3)
    movie_stats.columns = ['avg_rating', 'num_ratings']
    
    popular_movies = movie_stats[movie_stats.num_ratings >= 50].sort_values('avg_rating', ascending=False)
    
    print("Top 10 highest-rated movies (≥50 ratings):")
    for i, (movie_id, stats) in enumerate(popular_movies.head(10).iterrows(), 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title} ({stats['avg_rating']:.1f} stars, {stats['num_ratings']} ratings)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe system demonstrates:")
    print("✅ Data loading and preprocessing")
    print("✅ Baseline recommendation models")
    print("✅ Neural recommendation models")
    print("✅ Movie recommendations for users")
    print("✅ Model evaluation metrics")
    print("\nTo run the full system with all features:")
    print("pip install scikit-surprise seaborn")
    print("python main.py")

if __name__ == "__main__":
    main() 