#!/usr/bin/env python3
"""
Test script for improved models
"""

import os
import sys
import pandas as pd
import numpy as np
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader
from neural_models import NeuralRecommender, MatrixFactorization, TwoTowerModel, NeuralMF
from baseline_models import BaselineModels, ContentBasedRecommender

def test_improved_models():
    print("Testing improved models...")
    
    # Load data
    data_loader = MovieLensDataLoader()
    data_loader.load_data()
    data_loader.clean_data(min_ratings=5)
    train_data, test_data = data_loader.split_data()
    
    movies_df = data_loader.movies
    user_to_idx = data_loader.user_to_idx
    item_to_idx = data_loader.item_to_idx
    
    print(f"Data loaded: {len(movies_df)} movies, {len(train_data)} train ratings")
    
    # Test Matrix Factorization
    print("\nTesting Matrix Factorization...")
    mf_model = MatrixFactorization(len(user_to_idx), len(item_to_idx), n_factors=32)
    neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)
    
    # Quick training for testing
    neural_rec.train_model("Matrix_Factorization", mf_model, epochs=5, batch_size=128)
    
    # Test prediction
    test_user = list(user_to_idx.keys())[0]
    test_item = list(item_to_idx.keys())[0]
    
    try:
        pred = neural_rec.predict_rating("Matrix_Factorization", test_user, test_item)
        print(f"Prediction for user {test_user}, item {test_item}: {pred:.3f}")
        print("✅ Matrix Factorization working!")
    except Exception as e:
        print(f"❌ Matrix Factorization error: {e}")
    
    # Test baseline models
    print("\nTesting baseline models...")
    baselines = BaselineModels(train_data, movies_df)
    baselines.fit_global_average()
    baselines.fit_popular_movies()
    
    try:
        global_pred = baselines.predict_global_average([test_user], [test_item])
        print(f"Global average prediction: {global_pred[0]:.3f}")
        print("✅ Baseline models working!")
    except Exception as e:
        print(f"❌ Baseline models error: {e}")
    
    print("\n🎉 Model testing completed!")

if __name__ == "__main__":
    test_improved_models()