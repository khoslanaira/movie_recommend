#!/usr/bin/env python3
"""
Test script for Movie Recommendation System
Quick verification that all components work correctly
"""

import os
import sys
import time
import pandas as pd
import numpy as np

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_loader import MovieLensDataLoader
        from baseline_models import BaselineModels, PopularityRecommender, ContentBasedRecommender
        from collaborative_filtering import CollaborativeFiltering
        from neural_models import NeuralRecommender, MatrixFactorization
        from evaluation import RecommendationEvaluator
        from recommendation_demo import MovieRecommendationDemo
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loader():
    """Test data loading functionality"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import MovieLensDataLoader
        
        # Create a small test dataset
        test_ratings = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 2],
            'timestamp': [1000, 1001, 1002, 1003, 1004, 1005]
        })
        
        test_movies = pd.DataFrame({
            'item_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genre_0': [1, 0, 1],
            'genre_1': [0, 1, 0]
        })
        
        # Test basic functionality
        loader = MovieLensDataLoader()
        print("✅ Data loader created successfully")
        
        # Test data splitting
        train_data, test_data = loader.split_data(test_ratings, test_size=0.5)
        print(f"✅ Data splitting works: {len(train_data)} train, {len(test_data)} test")
        
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_baseline_models():
    """Test baseline models"""
    print("\nTesting baseline models...")
    
    try:
        from baseline_models import BaselineModels, PopularityRecommender
        
        # Create test data
        train_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        
        movies = pd.DataFrame({
            'item_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genre_0': [1, 0, 1],
            'genre_1': [0, 1, 0]
        })
        
        # Test baseline models
        baselines = BaselineModels(train_data, movies)
        baselines.fit_global_average()
        print("✅ Global average baseline works")
        
        # Test popularity recommender
        pop_rec = PopularityRecommender(train_data)
        pop_rec.fit()
        recommendations = pop_rec.recommend(1, n=2)
        print(f"✅ Popularity recommender works: {recommendations}")
        
        return True
    except Exception as e:
        print(f"❌ Baseline models test failed: {e}")
        return False

def test_collaborative_filtering():
    """Test collaborative filtering (basic functionality)"""
    print("\nTesting collaborative filtering...")
    
    try:
        from collaborative_filtering import CollaborativeFiltering
        
        # Create test data
        train_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating': [5, 4, 3, 5, 4, 2]
        })
        
        test_data = pd.DataFrame({
            'user_id': [1, 2],
            'item_id': [3, 2],
            'rating': [4, 3]
        })
        
        # Test CF setup
        cf = CollaborativeFiltering(train_data, test_data)
        cf.prepare_data()
        print("✅ Collaborative filtering setup works")
        
        return True
    except Exception as e:
        print(f"❌ Collaborative filtering test failed: {e}")
        return False

def test_neural_models():
    """Test neural models (basic functionality)"""
    print("\nTesting neural models...")
    
    try:
        import torch
        from neural_models import MatrixFactorization
        
        # Test model creation
        n_users, n_items = 10, 20
        model = MatrixFactorization(n_users, n_items, n_factors=16)
        print("✅ Neural model creation works")
        
        # Test forward pass
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 1, 2])
        predictions = model(users, items)
        print(f"✅ Forward pass works: {predictions.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Neural models test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation metrics"""
    print("\nTesting evaluation...")
    
    try:
        from evaluation import RecommendationEvaluator
        
        # Create test data
        test_data = pd.DataFrame({
            'user_id': [1, 2],
            'item_id': [1, 2],
            'rating': [4, 3]
        })
        
        movies = pd.DataFrame({
            'item_id': [1, 2],
            'title': ['Movie A', 'Movie B']
        })
        
        # Test evaluator
        evaluator = RecommendationEvaluator(test_data, movies)
        
        # Test metrics
        y_true = [4, 3, 5]
        y_pred = [4.1, 2.9, 4.8]
        
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        mae = evaluator.calculate_mae(y_true, y_pred)
        
        print(f"✅ Evaluation metrics work: RMSE={rmse:.3f}, MAE={mae:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def run_quick_demo():
    """Run a quick demo with minimal data"""
    print("\nRunning quick demo...")
    
    try:
        # Create minimal test data
        train_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'item_id': [1, 2, 1, 3, 2, 3, 1, 4, 2, 4],
            'rating': [5, 4, 3, 5, 4, 2, 4, 3, 5, 4]
        })
        
        test_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [3, 4, 1],
            'rating': [4, 3, 5]
        })
        
        movies = pd.DataFrame({
            'item_id': [1, 2, 3, 4],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'genre_0': [1, 0, 1, 0],
            'genre_1': [0, 1, 0, 1]
        })
        
        # Test baseline
        from baseline_models import BaselineModels
        baselines = BaselineModels(train_data, movies)
        baselines.fit_global_average()
        
        # Test evaluation
        from evaluation import RecommendationEvaluator
        evaluator = RecommendationEvaluator(test_data, movies)
        
        print("✅ Quick demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("MOVIE RECOMMENDATION SYSTEM - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Baseline Models", test_baseline_models),
        ("Collaborative Filtering", test_collaborative_filtering),
        ("Neural Models", test_neural_models),
        ("Evaluation", test_evaluation),
        ("Quick Demo", run_quick_demo)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\nTo run the complete system:")
        print("python main.py")
        print("\nTo run the interactive demo:")
        print("python recommendation_demo.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 