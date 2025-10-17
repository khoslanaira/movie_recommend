#!/usr/bin/env python3
"""
Simplified test script for Movie Recommendation System
Tests core functionality without requiring all dependencies
"""

import os
import sys
import pandas as pd
import numpy as np

def test_data_loading():
    """Test data loading with the actual MovieLens dataset"""
    print("Testing data loading with MovieLens 100K...")
    
    try:
        from data_loader import MovieLensDataLoader
        
        # Test with actual dataset
        loader = MovieLensDataLoader()
        ratings, movies, users = loader.load_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Ratings: {len(ratings)} records")
        print(f"   - Movies: {len(movies)} movies")
        print(f"   - Users: {len(users)} users")
        
        # Test data exploration
        loader.explore_data()
        
        # Test data cleaning
        loader.clean_data(min_ratings=5)
        
        # Test data splitting
        train_data, test_data = loader.split_data(test_size=0.2, temporal=True)
        print(f"   - Train set: {len(train_data)} ratings")
        print(f"   - Test set: {len(test_data)} ratings")
        
        return True, loader
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_baseline_models(loader):
    """Test baseline models with actual data"""
    print("\nTesting baseline models...")
    
    try:
        from baseline_models import BaselineModels, PopularityRecommender
        
        # Get cleaned data
        train_data = loader.train_data
        movies = loader.movies
        
        # Test baseline models
        baselines = BaselineModels(train_data, movies)
        baselines.fit_global_average()
        print("✅ Global average baseline works")
        
        baselines.fit_popular_movies(min_ratings=10)
        print("✅ Popular movies baseline works")
        
        # Test popularity recommender
        pop_rec = PopularityRecommender(train_data)
        pop_rec.fit()
        
        # Test recommendations
        user_id = train_data.user_id.iloc[0]
        recommendations = pop_rec.recommend(user_id, n=5)
        print(f"✅ Popularity recommender works: {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_models(loader):
    """Test neural models with actual data"""
    print("\nTesting neural models...")
    
    try:
        import torch
        from neural_models import MatrixFactorization
        
        # Get user-item mappings
        user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()
        
        # Test model creation
        n_users = len(user_to_idx)
        n_items = len(item_to_idx)
        model = MatrixFactorization(n_users, n_items, n_factors=32)
        print(f"✅ Neural model created: {n_users} users, {n_items} items")
        
        # Test forward pass
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 1, 2])
        predictions = model(users, items)
        print(f"✅ Forward pass works: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_basic(loader):
    """Test basic evaluation metrics"""
    print("\nTesting evaluation metrics...")
    
    try:
        from evaluation import RecommendationEvaluator
        
        # Test with actual data
        test_data = loader.test_data
        movies = loader.movies
        
        evaluator = RecommendationEvaluator(test_data, movies)
        
        # Test basic metrics
        y_true = [4, 3, 5, 2, 4]
        y_pred = [4.1, 2.9, 4.8, 2.1, 3.9]
        
        rmse = evaluator.calculate_rmse(y_true, y_pred)
        mae = evaluator.calculate_mae(y_true, y_pred)
        
        print(f"✅ Evaluation metrics work: RMSE={rmse:.3f}, MAE={mae:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recommendation_demo(loader):
    """Test recommendation demo functionality"""
    print("\nTesting recommendation demo...")
    
    try:
        from recommendation_demo import MovieRecommendationDemo
        
        # Create demo instance
        demo = MovieRecommendationDemo()
        demo.loader = loader
        demo.train_data = loader.train_data
        demo.test_data = loader.test_data
        demo.movies = loader.movies
        
        # Get user-item mappings
        user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()
        demo.user_to_idx = user_to_idx
        demo.item_to_idx = item_to_idx
        demo.idx_to_user = idx_to_user
        demo.idx_to_item = idx_to_item
        
        # Test movie title lookup
        movie_id = demo.movies.item_id.iloc[0]
        title = demo.get_movie_title(movie_id)
        print(f"✅ Movie title lookup works: {title}")
        
        # Test user ratings
        user_id = demo.train_data.user_id.iloc[0]
        ratings = demo.get_user_ratings(user_id)
        print(f"✅ User ratings lookup works: {len(ratings)} ratings")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified tests"""
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - SIMPLIFIED TEST SUITE")
    print("=" * 60)
    
    # Test data loading first
    data_success, loader = test_data_loading()
    
    if not data_success:
        print("❌ Data loading failed. Cannot proceed with other tests.")
        return False
    
    # Run other tests
    tests = [
        ("Baseline Models", lambda: test_baseline_models(loader)),
        ("Neural Models", lambda: test_neural_models(loader)),
        ("Evaluation", lambda: test_evaluation_basic(loader)),
        ("Recommendation Demo", lambda: test_recommendation_demo(loader))
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
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 1  # Data loading passed
    total = len(results) + 1
    
    print("Data Loading: ✅ PASS")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed! System is ready to run.")
        print("\nTo run the complete system:")
        print("python main.py")
        print("\nTo run the interactive demo:")
        print("python recommendation_demo.py")
        print("\nNote: Some advanced features may require additional dependencies:")
        print("- scikit-surprise (for collaborative filtering)")
        print("- seaborn (for advanced plotting)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 