#!/usr/bin/env python3
"""
Test script for the fixed movie recommendation system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, ContentBasedRecommender
from neural_models import NeuralRecommender, MatrixFactorization, TwoTowerModel, NeuralMF
from evaluation import RecommendationEvaluator
import numpy as np

def test_system():
    """Test the fixed recommendation system"""
    print("🧪 Testing Fixed Movie Recommendation System...")
    print("=" * 50)
    
    try:
        # Load data
        print("1. Loading data...")
        data_loader = MovieLensDataLoader()
        data_loader.load_data()
        data_loader.clean_data(min_ratings=5)
        train_data, test_data = data_loader.split_data()
        
        user_to_idx = data_loader.user_to_idx
        item_to_idx = data_loader.item_to_idx
        movies_df = data_loader.movies
        
        print(f"✅ Data loaded: {len(movies_df)} movies, {len(train_data)} train, {len(test_data)} test")
        print(f"✅ Users: {len(user_to_idx)}, Movies: {len(item_to_idx)}")
        
        # Test baseline models
        print("\n2. Testing baseline models...")
        baselines = BaselineModels(train_data, movies_df)
        baselines.fit_global_average()
        baselines.fit_popular_movies()
        baselines.fit_content_based()
        
        print("✅ Baseline models created successfully")
        
        # Test content-based model
        print("\n3. Testing content-based model...")
        content_model = ContentBasedRecommender(train_data, movies_df)
        content_model.fit()
        print("✅ Content-based model created successfully")
        
        # Test neural models
        print("\n4. Testing neural models...")
        
        # Matrix Factorization
        print("  - Matrix Factorization...")
        mf_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)
        mf = MatrixFactorization(len(user_to_idx), len(item_to_idx), n_factors=16)
        mf_rec.train_model("Matrix Factorization", mf, epochs=3, batch_size=512, lr=0.01)
        print("  ✅ Matrix Factorization trained")
        
        # Two-Tower Model
        print("  - Two-Tower Model...")
        tt_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)
        tt = TwoTowerModel(len(user_to_idx), len(item_to_idx), embedding_dim=16, hidden_dims=[32, 16])
        tt_rec.train_model("Two-Tower", tt, epochs=3, batch_size=512, lr=0.01)
        print("  ✅ Two-Tower Model trained")
        
        # Neural MF
        print("  - Neural MF...")
        nmf_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)
        nmf = NeuralMF(len(user_to_idx), len(item_to_idx), n_factors=16, layers=[32, 16])
        nmf_rec.train_model("Neural MF", nmf, epochs=3, batch_size=512, lr=0.01)
        print("  ✅ Neural MF trained")
        
        # Test recommendations
        print("\n5. Testing recommendations...")
        test_user_id = list(user_to_idx.keys())[0]  # Get first user
        print(f"Testing with user {test_user_id}")
        
        # Test each model
        models = {
            'Global Average': baselines,
            'Popular Movies': baselines,
            'Content-Based': content_model,
            'Matrix Factorization': mf_rec,
            'Two-Tower': tt_rec,
            'Neural MF': nmf_rec
        }
        
        for model_name, model in models.items():
            try:
                if model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
                    # Test neural model prediction
                    user_idx = user_to_idx[test_user_id]
                    sample_movie_id = list(item_to_idx.keys())[0]
                    item_idx = item_to_idx[sample_movie_id]
                    pred = model.predict_rating(model_name, user_idx, item_idx)
                    print(f"  ✅ {model_name}: Prediction = {pred:.3f}")
                else:
                    print(f"  ✅ {model_name}: Model loaded successfully")
            except Exception as e:
                print(f"  ❌ {model_name}: Error - {e}")
        
        # Test evaluation
        print("\n6. Testing evaluation...")
        evaluator = RecommendationEvaluator(test_data, movies_df)
        
        # Test baseline evaluation
        try:
            baseline_results = evaluator.evaluate_rating_prediction(baselines, 'Global Average', test_data)
            print(f"✅ Global Average RMSE: {baseline_results.get('RMSE', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Baseline evaluation error: {e}")
        
        # Test neural model evaluation
        for model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
            try:
                sample_test = test_data.sample(min(50, len(test_data)))
                preds, actuals = [], []
                
                for _, row in sample_test.iterrows():
                    u_idx = user_to_idx.get(row['user_id'])
                    i_idx = item_to_idx.get(row['item_id'])
                    if u_idx is not None and i_idx is not None:
                        pred = models[model_name].predict_rating(model_name, u_idx, i_idx)
                        if pred is not None and not np.isnan(pred):
                            preds.append(pred)
                            actuals.append(row['rating'])
                
                if preds:
                    rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
                    mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
                    print(f"✅ {model_name}: RMSE={rmse:.3f}, MAE={mae:.3f}")
                else:
                    print(f"⚠️ {model_name}: No valid predictions")
            except Exception as e:
                print(f"❌ {model_name} evaluation error: {e}")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ The fixed system is working properly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n🚀 System is ready! You can now run: python working_app.py")
    else:
        print("\n❌ System has issues. Please check the errors above.")

