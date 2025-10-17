#!/usr/bin/env python3
"""
Train improved models with better performance
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader
from improved_neural_models import ImprovedNeuralRecommender, ImprovedMatrixFactorization, DeepTwoTowerModel, AdvancedNeuralMF
from baseline_models import BaselineModels, ContentBasedRecommender
from evaluation import RecommendationEvaluator

def setup_environment():
    print("=" * 60)
    print("IMPROVED MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure data exists
    if not os.path.exists("data/u.data"):
        print("MovieLens dataset not found. Downloading...")
        from data_downloader import download_movielens_100k
        download_movielens_100k()
    else:
        print("MovieLens dataset found.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_and_prepare_data():
    print("\n" + "=" * 40)
    print("PHASE 1: DATA LOADING AND PREPARATION")
    print("=" * 40)

    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    train_data, test_data = loader.split_data(test_size=0.2, temporal=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()

    print(f"\nData preparation completed!")
    print(f"Train set: {len(train_data)} ratings")
    print(f"Test set: {len(test_data)} ratings")
    print(f"Users: {len(user_to_idx)}, Movies: {len(item_to_idx)}")
    return loader, train_data, test_data, user_to_idx, item_to_idx

def train_improved_neural_models(train_data, test_data, user_to_idx, item_to_idx, device):
    print("\n" + "=" * 40)
    print("PHASE 2: IMPROVED NEURAL MODELS")
    print("=" * 40)

    neural_rec = ImprovedNeuralRecommender(train_data, test_data, user_to_idx, item_to_idx, device)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    print("Training Improved Matrix Factorization...")
    mf_model = ImprovedMatrixFactorization(n_users, n_items, n_factors=128, dropout=0.3)
    neural_rec.train_model('Improved_Matrix_Factorization', mf_model, 
                          epochs=120, batch_size=128, lr=0.0005)

    print("Training Deep Two-Tower Model...")
    two_tower_model = DeepTwoTowerModel(n_users, n_items, embedding_dim=128, 
                                       hidden_dims=[256, 128, 64], dropout=0.3)
    neural_rec.train_model('Deep_Two_Tower', two_tower_model, 
                          epochs=100, batch_size=128, lr=0.0005)

    print("Training Advanced Neural MF...")
    neural_mf_model = AdvancedNeuralMF(n_users, n_items, n_factors=128, 
                                     layers=[256, 128, 64], dropout=0.3)
    neural_rec.train_model('Advanced_Neural_MF', neural_mf_model, 
                          epochs=110, batch_size=128, lr=0.0005)

    neural_models = {
        'Improved_Matrix_Factorization': neural_rec.models.get('Improved_Matrix_Factorization'),
        'Deep_Two_Tower': neural_rec.models.get('Deep_Two_Tower'),
        'Advanced_Neural_MF': neural_rec.models.get('Advanced_Neural_MF')
    }

    print("Improved neural models training completed!")
    return neural_models, neural_rec

def evaluate_improved_models(test_data, movies, neural_models, neural_rec):
    print("\n" + "=" * 40)
    print("PHASE 3: MODEL EVALUATION")
    print("=" * 40)

    evaluator = RecommendationEvaluator(test_data, movies)
    
    # Create adapter for neural models
    class ImprovedNeuralAdapter:
        def __init__(self, model_name, model_obj, neural_recommender):
            self.model_name = model_name
            self.model = model_obj
            self.neural_recommender = neural_recommender
            self.user_to_idx = getattr(neural_recommender, 'user_to_idx', None)
            self.item_to_idx = getattr(neural_recommender, 'item_to_idx', None)
            self.device = getattr(neural_recommender, 'device', 'cpu')

        def predict_rating(self, user_id, item_id):
            try:
                if self.neural_recommender is not None and hasattr(self.neural_recommender, 'predict_rating'):
                    val = self.neural_recommender.predict_rating(self.model_name, user_id, item_id)
                    return float(val) if val is not None else None
            except Exception:
                pass

            if self.user_to_idx is None or self.item_to_idx is None:
                return None
            if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                return None

            self.model.eval()
            with torch.no_grad():
                u_idx = self.user_to_idx[user_id]
                i_idx = self.item_to_idx[item_id]
                u_t = torch.tensor([u_idx], device=self.device)
                i_t = torch.tensor([i_idx], device=self.device)
                out = self.model(u_t, i_t)
                return float(out.cpu().item())

    # Evaluate neural models
    neural_results = {}
    for name, model in neural_models.items():
        if model is None:
            continue
        adapter = ImprovedNeuralAdapter(name, model, neural_rec)
        try:
            neural_results[name] = evaluator.evaluate_rating_prediction(adapter, name, test_data)
            print(f"{name} - RMSE: {neural_results[name]['RMSE']:.3f}, MAE: {neural_results[name]['MAE']:.3f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            neural_results[name] = {'RMSE': np.nan, 'MAE': np.nan}

    return neural_results

def main():
    start_time = time.time()
    try:
        device = setup_environment()
        loader, train_data, test_data, user_to_idx, item_to_idx = load_and_prepare_data()

        # Train improved neural models
        neural_models, neural_rec = train_improved_neural_models(
            train_data, test_data, user_to_idx, item_to_idx, device
        )

        # Evaluate models
        results = evaluate_improved_models(test_data, loader.movies, neural_models, neural_rec)
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv('improved_model_results.csv')
        
        print("\n" + "=" * 50)
        print("IMPROVED MODEL RESULTS")
        print("=" * 50)
        print(results_df.round(3))
        
        # Generate JSON for frontend
        import json
        json_results = []
        for model, row in results_df.iterrows():
            rmse = row.get("RMSE", np.nan)
            mae = row.get("MAE", np.nan)
            json_results.append({
                "Model": model,
                "RMSE": None if pd.isna(rmse) else round(rmse, 3),
                "MAE": None if pd.isna(mae) else round(mae, 3),
                "Status": "trained" if not pd.isna(rmse) else "skipped",
                "Performance": "Excellent" if rmse < 1.0 else "Good" if rmse < 1.2 else "Poor"
            })

        with open("improved_model_results.json", "w") as f:
            json.dump(json_results, f, indent=4)

        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        print("Results saved to improved_model_results.csv and improved_model_results.json")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
