#!/usr/bin/env python3
"""
Movie Recommendation System - IIT Ropar Project
Updated main.py — integrates with improved neural_models and safe evaluation adapter.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Import our modules
from data_downloader import download_movielens_100k
from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, PopularityRecommender, ContentBasedRecommender
try:
    from collaborative_filtering import CollaborativeFiltering, AdvancedCollaborativeFiltering
    COLLABORATIVE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-surprise not available. Collaborative filtering models will be skipped.")
    COLLABORATIVE_AVAILABLE = False

from neural_models import NeuralRecommender, MatrixFactorization, TwoTowerModel, NeuralMF
from evaluation import RecommendationEvaluator
# recommendation_demo is optional; import if present
try:
    from recommendation_demo import MovieRecommendationDemo
except Exception:
    MovieRecommendationDemo = None


def setup_environment():
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - IIT ROPAR PROJECT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure data exists
    if not os.path.exists("ml-100k/u.data"):
        print("MovieLens dataset not found. Downloading...")
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
    ratings, movies, users = loader.load_data()

    loader.explore_data()
    loader.clean_data(min_ratings=5)

    train_data, test_data = loader.split_data(test_size=0.2, temporal=True)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()

    print(f"\nData preparation completed!")
    print(f"Train set: {len(train_data)} ratings")
    print(f"Test set: {len(test_data)} ratings")
    print(f"Users: {len(user_to_idx)}, Movies: {len(item_to_idx)}")
    return loader, train_data, test_data, movies, user_to_idx, item_to_idx


def train_baseline_models(train_data, movies):
    print("\n" + "=" * 40)
    print("PHASE 2: BASELINE MODELS")
    print("=" * 40)

    baselines = BaselineModels(train_data, movies)
    baselines.fit_global_average()
    baselines.fit_popular_movies(min_ratings=10)
    baselines.fit_content_based()

    pop_rec = PopularityRecommender(train_data)
    pop_rec.fit()

    content_rec = ContentBasedRecommender(train_data, movies)
    content_rec.fit()

    baseline_models = {
        'Global_Average': baselines,
        'Popular_Movies': baselines.popular_movies,
        'Popularity': pop_rec,
        'Content_Based': content_rec
    }

    print("Baseline models training completed!")
    return baseline_models


def train_collaborative_filtering(train_data, test_data):
    print("\n" + "=" * 40)
    print("PHASE 3: COLLABORATIVE FILTERING")
    print("=" * 40)

    if not COLLABORATIVE_AVAILABLE:
        print("Skipping collaborative filtering (scikit-surprise not available)")
        return {}

    try:
        cf = CollaborativeFiltering(train_data, test_data)
        cf.prepare_data()
        cf.fit_svd(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        cf.fit_user_knn(k=50)
        cf.fit_item_knn(k=50)
        cf.fit_knn_with_means(k=50)

        cf_models = {
            'SVD': cf.models.get('SVD'),
            'User_KNN': cf.models.get('User_KNN'),
            'Item_KNN': cf.models.get('Item_KNN'),
            'KNN_Means': cf.models.get('KNN_Means')
        }

        print("Collaborative filtering models training completed!")
        return cf_models

    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return {}


def train_neural_models(train_data, test_data, user_to_idx, item_to_idx, device):
    print("\n" + "=" * 40)
    print("PHASE 4: NEURAL MODELS")
    print("=" * 40)

    neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx, device)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    print("Training Neural Matrix Factorization...")
    mf_model = MatrixFactorization(n_users, n_items, n_factors=64, dropout=0.1)
    neural_rec.train_model('Matrix_Factorization', mf_model, epochs=80, batch_size=256, lr=0.001)

    print("Training Two-Tower Model...")
    two_tower_model = TwoTowerModel(n_users, n_items, embedding_dim=64, dropout=0.2)
    neural_rec.train_model('Two_Tower', two_tower_model, epochs=60, batch_size=256, lr=0.001)

    print("Training Neural MF...")
    neural_mf_model = NeuralMF(n_users, n_items, n_factors=64, layers=[64, 32], dropout=0.2)
    neural_rec.train_model('Neural_MF', neural_mf_model, epochs=70, batch_size=256, lr=0.001)

    neural_models = {
        'Matrix_Factorization': neural_rec.models.get('Matrix_Factorization'),
        'Two_Tower': neural_rec.models.get('Two_Tower'),
        'Neural_MF': neural_rec.models.get('Neural_MF')
    }

    print("Neural models training completed!")
    return neural_models, neural_rec


def evaluate_all_models_comprehensive(train_data, test_data, movies, all_models):
    print("\n" + "=" * 40)
    print("PHASE 5: MODEL EVALUATION")
    print("=" * 40)

    evaluator = RecommendationEvaluator(test_data, movies)

    # Baselines
    print("Evaluating baseline models...")
    baseline_results = {}
    for name, model in all_models['baseline'].items():
        if name == 'Global_Average':
            baseline_results[name] = evaluator.evaluate_rating_prediction(model, name, test_data)
        elif name in ['Popularity', 'Content_Based']:
            baseline_results[name] = evaluator.evaluate_recommendations(model, name, train_data, test_data)

    # Collaborative filtering
    print("Evaluating collaborative filtering models...")
    cf_results = {}
    for name, model in all_models['collaborative'].items():
        if model is None:
            continue
        cf_results[name] = evaluator.evaluate_rating_prediction(model, name, test_data)

    # Neural models
    print("Evaluating neural models...")
    neural_results = {}
    neural_rec = all_models.get('neural_rec')

    class NeuralAdapter:
        def __init__(self, model_name, model_obj, neural_recommender):
            self.model_name = model_name
            self.model = model_obj
            self.neural_recommender = neural_recommender
            self.user_to_idx = getattr(model_obj, 'user_to_idx', None) or getattr(neural_recommender, 'user_to_idx', None)
            self.item_to_idx = getattr(model_obj, 'item_to_idx', None) or getattr(neural_recommender, 'item_to_idx', None)
            self.device = getattr(model_obj, 'device', None) or getattr(neural_recommender, 'device', 'cpu')

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

        def get_recommendations(self, user_id, n=10):
            if self.neural_recommender is not None:
                try:
                    return self.neural_recommender.get_recommendations(self.model_name, user_id, n=n)
                except Exception:
                    pass
            return []

        def forward(self, user_tensor, item_tensor):
            return self.model(user_tensor.to(self.device), item_tensor.to(self.device))

        def __call__(self, user_tensor, item_tensor):
            return self.forward(user_tensor, item_tensor)

    for name, model in all_models['neural'].items():
        if model is None:
            continue
        adapter = NeuralAdapter(name, model, neural_rec)
        try:
            neural_results[name] = evaluator.evaluate_rating_prediction(adapter, name, test_data)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            neural_results[name] = {'RMSE': np.nan, 'MAE': np.nan}
        try:
            neural_results[name].update(evaluator.evaluate_recommendations(adapter, name, train_data, test_data))
        except Exception:
            neural_results[name].update({'NDCG@10': np.nan, 'Hit_Rate@10': np.nan,
                                         'Precision@10': np.nan, 'Recall@10': np.nan, 'F1@10': np.nan})

    all_results = {**baseline_results, **cf_results, **neural_results}
    comparison_df = pd.DataFrame(all_results).T

    print("\n" + "=" * 50)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 50)
    print(comparison_df.round(3))

    # Save CSV
    comparison_df.to_csv('model_results.csv')

    # ✅ NEW: Generate JSON for frontend
    import json

    def classify_performance(rmse):
        if pd.isna(rmse):
            return "Unknown"
        if rmse < 1.1:
            return "Excellent"
        elif rmse < 1.3:
            return "Good"
        elif rmse < 2.0:
            return "Poor"
        return "Very Poor"

    json_results = []
    for model, row in comparison_df.iterrows():
        rmse = row.get("RMSE", np.nan)
        mae = row.get("MAE", np.nan)
        json_results.append({
            "Model": model,
            "RMSE": None if pd.isna(rmse) else round(rmse, 3),
            "MAE": None if pd.isna(mae) else round(mae, 3),
            "Status": "trained" if not pd.isna(rmse) else "skipped",
            "Performance": classify_performance(rmse)
        })

    with open("model_results.json", "w") as f:
        json.dump(json_results, f, indent=4)

    print("Results saved to model_results.csv and model_results.json")
    return comparison_df



def main():
    start_time = time.time()
    try:
        device = setup_environment()
        loader, train_data, test_data, movies, user_to_idx, item_to_idx = load_and_prepare_data()

        baseline_models = train_baseline_models(train_data, movies)
        cf_models = train_collaborative_filtering(train_data, test_data) if COLLABORATIVE_AVAILABLE else {}

        neural_models, neural_rec = train_neural_models(train_data, test_data, user_to_idx, item_to_idx, device)

        all_models = {
            'baseline': baseline_models,
            'collaborative': cf_models,
            'neural': neural_models,
            'neural_rec': neural_rec
        }

        comparison_df = evaluate_all_models_comprehensive(train_data, test_data, movies, all_models)
        print("\nTraining & evaluation complete. Results written to model_results.csv")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
