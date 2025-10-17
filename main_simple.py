#!/usr/bin/env python3
"""
Movie Recommendation System - Simplified Version
Works without scikit-surprise but provides comprehensive functionality
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Import our modules
from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, PopularityRecommender, ContentBasedRecommender
from neural_models import NeuralRecommender, MatrixFactorization, TwoTowerModel, NeuralMF
from evaluation import RecommendationEvaluator

def setup_environment():
    """Setup the environment"""
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - SIMPLIFIED VERSION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if data exists
    if not os.path.exists("movie/ml-100k/u.data"):
        print("❌ MovieLens dataset not found in movie/ml-100k/")
        print("Please ensure the dataset is in the correct location")
        return None
    
    print("✅ MovieLens dataset found.")
    
    # Set device for neural models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device

def load_and_prepare_data():
    """Load and prepare the MovieLens dataset"""
    print("\n" + "=" * 40)
    print("PHASE 1: DATA LOADING AND PREPARATION")
    print("=" * 40)
    
    # Initialize data loader
    loader = MovieLensDataLoader()
    
    # Load data
    ratings, movies, users = loader.load_data()
    
    # Explore data
    loader.explore_data()
    
    # Clean data
    loader.clean_data(min_ratings=5)
    
    # Split data
    train_data, test_data = loader.split_data(test_size=0.2, temporal=True)
    
    # Get user-item mappings
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = loader.get_user_item_mapping()
    
    print(f"\nData preparation completed!")
    print(f"Train set: {len(train_data)} ratings")
    print(f"Test set: {len(test_data)} ratings")
    print(f"Users: {len(user_to_idx)}, Movies: {len(item_to_idx)}")
    
    return loader, train_data, test_data, movies, user_to_idx, item_to_idx

def train_baseline_models(train_data, movies):
    """Train baseline recommendation models"""
    print("\n" + "=" * 40)
    print("PHASE 2: BASELINE MODELS")
    print("=" * 40)
    
    # Initialize baseline models
    baselines = BaselineModels(train_data, movies)
    
    # Fit models
    print("Training Global Average baseline...")
    baselines.fit_global_average()
    
    print("Training Popular Movies baseline...")
    baselines.fit_popular_movies(min_ratings=10)
    
    print("Training Content-Based model...")
    baselines.fit_content_based()
    
    # Create individual recommenders
    print("Training Popularity Recommender...")
    pop_rec = PopularityRecommender(train_data)
    pop_rec.fit()
    
    print("Training Content-Based Recommender...")
    content_rec = ContentBasedRecommender(train_data, movies)
    content_rec.fit()
    
    baseline_models = {
        'Global_Average': baselines,
        'Popular_Movies': baselines.popular_movies,
        'Popularity': pop_rec,
        'Content_Based': content_rec
    }
    
    print("✅ Baseline models training completed!")
    return baseline_models

def train_neural_models(train_data, test_data, user_to_idx, item_to_idx, device):
    """Train neural recommendation models"""
    print("\n" + "=" * 40)
    print("PHASE 3: NEURAL MODELS")
    print("=" * 40)
    
    # Initialize neural recommender
    neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx, device)
    
    # Get dimensions
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    # Train Matrix Factorization
    print("Training Neural Matrix Factorization...")
    mf_model = MatrixFactorization(n_users, n_items, n_factors=64)
    neural_rec.train_model('Matrix_Factorization', mf_model, epochs=10, batch_size=1024)
    
    # Train Two-Tower Model
    print("Training Two-Tower Model...")
    two_tower_model = TwoTowerModel(n_users, n_items, embedding_dim=64)
    neural_rec.train_model('Two_Tower', two_tower_model, epochs=8, batch_size=1024)
    
    # Train Neural MF
    print("Training Neural MF...")
    neural_mf_model = NeuralMF(n_users, n_items, n_factors=64, layers=[100, 50])
    neural_rec.train_model('Neural_MF', neural_mf_model, epochs=10, batch_size=1024)
    
    neural_models = {
        'Neural_MF': neural_rec.models['Matrix_Factorization'],
        'Two_Tower': neural_rec.models['Two_Tower'],
        'Neural_MF_MLP': neural_rec.models['Neural_MF']
    }
    
    print("✅ Neural models training completed!")
    return neural_models, neural_rec

def evaluate_all_models(train_data, test_data, movies, baseline_models, neural_models):
    """Evaluate all models"""
    print("\n" + "=" * 40)
    print("PHASE 4: MODEL EVALUATION")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(test_data, movies)
    
    # Evaluate baseline models
    print("Evaluating baseline models...")
    baseline_results = {}
    
    # Global Average
    baseline_results['Global_Average'] = evaluator.evaluate_rating_prediction(
        baseline_models['Global_Average'], 'Global_Average', test_data
    )
    
    # Popularity and Content-Based
    for name in ['Popularity', 'Content_Based']:
        baseline_results[name] = evaluator.evaluate_recommendations(
            baseline_models[name], name, train_data, test_data
        )
    
    # Evaluate neural models
    print("Evaluating neural models...")
    neural_results = {}
    for name, model in neural_models.items():
        neural_results[name] = evaluator.evaluate_rating_prediction(
            model, name, test_data
        )
    
    # Combine all results
    all_results = {**baseline_results, **neural_results}
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    print("\n" + "=" * 50)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 50)
    print(comparison_df.round(3))
    
    # Generate plots and report
    evaluator.plot_comparison(comparison_df, save_path='model_comparison.png')
    report = evaluator.generate_report(comparison_df, save_path='evaluation_report.txt')
    
    # Save results
    comparison_df.to_csv('model_results.csv')
    
    return comparison_df, report

def generate_final_report(comparison_df, start_time):
    """Generate final project report"""
    print("\n" + "=" * 40)
    print("PHASE 5: FINAL REPORT")
    print("=" * 40)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    report = []
    report.append("=" * 80)
    report.append("MOVIE RECOMMENDATION SYSTEM - SIMPLIFIED VERSION REPORT")
    report.append("=" * 80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total execution time: {total_time/60:.2f} minutes")
    report.append("")
    
    # Project overview
    report.append("PROJECT OVERVIEW:")
    report.append("-" * 20)
    report.append("Domain: Movie Recommendations")
    report.append("Dataset: MovieLens 100K")
    report.append("Goal: Predict user ratings and recommend top-N movies")
    report.append("Success Metrics: RMSE, MAE, NDCG@10, Hit Rate@10")
    report.append("Note: Simplified version without scikit-surprise")
    report.append("")
    
    # Models implemented
    report.append("MODELS IMPLEMENTED:")
    report.append("-" * 20)
    report.append("Baseline Models:")
    report.append("  - Global Average")
    report.append("  - Popular Movies")
    report.append("  - Content-Based Filtering")
    report.append("")
    report.append("Neural Models:")
    report.append("  - Neural Matrix Factorization")
    report.append("  - Two-Tower Model")
    report.append("  - Neural MF with MLP")
    report.append("")
    
    # Results summary
    report.append("RESULTS SUMMARY:")
    report.append("-" * 20)
    
    if 'RMSE' in comparison_df.columns:
        best_rmse_model = comparison_df['RMSE'].idxmin()
        best_rmse = comparison_df.loc[best_rmse_model, 'RMSE']
        report.append(f"Best RMSE: {best_rmse_model} ({best_rmse:.3f})")
    
    if 'NDCG@10' in comparison_df.columns:
        best_ndcg_model = comparison_df['NDCG@10'].idxmax()
        best_ndcg = comparison_df.loc[best_ndcg_model, 'NDCG@10']
        report.append(f"Best NDCG@10: {best_ndcg_model} ({best_ndcg:.3f})")
    
    report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS:")
    report.append("-" * 20)
    report.append(comparison_df.round(3).to_string())
    report.append("")
    
    # Success criteria
    report.append("SUCCESS CRITERIA CHECKLIST:")
    report.append("-" * 30)
    
    # Check RMSE < 1.0
    if 'RMSE' in comparison_df.columns:
        best_rmse = comparison_df['RMSE'].min()
        rmse_success = best_rmse < 1.0
        report.append(f"✅ RMSE < 1.0: {best_rmse:.3f} {'✓' if rmse_success else '✗'}")
    
    # Check if we have at least 3 algorithms
    num_models = len(comparison_df)
    models_success = num_models >= 3
    report.append(f"✅ At least 3 algorithms: {num_models} models {'✓' if models_success else '✗'}")
    
    # Check if we have proper evaluation
    evaluation_success = 'RMSE' in comparison_df.columns and 'MAE' in comparison_df.columns
    report.append(f"✅ Proper train/test evaluation: {'✓' if evaluation_success else '✗'}")
    
    # Check if we have interactive demo
    demo_success = os.path.exists('recommendation_demo.py')
    report.append(f"✅ Interactive recommendation demo: {'✓' if demo_success else '✗'}")
    
    # Check if we have clear performance comparison
    comparison_success = len(comparison_df) > 1
    report.append(f"✅ Clear performance comparison: {'✓' if comparison_success else '✗'}")
    
    report.append("")
    report.append("=" * 80)
    
    # Write report
    full_report = "\n".join(report)
    with open('final_project_report.txt', 'w') as f:
        f.write(full_report)
    
    print(full_report)
    return full_report

def demo_recommendations(train_data, movies, baseline_models, neural_models, user_to_idx, item_to_idx):
    """Demonstrate movie recommendations"""
    print("\n" + "=" * 40)
    print("DEMO: MOVIE RECOMMENDATIONS")
    print("=" * 40)
    
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
    pop_recs = baseline_models['Popularity'].recommend(sample_user, n=5)
    for i, movie_id in enumerate(pop_recs, 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title}")
    
    print("\nRecommendations from Content-Based model:")
    content_recs = baseline_models['Content_Based'].recommend(sample_user, n=5)
    for i, movie_id in enumerate(content_recs, 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title}")
    
    print("\nRecommendations from Neural MF model:")
    neural_recs = neural_models['Neural_MF'].get_recommendations('Matrix_Factorization', sample_user, n=5)
    for i, movie_id in enumerate(neural_recs, 1):
        movie_title = movies[movies.item_id == movie_id].title.iloc[0]
        print(f"  {i}. {movie_title}")

def main():
    """Main function to run the simplified movie recommendation system"""
    start_time = time.time()
    
    try:
        # Setup environment
        device = setup_environment()
        if device is None:
            return 1
        
        # Load and prepare data
        loader, train_data, test_data, movies, user_to_idx, item_to_idx = load_and_prepare_data()
        
        # Train baseline models
        baseline_models = train_baseline_models(train_data, movies)
        
        # Train neural models
        neural_models, neural_rec = train_neural_models(train_data, test_data, user_to_idx, item_to_idx, device)
        
        # Evaluate all models
        comparison_df, evaluation_report = evaluate_all_models(
            train_data, test_data, movies, baseline_models, neural_models
        )
        
        # Generate final report
        final_report = generate_final_report(comparison_df, start_time)
        
        # Demo recommendations
        demo_recommendations(train_data, movies, baseline_models, neural_models, user_to_idx, item_to_idx)
        
        print("\n" + "=" * 60)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Files generated:")
        print("- model_results.csv")
        print("- model_comparison.png")
        print("- evaluation_report.txt")
        print("- final_project_report.txt")
        print("\nTo run interactive demo: python recommendation_demo.py")
        print("\nNote: This version works without scikit-surprise")
        print("For full collaborative filtering, install Visual C++ Build Tools")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 