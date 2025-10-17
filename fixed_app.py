#!/usr/bin/env python3
"""
Fixed Movie Recommendation System - Web Frontend
Flask web application for interactive movie recommendations
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, ContentBasedRecommender
from neural_models import NeuralRecommender, MatrixFactorization, TwoTowerModel, NeuralMF
from evaluation import RecommendationEvaluator
from data_collector import DataCollector
from conversational_agent import ConversationalAgent

app = Flask(__name__)
app.secret_key = 'movie_recommendation_system_2024'

# Globals
data_loader = None
models = {}
evaluator = None
movies_df = None
ratings_df = None
user_to_idx = None
item_to_idx = None
data_collector = None
conversational_agent = None

def load_data_and_models():
    """Load MovieLens data, initialize and train models."""
    global data_loader, models, evaluator, movies_df, ratings_df, user_to_idx, item_to_idx, data_collector, conversational_agent

    try:
        print("Loading data and initializing models...")
        data_loader = MovieLensDataLoader()
        data_loader.load_data()
        data_loader.clean_data(min_ratings=5)
        train_data, test_data = data_loader.split_data()

        movies_df = data_loader.movies
        ratings_df = data_loader.ratings
        user_to_idx = data_loader.user_to_idx
        item_to_idx = data_loader.item_to_idx

        print(f"Data loaded: {len(movies_df)} movies, {len(ratings_df)} ratings")
        print(f"Users: {len(user_to_idx)}, Movies: {len(item_to_idx)}")

        models.clear()

        # Baseline models
        print("Training baseline models...")
        baselines = BaselineModels(train_data, movies_df)
        baselines.fit_global_average()
        baselines.fit_popular_movies()
        baselines.fit_content_based()
        models['Global Average'] = baselines
        models['Popular Movies'] = baselines

        # Content-based
        print("Training content-based model...")
        content_model = ContentBasedRecommender(train_data, movies_df)
        content_model.fit()
        models['Content-Based'] = content_model

        # Neural models - with better settings
        print("Training neural models...")
        neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)

        mf = MatrixFactorization(len(user_to_idx), len(item_to_idx), n_factors=32)
        neural_rec.train_model("Matrix Factorization", mf, epochs=20, batch_size=512, lr=0.01)
        models["Matrix Factorization"] = neural_rec

        tt = TwoTowerModel(len(user_to_idx), len(item_to_idx), embedding_dim=32)
        neural_rec.train_model("Two-Tower", tt, epochs=15, batch_size=512, lr=0.01)
        models["Two-Tower"] = neural_rec

        nmf = NeuralMF(len(user_to_idx), len(item_to_idx), n_factors=32)
        neural_rec.train_model("Neural MF", nmf, epochs=18, batch_size=512, lr=0.01)
        models["Neural MF"] = neural_rec

        evaluator = RecommendationEvaluator(test_data, movies_df)

        # Initialize data collector and conversational agent
        print("Initializing data collection system...")
        data_collector = DataCollector()
        
        print("Initializing conversational AI agent...")
        conversational_agent = ConversationalAgent(models, data_collector, movies_df)

        print("✅ All models trained successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading data/models: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html', models=list(models.keys()))


@app.route('/compare')
def compare():
    return render_template('compare.html', models=list(models.keys()))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/api/movies')
def api_movies():
    """Search movies (or return top 50)."""
    try:
        query = request.args.get('q', '').lower()
        filtered_movies = (
            movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
            if query else movies_df.head(50)
        )

        return jsonify([
            {
                'id': int(row['item_id']),
                'title': row['title'],
                'year': row.get('release_date', 'Unknown'),
                'genres': row.get('genres', 'Unknown')
            }
            for _, row in filtered_movies.iterrows()
        ])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """Return recommendations for a given user."""
    try:
        print(f"Getting recommendations for user {user_id}")
        
        # Check if user exists in the dataset
        if user_to_idx is None or user_id not in user_to_idx:
            print(f"User {user_id} not found in user_to_idx")
            return jsonify({'error': f'User {user_id} not found. Please enter a valid User ID between 1 and {max(user_to_idx.keys()) if user_to_idx else 943}.'}), 404

        print(f"User {user_id} found, generating recommendations...")
        
        recs = {}
        
        # Get user's rated movies for filtering
        user_rated_movies = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'].tolist())
        print(f"User {user_id} has rated {len(user_rated_movies)} movies")
        
        for model_name, model in models.items():
            try:
                print(f"Processing {model_name}...")
                
                if model_name == 'Content-Based':
                    # Get content-based recommendations
                    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                    if not user_ratings.empty:
                        top_movies = user_ratings.nlargest(5, 'rating')['item_id'].tolist()
                        similar = []
                        for mid in top_movies:
                            similar_movies = model.get_similar_movies(mid, n=5)
                            similar.extend(similar_movies)
                        # Remove already rated movies
                        similar = [m for m in similar if m not in user_rated_movies]
                        recs[model_name] = similar[:10]
                    else:
                        recs[model_name] = []

                elif model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
                    # Get neural network recommendations
                    user_idx = user_to_idx[user_id]
                    preds = []
                    
                    # Sample movies for faster prediction (instead of all movies)
                    sample_movies = movies_df['item_id'].sample(min(500, len(movies_df))).tolist()
                    
                    for item_id in sample_movies:
                        if item_id not in user_rated_movies and item_id in item_to_idx:
                            try:
                                pred = model.predict_rating(model_name, user_idx, item_to_idx[item_id])
                                if pred is not None and not np.isnan(pred) and pred > 0:
                                    preds.append((item_id, pred))
                            except Exception as e:
                                continue
                    
                    preds.sort(key=lambda x: x[1], reverse=True)
                    recs[model_name] = [iid for iid, _ in preds[:10]]

                elif model_name == 'Popular Movies':
                    # Get popular movies
                    popular = model.get_popular_movies(20)
                    recs[model_name] = [m for m in popular if m not in user_rated_movies][:10]

                elif model_name == 'Global Average':
                    # For global average, just return some popular movies
                    popular = models['Popular Movies'].get_popular_movies(20)
                    recs[model_name] = [m for m in popular if m not in user_rated_movies][:10]

                print(f"{model_name}: {len(recs.get(model_name, []))} recommendations")

            except Exception as e:
                print(f"⚠ Error in {model_name}: {e}")
                recs[model_name] = []

        # Convert movie IDs to movie details
        result = {}
        for model_name, mids in recs.items():
            result[model_name] = []
            for mid in mids:
                try:
                    movie_info = movies_df[movies_df['item_id'] == mid]
                    if not movie_info.empty:
                        movie_row = movie_info.iloc[0]
                        result[model_name].append({
                            'id': int(mid),
                            'title': str(movie_row.get('title', 'Unknown')),
                            'year': str(movie_row.get('release_date', 'Unknown')),
                            'genres': str(movie_row.get('genres', 'Unknown'))
                        })
                except Exception as e:
                    print(f"Error processing movie {mid}: {e}")
                    continue

        print(f"Final result: {sum(len(movies) for movies in result.values())} total recommendations")
        return jsonify(result)

    except Exception as e:
        print(f"Error in api_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Return dataset statistics."""
    try:
        if movies_df is None or ratings_df is None:
            return jsonify({'error': 'Data not loaded'}), 500

        stats = {
            'total_movies': len(movies_df),
            'total_users': len(ratings_df['user_id'].unique()),
            'total_ratings': len(ratings_df),
            'avg_rating': round(ratings_df['rating'].mean(), 2)
        }
        
        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-performance')
def api_model_performance():
    """Return model performance metrics."""
    try:
        if not evaluator:
            return jsonify({'error': 'Evaluator not initialized'}), 500

        performance = {}
        for model_name, model in models.items():
            try:
                if model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
                    preds, actuals = [], []
                    # Sample test data for faster evaluation
                    sample_test = evaluator.test_data.sample(min(1000, len(evaluator.test_data)))
                    for _, row in sample_test.iterrows():
                        u_idx = user_to_idx.get(row['user_id'])
                        i_idx = item_to_idx.get(row['item_id'])
                        if u_idx is not None and i_idx is not None:
                            pred = model.predict_rating(model_name, u_idx, i_idx)
                            if pred is not None and not np.isnan(pred):
                                preds.append(pred)
                                actuals.append(row['rating'])
                    if preds:
                        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))
                        mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
                        performance[model_name] = {'RMSE': round(rmse, 3), 'MAE': round(mae, 3)}
                    else:
                        performance[model_name] = {'RMSE': 'N/A', 'MAE': 'N/A'}

                else:
                    performance[model_name] = {'RMSE': 'N/A', 'MAE': 'N/A'}

            except Exception as e:
                performance[model_name] = {'RMSE': 'Error', 'MAE': 'Error', 'Error': str(e)}

        return jsonify(performance)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Data collection endpoints
@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    """Chat with the conversational AI agent."""
    try:
        if not conversational_agent:
            return jsonify({'error': 'Conversational agent not initialized'}), 500

        data = request.json
        user_id = data.get('user_id')
        message = data.get('message', '')

        if not user_id:
            return jsonify({'error': 'User ID required'}), 400

        if not message:
            return jsonify({'error': 'Message required'}), 400

        response = conversational_agent.process_message(user_id, message)
        
        return jsonify({
            'response': response,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rate-movie', methods=['POST'])
def rate_movie():
    """Rate a movie and collect the data."""
    try:
        if not data_collector:
            return jsonify({'error': 'Data collector not initialized'}), 500

        data = request.json
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        rating = data.get('rating')

        if not all([user_id, movie_id, rating]):
            return jsonify({'error': 'user_id, movie_id, and rating required'}), 400

        if not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400

        data_collector.collect_rating(user_id, movie_id, rating)

        return jsonify({
            'message': 'Rating collected successfully',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Fixed Movie Recommendation System Web App...")
    if load_data_and_models():
        print("✅ Data/models ready. Visit http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load data/models. Check logs.")
