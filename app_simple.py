#!/usr/bin/env python3
"""
Movie Recommendation System - Simplified Web Frontend (No PyTorch)
Flask web application for interactive movie recommendations without neural models
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules (excluding neural models)
from data_loader import MovieLensDataLoader
from improved_baseline_models import ImprovedBaselineModels, ImprovedPopularityRecommender, ImprovedContentBasedRecommender
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

        models.clear()

        # Improved baseline models
        print("Training improved baseline models...")
        baselines = ImprovedBaselineModels(train_data, movies_df)
        baselines.fit_global_average()
        baselines.fit_popular_movies()
        models['Improved Global Average'] = baselines

        # Improved popularity recommender
        print("Training improved popularity model...")
        popularity_model = ImprovedPopularityRecommender(train_data)
        popularity_model.fit()
        models['Improved Popularity'] = popularity_model

        # Improved content-based
        print("Training improved content-based model...")
        content_model = ImprovedContentBasedRecommender(train_data, movies_df)
        content_model.fit()
        models['Improved Content-Based'] = content_model

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
        if user_id not in user_to_idx:
            return jsonify({'error': 'User not found'}), 404

        recs = {}
        for model_name, model in models.items():
            try:
                if model_name == 'Improved Content-Based':
                    recs[model_name] = model.get_recommendations(user_id, n=10)

                elif model_name == 'Improved Popularity':
                    recs[model_name] = model.get_recommendations(user_id, n=10)

                elif model_name == 'Improved Global Average':
                    # For global average, recommend popular movies
                    rated = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'item_id'])
                    popular = model.get_popular_movies(10)
                    recs[model_name] = [m for m in popular if m not in rated]

            except Exception as e:
                print(f"⚠ Error in {model_name}: {e}")
                recs[model_name] = []

        result = {
            model_name: [
                {
                    'id': int(mid),
                    'title': movies_df.loc[movies_df['item_id'] == mid, 'title'].values[0],
                    'year': movies_df.loc[movies_df['item_id'] == mid, 'release_date'].values[0],
                }
                for mid in mids if not movies_df.loc[movies_df['item_id'] == mid].empty
            ]
            for model_name, mids in recs.items()
        }

        return jsonify(result)

    except Exception as e:
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
                if model_name == 'Improved Global Average':
                    # Evaluate improved global average
                    preds = []
                    actuals = []
                    for _, row in evaluator.test_data.iterrows():
                        pred = model.predict_rating(row['user_id'], row['item_id'])
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


# Data collection and conversational AI endpoints
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

        # Collect the rating
        data_collector.collect_rating(user_id, movie_id, rating)

        return jsonify({
            'message': 'Rating collected successfully',
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/collected-data-stats')
def collected_data_stats():
    """Get statistics about collected user data."""
    try:
        if not data_collector:
            return jsonify({'error': 'Data collector not initialized'}), 500

        stats = data_collector.get_data_stats()
        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Movie Recommendation System Web App (Simplified)...")
    if load_data_and_models():
        print("✅ Data/models ready. Visit http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load data/models. Check logs.")
