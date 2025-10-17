#!/usr/bin/env python3
"""
Movie Recommendation System - Web Frontend
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

        models.clear()

        # Baseline models
        print("Training baseline models...")
        baselines = BaselineModels(train_data, movies_df)
        baselines.fit_global_average()
        baselines.fit_popular_movies()
        models['Global Average'] = baselines
        models['Popular Movies'] = baselines

        # Content-based
        print("Training content-based model...")
        content_model = ContentBasedRecommender(train_data, movies_df)
        content_model.fit()
        models['Content-Based'] = content_model

        # Neural models
        print("Training neural models...")
        neural_rec = NeuralRecommender(train_data, test_data, user_to_idx, item_to_idx)

        mf = MatrixFactorization(len(user_to_idx), len(item_to_idx), n_factors=64)
        neural_rec.train_model("Matrix Factorization", mf, epochs=50, batch_size=256, lr=0.001)
        models["Matrix Factorization"] = neural_rec

        tt = TwoTowerModel(len(user_to_idx), len(item_to_idx), embedding_dim=64)
        neural_rec.train_model("Two-Tower", tt, epochs=40, batch_size=256, lr=0.001)
        models["Two-Tower"] = neural_rec

        nmf = NeuralMF(len(user_to_idx), len(item_to_idx), n_factors=64)
        neural_rec.train_model("Neural MF", nmf, epochs=45, batch_size=256, lr=0.001)
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
        # Check if user exists in the dataset
        if user_to_idx is None or user_id not in user_to_idx:
            return jsonify({'error': 'User not found. Please enter a valid User ID between 1 and 943.'}), 404

        recs = {}
        for model_name, model in models.items():
            try:
                if model_name == 'Content-Based':
                    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                    if not user_ratings.empty:
                        top_movies = user_ratings.nlargest(5, 'rating')['item_id'].tolist()
                        similar = []
                        for mid in top_movies:
                            similar.extend(model.get_similar_movies(mid, n=10))
                        similar = list(set(similar) - set(user_ratings['item_id']))
                        recs[model_name] = similar[:10]
                    else:
                        recs[model_name] = []

                elif model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
                    user_idx = user_to_idx[user_id]
                    rated = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'item_id'])
                    preds = []
                    for item_id in movies_df['item_id']:
                        if item_id not in rated and item_id in item_to_idx:
                            try:
                                pred = model.predict_rating(model_name, user_idx, item_to_idx[item_id])
                                if pred is not None and not np.isnan(pred):
                                    preds.append((item_id, pred))
                            except Exception as e:
                                print(f"Error predicting for item {item_id}: {e}")
                                continue
                    preds.sort(key=lambda x: x[1], reverse=True)
                    recs[model_name] = [iid for iid, _ in preds[:10]]

                elif model_name == 'Popular Movies':
                    rated = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'item_id'])
                    popular = model.get_popular_movies(10)
                    recs[model_name] = [m for m in popular if m not in rated]

            except Exception as e:
                print(f"⚠ Error in {model_name}: {e}")
                recs[model_name] = []

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
                if model_name in ['Matrix Factorization', 'Two-Tower', 'Neural MF']:
                    preds, actuals = [], []
                    for _, row in evaluator.test_data.iterrows():
                        u_idx = user_to_idx.get(row['user_id'])
                        i_idx = item_to_idx.get(row['item_id'])
                        if u_idx is not None and i_idx is not None:
                            preds.append(model.predict_rating(model_name, u_idx, i_idx))
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


# New API endpoints for data collection and conversational AI

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


@app.route('/api/interaction', methods=['POST'])
def record_interaction():
    """Record user interaction (view, click, search, etc.)."""
    try:
        if not data_collector:
            return jsonify({'error': 'Data collector not initialized'}), 500

        data = request.json
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        interaction_type = data.get('interaction_type')
        interaction_data = data.get('interaction_data')

        if not all([user_id, movie_id, interaction_type]):
            return jsonify({'error': 'user_id, movie_id, and interaction_type required'}), 400

        # Collect the interaction
        data_collector.collect_interaction(user_id, movie_id, interaction_type, interaction_data)

        return jsonify({
            'message': 'Interaction recorded successfully',
            'user_id': user_id,
            'movie_id': movie_id,
            'interaction_type': interaction_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback about a movie."""
    try:
        if not data_collector:
            return jsonify({'error': 'Data collector not initialized'}), 500

        data = request.json
        user_id = data.get('user_id')
        movie_id = data.get('movie_id')
        feedback_text = data.get('feedback_text')
        sentiment_score = data.get('sentiment_score')

        if not all([user_id, movie_id, feedback_text]):
            return jsonify({'error': 'user_id, movie_id, and feedback_text required'}), 400

        # Collect the feedback
        data_collector.collect_feedback(user_id, movie_id, feedback_text, sentiment_score)

        return jsonify({
            'message': 'Feedback submitted successfully',
            'user_id': user_id,
            'movie_id': movie_id
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


@app.route('/api/user/<int:user_id>')
def get_user_info(user_id):
    """Get basic user information."""
    try:
        if user_to_idx is None or user_id not in user_to_idx:
            return jsonify({'error': 'User not found'}), 404

        # Get user's rating count
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        total_ratings = len(user_ratings)

        return jsonify({
            'user_id': user_id,
            'total_ratings': total_ratings,
            'avg_rating': round(user_ratings['rating'].mean(), 2) if total_ratings > 0 else 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user-data/<int:user_id>')
def get_user_data(user_id):
    """Get collected data for a specific user."""
    try:
        if not data_collector:
            return jsonify({'error': 'Data collector not initialized'}), 500

        ratings = data_collector.get_user_ratings(user_id)
        interactions = data_collector.get_user_interactions(user_id)

        return jsonify({
            'user_id': user_id,
            'ratings': ratings.to_dict('records') if not ratings.empty else [],
            'interactions': interactions.to_dict('records') if not interactions.empty else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Movie Recommendation System Web App...")
    if load_data_and_models():
        print("✅ Data/models ready. Visit http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load data/models. Check logs.")
