#!/usr/bin/env python3
"""
Minimal test app to debug the user validation issue
"""

from flask import Flask, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataLoader

app = Flask(__name__)

# Global variables
user_to_idx = None
movies_df = None

def load_data():
    global user_to_idx, movies_df
    print("Loading data...")
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    
    user_to_idx = loader.user_to_idx
    movies_df = loader.movies
    
    print(f"Data loaded. Users: {len(user_to_idx)}")
    print(f"User 790 exists: {790 in user_to_idx}")
    print(f"User range: {min(user_to_idx.keys())} to {max(user_to_idx.keys())}")

@app.route('/api/test-user/<int:user_id>')
def test_user(user_id):
    print(f"Testing user {user_id}")
    print(f"user_to_idx is None: {user_to_idx is None}")
    
    if user_to_idx is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    print(f"User {user_id} in user_to_idx: {user_id in user_to_idx}")
    print(f"User range: {min(user_to_idx.keys())} to {max(user_to_idx.keys())}")
    
    if user_id not in user_to_idx:
        return jsonify({
            'error': f'User {user_id} not found. Valid range: {min(user_to_idx.keys())} to {max(user_to_idx.keys())}'
        }), 404
    
    return jsonify({
        'success': True,
        'user_id': user_id,
        'user_index': user_to_idx[user_id],
        'total_users': len(user_to_idx)
    })

if __name__ == '__main__':
    print("Starting minimal test app...")
    load_data()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
