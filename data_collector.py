#!/usr/bin/env python3
"""
Real-time Data Collection System for Movie Recommendations
Collects user interactions, ratings, and feedback for continuous learning
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class DataCollector:
    def __init__(self, db_path="user_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                rating REAL,
                timestamp DATETIME,
                source TEXT DEFAULT 'user_input'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                interaction_type TEXT,
                interaction_data TEXT,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                movie_id INTEGER,
                feedback_text TEXT,
                sentiment_score REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_start DATETIME,
                session_end DATETIME,
                total_interactions INTEGER,
                session_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
    
    def collect_rating(self, user_id, movie_id, rating, source="user_input"):
        """Collect user rating for a movie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_ratings (user_id, movie_id, rating, timestamp, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, movie_id, rating, datetime.now(), source))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Collected rating: User {user_id} rated Movie {movie_id} as {rating}")
    
    def collect_interaction(self, user_id, movie_id, interaction_type, interaction_data=None):
        """Collect user interaction (view, click, search, etc.)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(interaction_data) if interaction_data else None
        
        cursor.execute('''
            INSERT INTO user_interactions (user_id, movie_id, interaction_type, interaction_data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, movie_id, interaction_type, data_json, datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"🔄 Collected interaction: User {user_id} {interaction_type} Movie {movie_id}")
    
    def collect_feedback(self, user_id, movie_id, feedback_text, sentiment_score=None):
        """Collect user feedback and sentiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (user_id, movie_id, feedback_text, sentiment_score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, movie_id, feedback_text, sentiment_score, datetime.now()))
        
        conn.commit()
        conn.close()
        
        print(f"💬 Collected feedback: User {user_id} feedback for Movie {movie_id}")
    
    def start_session(self, user_id):
        """Start a new user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_start, total_interactions)
            VALUES (?, ?, ?)
        ''', (user_id, datetime.now(), 0))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id, total_interactions, session_data=None):
        """End a user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(session_data) if session_data else None
        
        cursor.execute('''
            UPDATE user_sessions 
            SET session_end = ?, total_interactions = ?, session_data = ?
            WHERE id = ?
        ''', (datetime.now(), total_interactions, data_json, session_id))
        
        conn.commit()
        conn.close()
    
    def get_user_ratings(self, user_id=None):
        """Get user ratings from collected data"""
        conn = sqlite3.connect(self.db_path)
        
        if user_id:
            query = "SELECT * FROM user_ratings WHERE user_id = ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:
            query = "SELECT * FROM user_ratings ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_user_interactions(self, user_id=None):
        """Get user interactions from collected data"""
        conn = sqlite3.connect(self.db_path)
        
        if user_id:
            query = "SELECT * FROM user_interactions WHERE user_id = ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(user_id,))
        else:
            query = "SELECT * FROM user_interactions ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_combined_data(self):
        """Get all collected data combined with original dataset"""
        # Get original MovieLens data
        from data_loader import MovieLensDataLoader
        loader = MovieLensDataLoader()
        original_ratings, movies, users = loader.load_data()
        
        # Get collected ratings
        collected_ratings = self.get_user_ratings()
        
        if not collected_ratings.empty:
            # Combine with original data
            combined_ratings = pd.concat([
                original_ratings[['user_id', 'item_id', 'rating', 'timestamp']],
                collected_ratings[['user_id', 'movie_id', 'rating', 'timestamp']].rename(columns={'movie_id': 'item_id'})
            ], ignore_index=True)
            
            # Remove duplicates
            combined_ratings = combined_ratings.drop_duplicates(subset=['user_id', 'item_id'])
            
            print(f"📈 Combined dataset: {len(combined_ratings)} total ratings")
            print(f"   - Original: {len(original_ratings)} ratings")
            print(f"   - Collected: {len(collected_ratings)} ratings")
            
            return combined_ratings, movies, users
        else:
            print("📊 No collected data yet, using original dataset")
            return original_ratings, movies, users
    
    def get_data_stats(self):
        """Get statistics about collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM user_ratings")
        total_ratings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_ratings")
        unique_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT movie_id) FROM user_ratings")
        unique_movies = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(rating) FROM user_ratings")
        avg_rating = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            'total_ratings': total_ratings,
            'unique_users': unique_users,
            'unique_movies': unique_movies,
            'avg_rating': round(avg_rating, 2) if avg_rating else 0
        }
        
        return stats

if __name__ == "__main__":
    # Test the data collector
    collector = DataCollector()
    
    # Test collecting some data
    collector.collect_rating(1, 1, 5.0)
    collector.collect_rating(1, 2, 4.0)
    collector.collect_interaction(1, 1, "view")
    collector.collect_feedback(1, 1, "Great movie!", 0.8)
    
    # Get stats
    stats = collector.get_data_stats()
    print(f"📊 Collected data stats: {stats}")
