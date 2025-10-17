import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

class BaselineModels:
    def __init__(self, train_data, movies):
        self.train_data = train_data
        self.movies = movies
        self.global_mean = None
        self.movie_stats = None
        self.popular_movies = None
        self.genre_similarity = None
        
    def fit_global_average(self):
        """Fit global average baseline"""
        self.global_mean = self.train_data.rating.mean()
        print(f"Global average rating: {self.global_mean:.3f}")
        return self.global_mean
    
    def fit_popular_movies(self, min_ratings=10):
        """Fit popular movies baseline"""
        # Calculate movie statistics
        self.movie_stats = self.train_data.groupby('item_id').agg({
            'rating': ['mean', 'count']
        }).round(3)
        self.movie_stats.columns = ['avg_rating', 'num_ratings']
        
        # Get popular movies (minimum ratings threshold)
        self.popular_movies = self.movie_stats[
            self.movie_stats.num_ratings >= min_ratings
        ].sort_values('avg_rating', ascending=False)
        
        print(f"Popular movies baseline: {len(self.popular_movies)} movies with ≥{min_ratings} ratings")
        return self.popular_movies
    
    def fit_content_based(self):
        """Fit content-based filtering using movie genres"""
        # Create genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        if not genre_cols:
            # Fallback: create dummy genres if no genre columns exist
            self.movies['genres'] = 'unknown'
        else:
            self.movies['genres'] = self.movies[genre_cols].apply(
                lambda x: ' '.join([str(i) for i, val in enumerate(x) if val == 1]), axis=1
            )
        
        # TF-IDF on genres
        tfidf = TfidfVectorizer()
        genre_matrix = tfidf.fit_transform(self.movies.genres.fillna(''))
        self.genre_similarity = cosine_similarity(genre_matrix)
        
        print(f"Content-based model: {self.genre_similarity.shape[0]} movies, {len(genre_cols)} genres")
        return self.genre_similarity
    
    def predict_global_average(self, user_ids, item_ids):
        """Predict using global average"""
        return np.full(len(user_ids), self.global_mean)
    
    def predict_popular_movies(self, user_ids, item_ids, n=10):
        """Get top-N popular movie recommendations"""
        return self.popular_movies.head(n).index.tolist()
    
    def predict_content_based(self, user_id, n=10):
        """Get content-based recommendations for a user"""
        if self.genre_similarity is None:
            self.fit_content_based()
        
        # Get user's rated movies
        user_ratings = self.train_data[self.train_data.user_id == user_id]
        
        if len(user_ratings) == 0:
            # Cold start: return popular movies
            return self.popular_movies.head(n).index.tolist()
        
        # Create user profile from rated movies
        user_profile = np.zeros(self.genre_similarity.shape[0])
        for _, row in user_ratings.iterrows():
            movie_idx = row.item_id - 1  # Convert to 0-indexed
            if movie_idx < len(user_profile):
                user_profile += self.genre_similarity[movie_idx] * row.rating
        
        # Normalize by number of ratings
        user_profile /= len(user_ratings)
        
        # Find similar movies (exclude already rated)
        rated_movies = set(user_ratings.item_id)
        similarities = user_profile
        
        # Get top similar movies not rated by user
        movie_scores = [(i+1, similarities[i]) for i in range(len(similarities)) 
                       if i+1 not in rated_movies]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n]]
    
    def evaluate_baselines(self, test_data):
        """Evaluate baseline models"""
        results = {}
        
        # Global average
        global_preds = self.predict_global_average(
            test_data.user_id.values, test_data.item_id.values
        )
        global_rmse = math.sqrt(mean_squared_error(test_data.rating, global_preds))
        global_mae = mean_absolute_error(test_data.rating, global_preds)
        
        results['Global_Average'] = {
            'RMSE': global_rmse,
            'MAE': global_mae
        }
        
        print(f"Global Average - RMSE: {global_rmse:.3f}, MAE: {global_mae:.3f}")
        
        return results
    
    def predict_rating(self, user_id, item_id):
        """Predict rating for a user-item pair using global average"""
        return self.global_mean
    
    def get_popular_movies(self, n=10):
        """Get top-N popular movies"""
        if self.popular_movies is not None:
            return self.popular_movies.head(n).index.tolist()
        return []

class PopularityRecommender:
    """Simple popularity-based recommender"""
    
    def __init__(self, train_data):
        self.train_data = train_data
        self.movie_popularity = None
        
    def fit(self):
        """Calculate movie popularity scores"""
        # Count ratings per movie
        self.movie_popularity = self.train_data.item_id.value_counts()
        print(f"Fitted popularity model for {len(self.movie_popularity)} movies")
        
    def recommend(self, user_id, n=10):
        """Recommend top-N most popular movies"""
        if self.movie_popularity is None:
            self.fit()
        
        # Get movies user hasn't rated
        user_movies = set(self.train_data[self.train_data.user_id == user_id].item_id)
        all_movies = set(self.movie_popularity.index)
        unseen_movies = list(all_movies - user_movies)
        
        # Sort by popularity
        unseen_popularity = self.movie_popularity[unseen_movies].sort_values(ascending=False)
        return unseen_popularity.head(n).index.tolist()

class ContentBasedRecommender:
    """Content-based recommender using movie genres"""
    
    def __init__(self, train_data, movies):
        self.train_data = train_data
        self.movies = movies
        self.genre_similarity = None
        self.tfidf = None
        
    def fit(self):
        """Fit the content-based model"""
        # Create genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        self.movies['genres'] = self.movies[genre_cols].apply(
            lambda x: ' '.join([str(i) for i, val in enumerate(x) if val == 1]), axis=1
        )
        
        # TF-IDF on genres
        self.tfidf = TfidfVectorizer()
        genre_matrix = self.tfidf.fit_transform(self.movies.genres.fillna(''))
        self.genre_similarity = cosine_similarity(genre_matrix)
        
        print(f"Fitted content-based model: {self.genre_similarity.shape[0]} movies")
        
    def recommend(self, user_id, n=10):
        """Get content-based recommendations for a user"""
        if self.genre_similarity is None:
            self.fit()
        
        # Get user's rated movies
        user_ratings = self.train_data[self.train_data.user_id == user_id]
        
        if len(user_ratings) == 0:
            # Cold start: return empty list
            return []
        
        # Create user profile
        user_profile = np.zeros(self.genre_similarity.shape[0])
        for _, row in user_ratings.iterrows():
            movie_idx = row.item_id - 1
            if movie_idx < len(user_profile):
                user_profile += self.genre_similarity[movie_idx] * row.rating
        
        # Normalize
        user_profile /= len(user_ratings)
        
        # Get recommendations
        rated_movies = set(user_ratings.item_id)
        movie_scores = []
        
        for i in range(len(user_profile)):
            movie_id = i + 1
            if movie_id not in rated_movies:
                movie_scores.append((movie_id, user_profile[i]))
        
        # Sort by similarity score
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in movie_scores[:n]]
    
    def get_similar_movies(self, movie_id, n=10):
        """Get similar movies for a given movie"""
        if self.genre_similarity is None:
            self.fit()
        
        # Get movie index (assuming 1-based indexing)
        movie_idx = movie_id - 1
        if movie_idx < 0 or movie_idx >= len(self.genre_similarity):
            return []
        
        # Get similarities for this movie
        similarities = self.genre_similarity[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        movie_scores = [(i+1, similarities[i]) for i in range(len(similarities)) if i+1 != movie_id]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n]]

if __name__ == "__main__":
    # Test baseline models
    from data_loader import MovieLensDataLoader
    
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    train_data, test_data = loader.split_data()
    
    baselines = BaselineModels(train_data, loader.movies)
    baselines.fit_global_average()
    baselines.fit_popular_movies()
    baselines.fit_content_based()
    
    results = baselines.evaluate_baselines(test_data) 