import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

class ImprovedBaselineModels:
    def __init__(self, train_data, movies):
        self.train_data = train_data
        self.movies = movies
        self.global_mean = None
        self.movie_stats = None
        self.popular_movies = None
        self.genre_similarity = None
        self.user_biases = None
        self.movie_biases = None
        
    def fit_global_average(self):
        """Fit improved global average with user and movie biases"""
        self.global_mean = self.train_data.rating.mean()
        
        # Calculate user biases (how much each user deviates from global mean)
        user_ratings = self.train_data.groupby('user_id')['rating'].agg(['mean', 'count'])
        user_ratings.columns = ['avg_rating', 'num_ratings']
        self.user_biases = (user_ratings['avg_rating'] - self.global_mean).to_dict()
        
        # Calculate movie biases (how much each movie deviates from global mean)
        movie_ratings = self.train_data.groupby('item_id')['rating'].agg(['mean', 'count'])
        movie_ratings.columns = ['avg_rating', 'num_ratings']
        self.movie_biases = (movie_ratings['avg_rating'] - self.global_mean).to_dict()
        
        print(f"Global average rating: {self.global_mean:.3f}")
        print(f"User biases calculated for {len(self.user_biases)} users")
        print(f"Movie biases calculated for {len(self.movie_biases)} movies")
        return self.global_mean
    
    def fit_popular_movies(self, min_ratings=10):
        """Fit improved popular movies with weighted scoring"""
        # Calculate movie statistics with better scoring
        movie_stats = self.train_data.groupby('item_id').agg({
            'rating': ['mean', 'count', 'std']
        }).round(3)
        movie_stats.columns = ['avg_rating', 'num_ratings', 'rating_std']
        
        # Calculate weighted score: (avg_rating * num_ratings) / (1 + rating_std)
        # This balances popularity with consistency
        movie_stats['weighted_score'] = (
            movie_stats['avg_rating'] * movie_stats['num_ratings']
        ) / (1 + movie_stats['rating_std'].fillna(0))
        
        # Get popular movies with minimum ratings threshold
        self.popular_movies = movie_stats[
            movie_stats.num_ratings >= min_ratings
        ].sort_values('weighted_score', ascending=False)
        
        print(f"Popular movies baseline: {len(self.popular_movies)} movies with ≥{min_ratings} ratings")
        return self.popular_movies
    
    def fit_content_based(self):
        """Fit improved content-based filtering with better genre handling"""
        # Create better genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        
        # Create genre names mapping
        genre_names = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Create genre strings with proper names
        def create_genre_string(row):
            genres = []
            for i, val in enumerate(row):
                if val == 1 and i < len(genre_names):
                    genres.append(genre_names[i])
            return ' '.join(genres)
        
        self.movies['genres'] = self.movies[genre_cols].apply(create_genre_string, axis=1)
        
        # TF-IDF on genres with better parameters
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2
        )
        genre_matrix = self.tfidf.fit_transform(self.movies.genres.fillna(''))
        self.genre_similarity = cosine_similarity(genre_matrix)
        
        print(f"Content-based model: {self.genre_similarity.shape[0]} movies, {len(genre_names)} genres")
        return self.genre_similarity
    
    def predict_rating(self, user_id, item_id):
        """Predict rating using improved global average with biases"""
        base_rating = self.global_mean
        
        # Add user bias
        user_bias = self.user_biases.get(user_id, 0)
        
        # Add movie bias
        movie_bias = self.movie_biases.get(item_id, 0)
        
        # Combine biases (with some damping to prevent extreme values)
        predicted_rating = base_rating + 0.7 * user_bias + 0.7 * movie_bias
        
        # Clamp to valid rating range
        return max(1.0, min(5.0, predicted_rating))
    
    def get_popular_movies(self, n=10):
        """Get top-N popular movies"""
        if self.popular_movies is None:
            self.fit_popular_movies()
        return self.popular_movies.head(n).index.tolist()
    
    def get_similar_movies(self, movie_id, n=10):
        """Get similar movies using content-based filtering"""
        if self.genre_similarity is None:
            self.fit_content_based()
        
        # Get movie index (convert to 0-indexed)
        movie_idx = movie_id - 1
        if movie_idx >= len(self.genre_similarity):
            return []
        
        # Get similarities
        similarities = self.genre_similarity[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        movie_scores = [(i+1, similarities[i]) for i in range(len(similarities)) 
                       if i+1 != movie_id]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n]]
    
    def evaluate_baselines(self, test_data):
        """Evaluate baseline models with better metrics"""
        results = {}
        
        # Improved global average
        preds = []
        actuals = []
        
        for _, row in test_data.iterrows():
            pred = self.predict_rating(row['user_id'], row['item_id'])
            preds.append(pred)
            actuals.append(row['rating'])
        
        rmse = math.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        
        results['Improved_Global_Average'] = {
            'RMSE': rmse,
            'MAE': mae
        }
        
        print(f"Improved Global Average - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        
        return results

class ImprovedPopularityRecommender:
    """Improved popularity-based recommender with better scoring"""
    
    def __init__(self, train_data):
        self.train_data = train_data
        self.movie_scores = None
        
    def fit(self):
        """Calculate improved movie popularity scores"""
        # Calculate movie statistics
        movie_stats = self.train_data.groupby('item_id').agg({
            'rating': ['mean', 'count', 'std']
        })
        movie_stats.columns = ['avg_rating', 'num_ratings', 'rating_std']
        
        # Calculate weighted score that balances rating and popularity
        # Use Bayesian average: (avg_rating * num_ratings + global_mean * min_ratings) / (num_ratings + min_ratings)
        global_mean = self.train_data.rating.mean()
        min_ratings = 5
        
        movie_stats['bayesian_score'] = (
            movie_stats['avg_rating'] * movie_stats['num_ratings'] + 
            global_mean * min_ratings
        ) / (movie_stats['num_ratings'] + min_ratings)
        
        # Additional penalty for high variance (inconsistent ratings)
        movie_stats['consistency_penalty'] = 1 / (1 + movie_stats['rating_std'].fillna(0))
        movie_stats['final_score'] = movie_stats['bayesian_score'] * movie_stats['consistency_penalty']
        
        self.movie_scores = movie_stats.sort_values('final_score', ascending=False)
        
        print(f"Fitted improved popularity model for {len(self.movie_scores)} movies")
        
    def get_recommendations(self, user_id, n=10):
        """Get top-N recommendations for a user"""
        if self.movie_scores is None:
            self.fit()
        
        # Get movies user hasn't rated
        user_movies = set(self.train_data[self.train_data.user_id == user_id].item_id)
        all_movies = set(self.movie_scores.index)
        unseen_movies = list(all_movies - user_movies)
        
        # Get top unseen movies
        unseen_scores = self.movie_scores.loc[unseen_movies].sort_values('final_score', ascending=False)
        return unseen_scores.head(n).index.tolist()

class ImprovedContentBasedRecommender:
    """Improved content-based recommender with better genre handling"""
    
    def __init__(self, train_data, movies):
        self.train_data = train_data
        self.movies = movies
        self.genre_similarity = None
        self.tfidf = None
        
    def fit(self):
        """Fit the improved content-based model"""
        # Create better genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        
        # Create genre names mapping
        genre_names = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Create genre strings with proper names
        def create_genre_string(row):
            genres = []
            for i, val in enumerate(row):
                if val == 1 and i < len(genre_names):
                    genres.append(genre_names[i])
            return ' '.join(genres)
        
        self.movies['genres'] = self.movies[genre_cols].apply(create_genre_string, axis=1)
        
        # TF-IDF with better parameters
        self.tfidf = TfidfVectorizer(
            max_features=200,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        genre_matrix = self.tfidf.fit_transform(self.movies.genres.fillna(''))
        self.genre_similarity = cosine_similarity(genre_matrix)
        
        print(f"Fitted improved content-based model: {self.genre_similarity.shape[0]} movies")
        
    def get_similar_movies(self, movie_id, n=10):
        """Get similar movies for a given movie"""
        if self.genre_similarity is None:
            self.fit()
        
        # Get movie index (convert to 0-indexed)
        movie_idx = movie_id - 1
        if movie_idx >= len(self.genre_similarity):
            return []
        
        # Get similarities
        similarities = self.genre_similarity[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        movie_scores = [(i+1, similarities[i]) for i in range(len(similarities)) 
                       if i+1 != movie_id]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n]]
    
    def get_recommendations(self, user_id, n=10):
        """Get content-based recommendations for a user"""
        if self.genre_similarity is None:
            self.fit()
        
        # Get user's rated movies
        user_ratings = self.train_data[self.train_data.user_id == user_id]
        
        if len(user_ratings) == 0:
            # Cold start: return empty list
            return []
        
        # Create user profile with weighted ratings
        user_profile = np.zeros(self.genre_similarity.shape[0])
        total_weight = 0
        
        for _, row in user_ratings.iterrows():
            movie_idx = row.item_id - 1
            if movie_idx < len(user_profile):
                # Weight by rating (higher ratings have more influence)
                weight = (row.rating - 2.5) / 2.5  # Normalize to [-1, 1]
                user_profile += self.genre_similarity[movie_idx] * weight
                total_weight += abs(weight)
        
        # Normalize by total weight
        if total_weight > 0:
            user_profile /= total_weight
        
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

if __name__ == "__main__":
    # Test improved baseline models
    from data_loader import MovieLensDataLoader
    
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    train_data, test_data = loader.split_data()
    
    baselines = ImprovedBaselineModels(train_data, loader.movies)
    baselines.fit_global_average()
    baselines.fit_popular_movies()
    baselines.fit_content_based()
    
    results = baselines.evaluate_baselines(test_data)
    print("Improved baseline results:", results)
