import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class MovieLensDataLoader:
    def __init__(self, data_dir="movie/ml-100k"):
        self.data_dir = data_dir
        self.ratings = None
        self.movies = None
        self.users = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load all MovieLens 100K data files"""
        print("Loading MovieLens 100K dataset...")
        
        # Load ratings data
        self.ratings = pd.read_csv(
            os.path.join(self.data_dir, 'u.data'), 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # Load movie metadata
        self.movies = pd.read_csv(
            os.path.join(self.data_dir, 'u.item'), 
            sep='|', 
            encoding='latin-1',
            names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                  [f'genre_{i}' for i in range(19)]
        )
        
        # Load user metadata
        self.users = pd.read_csv(
            os.path.join(self.data_dir, 'u.user'), 
            sep='|', 
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print(f"Loaded {len(self.ratings)} ratings from {self.ratings.user_id.nunique()} users")
        print(f"and {self.ratings.item_id.nunique()} movies")
        
        return self.ratings, self.movies, self.users
    
    def explore_data(self):
        """Explore and display dataset statistics"""
        if self.ratings is None:
            self.load_data()
        
        print("\n=== Dataset Statistics ===")
        print(f"Total ratings: {len(self.ratings)}")
        print(f"Unique users: {self.ratings.user_id.nunique()}")
        print(f"Unique movies: {self.ratings.item_id.nunique()}")
        
        # Calculate sparsity
        total_possible_ratings = self.ratings.user_id.nunique() * self.ratings.item_id.nunique()
        sparsity = 1 - len(self.ratings) / total_possible_ratings
        print(f"Sparsity: {sparsity:.3f}")
        
        # Rating distribution
        print(f"\nRating distribution:")
        rating_counts = self.ratings.rating.value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  {rating} stars: {count} ratings ({count/len(self.ratings)*100:.1f}%)")
        
        # Average rating
        print(f"\nAverage rating: {self.ratings.rating.mean():.2f}")
        print(f"Rating std: {self.ratings.rating.std():.2f}")
        
        # User activity
        user_activity = self.ratings.user_id.value_counts()
        print(f"\nUser activity:")
        print(f"  Most active user: {user_activity.max()} ratings")
        print(f"  Least active user: {user_activity.min()} ratings")
        print(f"  Average ratings per user: {user_activity.mean():.1f}")
        
        # Movie popularity
        movie_popularity = self.ratings.item_id.value_counts()
        print(f"\nMovie popularity:")
        print(f"  Most popular movie: {movie_popularity.max()} ratings")
        print(f"  Least popular movie: {movie_popularity.min()} ratings")
        print(f"  Average ratings per movie: {movie_popularity.mean():.1f}")
    
    def clean_data(self, min_ratings=5):
        """Clean data by filtering users and movies with minimum interactions"""
        if self.ratings is None:
            self.load_data()
        
        print(f"\nCleaning data (minimum {min_ratings} ratings per user/movie)...")
        
        # Remove duplicates
        initial_size = len(self.ratings)
        self.ratings = self.ratings.drop_duplicates()
        print(f"Removed {initial_size - len(self.ratings)} duplicate ratings")
        
        # Filter users with minimum ratings
        user_counts = self.ratings.user_id.value_counts()
        active_users = user_counts[user_counts >= min_ratings].index
        print(f"Filtered to {len(active_users)} active users (≥{min_ratings} ratings)")
        
        # Filter movies with minimum ratings
        movie_counts = self.ratings.item_id.value_counts()
        popular_movies = movie_counts[movie_counts >= min_ratings].index
        print(f"Filtered to {len(popular_movies)} popular movies (≥{min_ratings} ratings)")
        
        # Apply filters
        self.ratings = self.ratings[
            (self.ratings.user_id.isin(active_users)) & 
            (self.ratings.item_id.isin(popular_movies))
        ]
        
        print(f"Final dataset: {len(self.ratings)} ratings")
        print(f"Users: {self.ratings.user_id.nunique()}, Movies: {self.ratings.item_id.nunique()}")
        
        # Create user and item mappings for neural models
        self.user_to_idx, self.item_to_idx, self.idx_to_user, self.idx_to_item = self.get_user_item_mapping()
    
    def split_data(self, test_size=0.2, temporal=True):
        """Split data into train and test sets"""
        if self.ratings is None:
            self.load_data()
        
        print(f"\nSplitting data (test_size={test_size})...")
        
        if temporal:
            # Temporal split (sort by timestamp)
            self.ratings_sorted = self.ratings.sort_values('timestamp')
            split_idx = int((1 - test_size) * len(self.ratings_sorted))
            self.train_data = self.ratings_sorted.iloc[:split_idx]
            self.test_data = self.ratings_sorted.iloc[split_idx:]
            print("Using temporal split (chronological order)")
        else:
            # Random split
            self.train_data, self.test_data = train_test_split(
                self.ratings, test_size=test_size, random_state=42
            )
            print("Using random split")
        
        print(f"Train set: {len(self.train_data)} ratings")
        print(f"Test set: {len(self.test_data)} ratings")
        
        return self.train_data, self.test_data
    
    def get_movie_features(self):
        """Extract movie features for content-based filtering"""
        if self.movies is None:
            self.load_data()
        
        # Create genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        self.movies['genres'] = self.movies[genre_cols].apply(
            lambda x: ' '.join([str(i) for i, val in enumerate(x) if val == 1]), axis=1
        )
        
        # Create genre matrix
        genre_matrix = self.movies[genre_cols].values
        
        return self.movies, genre_matrix
    
    def get_user_item_mapping(self):
        """Create user and item ID mappings for neural models"""
        if self.ratings is None:
            self.load_data()
        
        # Create mappings (0-indexed for neural models)
        unique_users = sorted(self.ratings.user_id.unique())
        unique_items = sorted(self.ratings.item_id.unique())
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
        
        return user_to_idx, item_to_idx, idx_to_user, idx_to_item

if __name__ == "__main__":
    # Test the data loader
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.explore_data()
    loader.clean_data(min_ratings=5)
    loader.split_data(test_size=0.2, temporal=True) 