import pandas as pd
import numpy as np
import torch
from data_loader import MovieLensDataLoader
from baseline_models import BaselineModels, PopularityRecommender, ContentBasedRecommender
from collaborative_filtering import CollaborativeFiltering
from neural_models import NeuralRecommender, MatrixFactorization
from evaluation import RecommendationEvaluator

class MovieRecommendationDemo:
    """Interactive movie recommendation demo system"""
    
    def __init__(self):
        self.loader = None
        self.train_data = None
        self.test_data = None
        self.movies = None
        self.models = {}
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        
    def setup_system(self):
        """Setup the complete recommendation system"""
        print("Setting up Movie Recommendation System...")
        print("=" * 50)
        
        # Load and prepare data
        self.loader = MovieLensDataLoader()
        self.loader.load_data()
        self.loader.explore_data()
        self.loader.clean_data(min_ratings=5)
        self.train_data, self.test_data = self.loader.split_data()
        self.movies = self.loader.movies
        
        # Get user-item mappings
        self.user_to_idx, self.item_to_idx, self.idx_to_user, self.idx_to_item = self.loader.get_user_item_mapping()
        
        print("\nData setup completed!")
        print(f"Users: {len(self.user_to_idx)}, Movies: {len(self.item_to_idx)}")
        
    def train_baseline_models(self):
        """Train baseline recommendation models"""
        print("\nTraining Baseline Models...")
        print("-" * 30)
        
        # Global average baseline
        baselines = BaselineModels(self.train_data, self.movies)
        baselines.fit_global_average()
        baselines.fit_popular_movies()
        baselines.fit_content_based()
        
        self.models['Global_Average'] = baselines
        self.models['Popular_Movies'] = baselines.popular_movies
        
        # Popularity recommender
        pop_rec = PopularityRecommender(self.train_data)
        pop_rec.fit()
        self.models['Popularity'] = pop_rec
        
        # Content-based recommender
        content_rec = ContentBasedRecommender(self.train_data, self.movies)
        content_rec.fit()
        self.models['Content_Based'] = content_rec
        
        print("Baseline models trained!")
        
    def train_collaborative_filtering(self):
        """Train collaborative filtering models"""
        print("\nTraining Collaborative Filtering Models...")
        print("-" * 40)
        
        cf = CollaborativeFiltering(self.train_data, self.test_data)
        cf.prepare_data()
        
        # Train SVD
        cf.fit_svd(n_factors=100, n_epochs=20)
        self.models['SVD'] = cf.models['SVD']
        
        # Train User KNN
        cf.fit_user_knn(k=50)
        self.models['User_KNN'] = cf.models['User_KNN']
        
        # Train Item KNN
        cf.fit_item_knn(k=50)
        self.models['Item_KNN'] = cf.models['Item_KNN']
        
        print("Collaborative filtering models trained!")
        
    def train_neural_models(self):
        """Train neural recommendation models"""
        print("\nTraining Neural Models...")
        print("-" * 25)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize neural recommender
        neural_rec = NeuralRecommender(
            self.train_data, self.test_data, 
            self.user_to_idx, self.item_to_idx, device
        )
        
        # Train Matrix Factorization
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        mf_model = MatrixFactorization(n_users, n_items, n_factors=64)
        neural_rec.train_model('Matrix_Factorization', mf_model, epochs=15)
        self.models['Neural_MF'] = neural_rec.models['Matrix_Factorization']
        
        # Store neural recommender for predictions
        self.neural_rec = neural_rec
        
        print("Neural models trained!")
        
    def get_movie_title(self, movie_id):
        """Get movie title by ID"""
        movie_info = self.movies[self.movies.item_id == movie_id]
        if len(movie_info) > 0:
            return movie_info.title.iloc[0]
        return f"Movie {movie_id}"
    
    def get_user_ratings(self, user_id):
        """Get user's movie ratings"""
        user_ratings = self.train_data[self.train_data.user_id == user_id]
        if len(user_ratings) == 0:
            return []
        
        ratings_with_titles = []
        for _, row in user_ratings.iterrows():
            title = self.get_movie_title(row.item_id)
            ratings_with_titles.append((title, row.rating))
        
        return sorted(ratings_with_titles, key=lambda x: x[1], reverse=True)
    
    def get_recommendations(self, user_id, model_name, n=10):
        """Get recommendations for a user from a specific model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return []
        
        model = self.models[model_name]
        
        try:
            if model_name == 'Neural_MF':
                recommendations = self.neural_rec.get_recommendations('Matrix_Factorization', user_id, n)
            elif hasattr(model, 'get_recommendations'):
                recommendations = model.get_recommendations(user_id, n)
            elif hasattr(model, 'recommend'):
                recommendations = model.recommend(user_id, n)
            elif model_name == 'Popular_Movies':
                recommendations = model.head(n).index.tolist()
            else:
                print(f"Model {model_name} doesn't support recommendations")
                return []
            
            # Convert to titles
            recommendations_with_titles = []
            for movie_id in recommendations:
                title = self.get_movie_title(movie_id)
                recommendations_with_titles.append((movie_id, title))
            
            return recommendations_with_titles
            
        except Exception as e:
            print(f"Error getting recommendations from {model_name}: {e}")
            return []
    
    def compare_recommendations(self, user_id, n=5):
        """Compare recommendations from different models"""
        print(f"\nComparing recommendations for User {user_id}")
        print("=" * 60)
        
        # Get user's ratings
        user_ratings = self.get_user_ratings(user_id)
        if user_ratings:
            print(f"User {user_id}'s top rated movies:")
            for i, (title, rating) in enumerate(user_ratings[:5], 1):
                print(f"{i}. {title} ({rating} stars)")
        else:
            print(f"User {user_id} has no ratings (cold start user)")
        
        print("\nRecommendations from different models:")
        print("-" * 40)
        
        for model_name in ['Popularity', 'Content_Based', 'SVD', 'Neural_MF']:
            if model_name in self.models:
                recommendations = self.get_recommendations(user_id, model_name, n)
                print(f"\n{model_name}:")
                for i, (movie_id, title) in enumerate(recommendations, 1):
                    print(f"  {i}. {title}")
    
    def interactive_demo(self):
        """Run interactive demo"""
        print("\n" + "=" * 60)
        print("MOVIE RECOMMENDATION SYSTEM - INTERACTIVE DEMO")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Get recommendations for a user")
            print("2. Compare recommendations from different models")
            print("3. Show user's rating history")
            print("4. Show movie information")
            print("5. Evaluate model performance")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                try:
                    user_id = int(input("Enter user ID: "))
                    if user_id not in self.user_to_idx:
                        print(f"User {user_id} not found in dataset")
                        continue
                    
                    print("\nAvailable models:")
                    for i, model_name in enumerate(self.models.keys(), 1):
                        print(f"{i}. {model_name}")
                    
                    model_choice = input("Select model (enter number): ").strip()
                    try:
                        model_idx = int(model_choice) - 1
                        model_names = list(self.models.keys())
                        if 0 <= model_idx < len(model_names):
                            model_name = model_names[model_idx]
                            n = int(input("Number of recommendations (default 10): ") or "10")
                            
                            recommendations = self.get_recommendations(user_id, model_name, n)
                            print(f"\nTop {n} recommendations from {model_name}:")
                            for i, (movie_id, title) in enumerate(recommendations, 1):
                                print(f"{i}. {title}")
                        else:
                            print("Invalid model selection")
                    except ValueError:
                        print("Invalid input")
                        
                except ValueError:
                    print("Invalid user ID")
                    
            elif choice == '2':
                try:
                    user_id = int(input("Enter user ID: "))
                    if user_id not in self.user_to_idx:
                        print(f"User {user_id} not found in dataset")
                        continue
                    self.compare_recommendations(user_id)
                except ValueError:
                    print("Invalid user ID")
                    
            elif choice == '3':
                try:
                    user_id = int(input("Enter user ID: "))
                    if user_id not in self.user_to_idx:
                        print(f"User {user_id} not found in dataset")
                        continue
                    
                    ratings = self.get_user_ratings(user_id)
                    if ratings:
                        print(f"\nUser {user_id}'s rating history:")
                        for i, (title, rating) in enumerate(ratings, 1):
                            print(f"{i}. {title} ({rating} stars)")
                    else:
                        print(f"User {user_id} has no ratings")
                except ValueError:
                    print("Invalid user ID")
                    
            elif choice == '4':
                try:
                    movie_id = int(input("Enter movie ID: "))
                    movie_info = self.movies[self.movies.item_id == movie_id]
                    if len(movie_info) > 0:
                        movie = movie_info.iloc[0]
                        print(f"\nMovie Information:")
                        print(f"Title: {movie.title}")
                        print(f"Release Date: {movie.release_date}")
                        print(f"IMDB URL: {movie.imdb_url}")
                        
                        # Show genres
                        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
                        genres = []
                        for col in genre_cols:
                            if movie[col] == 1:
                                genre_name = col.replace('genre_', '')
                                genres.append(genre_name)
                        print(f"Genres: {', '.join(genres)}")
                    else:
                        print(f"Movie {movie_id} not found")
                except ValueError:
                    print("Invalid movie ID")
                    
            elif choice == '5':
                print("\nEvaluating model performance...")
                evaluator = RecommendationEvaluator(self.test_data, self.movies)
                
                # Create models dict for evaluation
                eval_models = {}
                for name, model in self.models.items():
                    if name in ['SVD', 'User_KNN', 'Item_KNN', 'Neural_MF']:
                        eval_models[name] = model
                
                if eval_models:
                    comparison_df = evaluator.compare_models(eval_models, self.train_data, self.test_data)
                    print("\nModel Performance Summary:")
                    print(comparison_df.round(3))
                else:
                    print("No evaluable models found")
                    
            elif choice == '6':
                print("Thank you for using the Movie Recommendation System!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-6.")

def run_demo():
    """Run the complete recommendation system demo"""
    demo = MovieRecommendationDemo()
    
    # Setup system
    demo.setup_system()
    
    # Train models
    demo.train_baseline_models()
    demo.train_collaborative_filtering()
    demo.train_neural_models()
    
    # Run interactive demo
    demo.interactive_demo()

if __name__ == "__main__":
    run_demo() 