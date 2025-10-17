import pandas as pd
import numpy as np
try:
    from surprise import Dataset, Reader, SVD, KNNBasic, KNNWithMeans, KNNWithZScore
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
if SURPRISE_AVAILABLE:
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
import time

class CollaborativeFiltering:
    def __init__(self, train_data, test_data):
        if not SURPRISE_AVAILABLE:
            raise ImportError("scikit-surprise is required for CollaborativeFiltering")
        self.train_data = train_data
        self.test_data = test_data
        self.models = {}
        self.trainset = None
        self.testset = None
        self.reader = Reader(rating_scale=(1, 5))
        
    def prepare_data(self):
        """Prepare data for Surprise library"""
        # Create Surprise dataset
        data = Dataset.load_from_df(
            self.train_data[['user_id', 'item_id', 'rating']], 
            self.reader
        )
        
        # Build trainset
        self.trainset = data.build_full_trainset()
        
        # Create testset
        test_data_surprise = self.test_data[['user_id', 'item_id', 'rating']].values.tolist()
        self.testset = test_data_surprise
        
        print(f"Prepared data: {self.trainset.n_users} users, {self.trainset.n_items} items")
        print(f"Train ratings: {self.trainset.n_ratings}, Test ratings: {len(self.testset)}")
        
    def fit_svd(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """Fit SVD (Singular Value Decomposition) model"""
        print("Training SVD model...")
        start_time = time.time()
        
        svd = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        svd.fit(self.trainset)
        
        training_time = time.time() - start_time
        print(f"SVD training completed in {training_time:.2f} seconds")
        
        self.models['SVD'] = svd
        return svd
    
    def fit_user_knn(self, k=50, sim_options=None):
        """Fit user-based collaborative filtering"""
        if sim_options is None:
            sim_options = {'name': 'cosine', 'user_based': True}
        
        print("Training User-based KNN model...")
        start_time = time.time()
        
        user_knn = KNNBasic(k=k, sim_options=sim_options)
        user_knn.fit(self.trainset)
        
        training_time = time.time() - start_time
        print(f"User KNN training completed in {training_time:.2f} seconds")
        
        self.models['User_KNN'] = user_knn
        return user_knn
    
    def fit_item_knn(self, k=50, sim_options=None):
        """Fit item-based collaborative filtering"""
        if sim_options is None:
            sim_options = {'name': 'cosine', 'user_based': False}
        
        print("Training Item-based KNN model...")
        start_time = time.time()
        
        item_knn = KNNBasic(k=k, sim_options=sim_options)
        item_knn.fit(self.trainset)
        
        training_time = time.time() - start_time
        print(f"Item KNN training completed in {training_time:.2f} seconds")
        
        self.models['Item_KNN'] = item_knn
        return item_knn
    
    def fit_knn_with_means(self, k=50, sim_options=None):
        """Fit KNN with means model"""
        if sim_options is None:
            sim_options = {'name': 'cosine', 'user_based': True}
        
        print("Training KNN with means model...")
        start_time = time.time()
        
        knn_means = KNNWithMeans(k=k, sim_options=sim_options)
        knn_means.fit(self.trainset)
        
        training_time = time.time() - start_time
        print(f"KNN with means training completed in {training_time:.2f} seconds")
        
        self.models['KNN_Means'] = knn_means
        return knn_means
    
    def evaluate_model(self, model_name):
        """Evaluate a specific model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        # Make predictions on test set
        predictions = model.test(self.testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        results = {
            'RMSE': rmse,
            'MAE': mae
        }
        
        print(f"{model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nEvaluating {model_name}...")
            results[model_name] = self.evaluate_model(model_name)
        
        return results
    
    def cross_validate_model(self, model, cv=5):
        """Perform cross-validation on a model"""
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_results = cross_validate(model, self.trainset, measures=['RMSE', 'MAE'], cv=cv, verbose=True)
        
        print(f"Cross-validation results:")
        print(f"  RMSE: {cv_results['test_rmse'].mean():.3f} (+/- {cv_results['test_rmse'].std() * 2:.3f})")
        print(f"  MAE: {cv_results['test_mae'].mean():.3f} (+/- {cv_results['test_mae'].std() * 2:.3f})")
        
        return cv_results
    
    def get_recommendations(self, model_name, user_id, n=10):
        """Get top-N recommendations for a user"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return []
        
        model = self.models[model_name]
        
        # Get all items
        all_items = self.trainset.all_items()
        
        # Get user's rated items
        user_items = set()
        for item_id in self.trainset.ur[self.trainset.to_inner_uid(user_id)]:
            user_items.add(self.trainset.to_raw_iid(item_id))
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in all_items:
            raw_item_id = self.trainset.to_raw_iid(item_id)
            if raw_item_id not in user_items:
                pred = model.predict(user_id, raw_item_id)
                predictions.append((raw_item_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in predictions[:n]]
    
    def predict_rating(self, model_name, user_id, item_id):
        """Predict rating for a specific user-item pair"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        prediction = model.predict(user_id, item_id)
        return prediction.est

class AdvancedCollaborativeFiltering:
    """Advanced collaborative filtering with multiple algorithms"""
    
    def __init__(self, train_data, test_data):
        if not SURPRISE_AVAILABLE:
            raise ImportError("scikit-surprise is required for AdvancedCollaborativeFiltering")
        self.train_data = train_data
        self.test_data = test_data
        self.cf = CollaborativeFiltering(train_data, test_data)
        
    def train_all_models(self):
        """Train all collaborative filtering models"""
        print("=== Training Collaborative Filtering Models ===")
        
        # Prepare data
        self.cf.prepare_data()
        
        # Train models
        self.cf.fit_svd(n_factors=100, n_epochs=20)
        self.cf.fit_user_knn(k=50)
        self.cf.fit_item_knn(k=50)
        self.cf.fit_knn_with_means(k=50)
        
        print("All models trained successfully!")
        
    def compare_models(self):
        """Compare all trained models"""
        print("\n=== Model Comparison ===")
        results = self.cf.evaluate_all_models()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        print("\nModel Performance Comparison:")
        print(comparison_df.round(3))
        
        return comparison_df
    
    def get_best_model(self):
        """Get the best performing model based on RMSE"""
        results = self.cf.evaluate_all_models()
        
        if not results:
            return None
        
        # Find model with lowest RMSE
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
        print(f"\nBest model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.3f})")
        
        return best_model[0]

if __name__ == "__main__":
    # Test collaborative filtering
    from data_loader import MovieLensDataLoader
    
    loader = MovieLensDataLoader()
    loader.load_data()
    loader.clean_data(min_ratings=5)
    train_data, test_data = loader.split_data()
    
    # Test basic collaborative filtering
    cf = CollaborativeFiltering(train_data, test_data)
    cf.prepare_data()
    cf.fit_svd()
    cf.fit_user_knn()
    cf.fit_item_knn()
    
    results = cf.evaluate_all_models()
    
    # Test advanced collaborative filtering
    advanced_cf = AdvancedCollaborativeFiltering(train_data, test_data)
    advanced_cf.train_all_models()
    comparison = advanced_cf.compare_models()
    best_model = advanced_cf.get_best_model() 