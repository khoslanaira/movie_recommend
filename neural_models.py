import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time

class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens ratings"""
    def __init__(self, ratings_df, user_to_idx, item_to_idx, mean_rating=None, std_rating=None):
        self.users = torch.tensor([user_to_idx[user_id] for user_id in ratings_df.user_id.values], dtype=torch.long)
        self.items = torch.tensor([item_to_idx[item_id] for item_id in ratings_df.item_id.values], dtype=torch.long)
        self.ratings = torch.tensor(ratings_df.rating.values, dtype=torch.float32)
        self.mean_rating = mean_rating
        self.std_rating = std_rating

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating = self.ratings[idx]
        if self.mean_rating is not None and self.std_rating is not None:
            rating = (rating - self.mean_rating) / self.std_rating
        return self.users[idx], self.items[idx], rating


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, dropout=0.1):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user, item):
        user_embed = self.dropout(self.user_factors(user))
        item_embed = self.dropout(self.item_factors(item))
        dot_product = (user_embed * item_embed).sum(dim=1)
        return dot_product + self.user_biases(user).squeeze() + self.item_biases(item).squeeze() + self.global_bias

    def predict_rating(self, user_id, item_id, user_to_idx=None, item_to_idx=None, device=None, mean=None, std=None):
        u_map = user_to_idx or getattr(self, "user_to_idx", None)
        i_map = item_to_idx or getattr(self, "item_to_idx", None)
        dev = device or getattr(self, "device", "cpu")
        if u_map is None or i_map is None:
            return 0.0
        if user_id not in u_map or item_id not in i_map:
            return 0.0

        user_idx = torch.tensor([u_map[user_id]], device=dev)
        item_idx = torch.tensor([i_map[item_id]], device=dev)
        self.eval()
        with torch.no_grad():
            out = self(user_idx, item_idx).cpu().item()
            return float(out * std + mean) if mean is not None and std is not None else float(out)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_fc = self._build_fc_layers(embedding_dim, hidden_dims, dropout)
        self.item_fc = self._build_fc_layers(embedding_dim, hidden_dims, dropout)
        self.output_layer = nn.Linear(hidden_dims[-1] * 2, 1)
        self._init_weights()

    def _build_fc_layers(self, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        for module in [self.user_fc, self.item_fc]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user, item):
        user_vec = self.user_fc(self.user_embedding(user))
        item_vec = self.item_fc(self.item_embedding(item))
        combined = torch.cat([user_vec, item_vec], dim=1)
        return self.output_layer(combined).squeeze()

    def predict_rating(self, user_id, item_id, user_to_idx=None, item_to_idx=None, device=None, mean=None, std=None):
        u_map = user_to_idx or getattr(self, "user_to_idx", None)
        i_map = item_to_idx or getattr(self, "item_to_idx", None)
        dev = device or getattr(self, "device", "cpu")
        if u_map is None or i_map is None:
            return 0.0
        if user_id not in u_map or item_id not in i_map:
            return 0.0

        user_idx = torch.tensor([u_map[user_id]], device=dev)
        item_idx = torch.tensor([i_map[item_id]], device=dev)
        self.eval()
        with torch.no_grad():
            out = self(user_idx, item_idx).cpu().item()
            return float(out * std + mean) if mean is not None and std is not None else float(out)


class NeuralMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, layers=[64, 32], dropout=0.2):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.mlp = self._build_mlp(n_factors * 2, layers, dropout)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _build_mlp(self, input_dim, layers, dropout):
        mlp_layers = []
        prev_dim = input_dim
        for layer_dim in layers:
            mlp_layers += [nn.Linear(prev_dim, layer_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = layer_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*mlp_layers)

    def _init_weights(self):
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user, item):
        combined = torch.cat([self.user_factors(user), self.item_factors(item)], dim=1)
        mlp_pred = self.mlp(combined).squeeze()
        return mlp_pred + self.user_biases(user).squeeze() + self.item_biases(item).squeeze() + self.global_bias

    def predict_rating(self, user_id, item_id, user_to_idx=None, item_to_idx=None, device=None, mean=None, std=None):
        u_map = user_to_idx or getattr(self, "user_to_idx", None)
        i_map = item_to_idx or getattr(self, "item_to_idx", None)
        dev = device or getattr(self, "device", "cpu")
        if u_map is None or i_map is None:
            return 0.0
        if user_id not in u_map or item_id not in i_map:
            return 0.0

        user_idx = torch.tensor([u_map[user_id]], device=dev)
        item_idx = torch.tensor([i_map[item_id]], device=dev)
        self.eval()
        with torch.no_grad():
            out = self(user_idx, item_idx).cpu().item()
            return float(out * std + mean) if mean is not None and std is not None else float(out)


class NeuralRecommender:
    def __init__(self, train_data, test_data, user_to_idx, item_to_idx, device='cpu'):
        # Don't normalize ratings - this often hurts performance
        self.mean_rating = train_data.rating.mean()
        self.std_rating = 1.0  # Don't normalize
        self.train_data = train_data
        self.test_data = test_data
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.device = device
        self.models = {}
        self.train_dataset = MovieLensDataset(train_data, user_to_idx, item_to_idx, self.mean_rating, self.std_rating)
        self.test_dataset = MovieLensDataset(test_data, user_to_idx, item_to_idx, self.mean_rating, self.std_rating)
        print(f"Neural recommender initialized with {len(train_data)} train and {len(test_data)} test samples")

    def train_model(self, model_name, model, epochs=100, batch_size=256, lr=0.001, weight_decay=1e-4, patience=10):
        print(f"Training {model_name}...")
        model.user_to_idx = self.user_to_idx
        model.item_to_idx = self.item_to_idx
        model.device = self.device
        model = model.to(self.device)

        # Validation Split
        val_size = int(0.1 * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size
        train_subset, val_subset = random_split(self.train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # Use MSE loss instead of SmoothL1Loss for better convergence
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items, ratings in val_loader:
                    users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                    preds = model(users, items)
                    val_loss += criterion(preds, ratings).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if epoch % 5 == 0 or epoch < 10:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Store the best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        # Load the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store the trained model
        self.models[model_name] = model
        print(f"✅ {model_name} training completed. Best Val Loss: {best_val_loss:.4f}")
        return model

    def predict_rating(self, model_name, user_id, item_id):
        if model_name not in self.models:
            print(f"[WARN] Model {model_name} not found!")
            return 0.0
        model = self.models[model_name]
        try:
            pred = model.predict_rating(user_id, item_id,
                                        user_to_idx=self.user_to_idx,
                                        item_to_idx=self.item_to_idx,
                                        device=self.device,
                                        mean=self.mean_rating,
                                        std=self.std_rating)
            return float(pred) if pred is not None else 0.0
        except Exception as e:
            print(f"[ERROR] Prediction failed for {model_name}: {e}")
            return 0.0