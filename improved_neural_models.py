import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time

class ImprovedMovieLensDataset(Dataset):
    """Enhanced PyTorch Dataset with better preprocessing"""
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
        # Better normalization
        if self.mean_rating is not None and self.std_rating is not None:
            rating = (rating - self.mean_rating) / self.std_rating
        return self.users[idx], self.items[idx], rating


class ImprovedMatrixFactorization(nn.Module):
    """Enhanced Matrix Factorization with better architecture"""
    def __init__(self, n_users, n_items, n_factors=128, dropout=0.3):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        # Add MLP layers for better feature interaction
        self.mlp = nn.Sequential(
            nn.Linear(n_factors * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

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
        user_embed = self.dropout(self.user_factors(user))
        item_embed = self.dropout(self.item_factors(item))
        
        # Element-wise product
        element_wise = user_embed * item_embed
        
        # Concatenation for MLP
        concat_features = torch.cat([user_embed, item_embed], dim=1)
        mlp_output = self.mlp(concat_features).squeeze()
        
        # Combine different approaches
        dot_product = element_wise.sum(dim=1)
        bias_terms = self.user_biases(user).squeeze() + self.item_biases(item).squeeze() + self.global_bias
        
        return dot_product + mlp_output + bias_terms

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


class DeepTwoTowerModel(nn.Module):
    """Enhanced Two-Tower with deeper architecture"""
    def __init__(self, n_users, n_items, embedding_dim=128, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Deeper towers
        self.user_fc = self._build_deep_tower(embedding_dim, hidden_dims, dropout)
        self.item_fc = self._build_deep_tower(embedding_dim, hidden_dims, dropout)
        
        # Enhanced output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _build_deep_tower(self, input_dim, hidden_dims, dropout):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        for module in [self.user_fc, self.item_fc, self.output_layer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

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


class AdvancedNeuralMF(nn.Module):
    """Advanced Neural Matrix Factorization with attention"""
    def __init__(self, n_users, n_items, n_factors=128, layers=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Enhanced MLP with residual connections
        self.mlp = self._build_advanced_mlp(n_factors * 2, layers, dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(n_factors, num_heads=4, dropout=dropout)
        
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _build_advanced_mlp(self, input_dim, layers, dropout):
        mlp_layers = []
        prev_dim = input_dim
        for i, layer_dim in enumerate(layers):
            mlp_layers += [
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
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
        user_embed = self.user_factors(user)
        item_embed = self.item_factors(item)
        
        # Apply attention
        user_attended, _ = self.attention(user_embed.unsqueeze(1), user_embed.unsqueeze(1), user_embed.unsqueeze(1))
        item_attended, _ = self.attention(item_embed.unsqueeze(1), item_embed.unsqueeze(1), item_embed.unsqueeze(1))
        
        user_embed = user_attended.squeeze(1)
        item_embed = item_attended.squeeze(1)
        
        combined = torch.cat([user_embed, item_embed], dim=1)
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


class ImprovedNeuralRecommender:
    """Enhanced neural recommender with better training"""
    def __init__(self, train_data, test_data, user_to_idx, item_to_idx, device='cpu'):
        # Better normalization strategy
        self.mean_rating = train_data.rating.mean()
        self.std_rating = train_data.rating.std()
        self.train_data = train_data
        self.test_data = test_data
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.device = device
        self.models = {}
        
        # Use improved dataset
        self.train_dataset = ImprovedMovieLensDataset(train_data, user_to_idx, item_to_idx, self.mean_rating, self.std_rating)
        self.test_dataset = ImprovedMovieLensDataset(test_data, user_to_idx, item_to_idx, self.mean_rating, self.std_rating)
        print(f"Improved neural recommender initialized with {len(train_data)} train and {len(test_data)} test samples")

    def train_model(self, model_name, model, epochs=150, batch_size=128, lr=0.0005, weight_decay=1e-3, patience=15):
        print(f"Training {model_name} with improved settings...")
        model.user_to_idx = self.user_to_idx
        model.item_to_idx = self.item_to_idx
        model.device = self.device
        model = model.to(self.device)

        # Better validation split
        val_size = int(0.15 * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size
        train_subset, val_subset = random_split(self.train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=0)

        # Better loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Advanced learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                loss.backward()
                
                # Gradient clipping
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
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.models[model_name] = model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

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
