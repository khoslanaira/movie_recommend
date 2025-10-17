import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class RecommendationEvaluator:
    """Comprehensive evaluation system for recommendation models"""

    def __init__(self, test_data, movies):
        self.test_data = test_data
        self.movies = movies

    def calculate_rmse(self, y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))

    def calculate_mae(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def calculate_ndcg_at_k(self, recommended, relevant, k=10):
        dcg = 0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1 / math.log2(i + 2)
        idcg = sum([1 / math.log2(i + 2) for i in range(min(len(relevant), k))])
        return dcg / idcg if idcg > 0 else 0

    def calculate_hit_rate_at_k(self, recommended, relevant, k=10):
        return int(len(set(recommended[:k]) & set(relevant)) > 0)

    def calculate_precision_at_k(self, recommended, relevant, k=10):
        if len(recommended[:k]) == 0:
            return 0
        return len(set(recommended[:k]) & set(relevant)) / len(recommended[:k])

    def calculate_recall_at_k(self, recommended, relevant, k=10):
        if len(relevant) == 0:
            return 0
        return len(set(recommended[:k]) & set(relevant)) / len(relevant)

    def calculate_f1_at_k(self, recommended, relevant, k=10):
        precision = self.calculate_precision_at_k(recommended, relevant, k)
        recall = self.calculate_recall_at_k(recommended, relevant, k)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_rating_prediction(self, model, model_name, test_data):
        """Evaluate rating prediction performance"""
        predictions = []
        actuals = []

        print(f"Evaluating {model_name} for rating prediction...")

        for _, row in test_data.iterrows():
            user_id = row.user_id
            item_id = row.item_id
            actual_rating = row.rating

            pred = None

            # 1) NeuralAdapter or any model with predict_rating
            if hasattr(model, "predict_rating"):
                pred = model.predict_rating(user_id, item_id)

            # 2) Surprise models
            elif hasattr(model, "predict"):
                pred = model.predict(user_id, item_id).est

            # 3) Raw PyTorch models
            elif hasattr(model, "forward"):
                user_idx = getattr(model, "user_to_idx", {}).get(user_id, None)
                item_idx = getattr(model, "item_to_idx", {}).get(item_id, None)
                if user_idx is not None and item_idx is not None:
                    model.eval()
                    with torch.no_grad():
                        u = torch.tensor([user_idx])
                        i = torch.tensor([item_idx])
                        out = model(u, i)
                        pred = float(out.cpu().numpy()[0]) if hasattr(out, "cpu") else float(out[0])

            if pred is not None:
                predictions.append(pred)
                actuals.append(actual_rating)

        rmse = self.calculate_rmse(actuals, predictions)
        mae = self.calculate_mae(actuals, predictions)

        results = {"RMSE": rmse, "MAE": mae}

        print(f"{model_name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
        return results

    def evaluate_recommendations(self, model, model_name, train_data, test_data, k=10):
        """Evaluate top-K recommendations"""
        print(f"Evaluating {model_name} for recommendations...")

        # Relevant items (rated >= 4 in test)
        relevant_items = {}
        for _, row in test_data.iterrows():
            if row.rating >= 4:
                relevant_items.setdefault(row.user_id, set()).add(row.item_id)

        ndcg_scores, hit_rates, precision_scores, recall_scores, f1_scores = [], [], [], [], []

        for user_id, relevant in relevant_items.items():
            recommended = []
            if hasattr(model, "get_recommendations"):
                recommended = model.get_recommendations(user_id, n=k)
            elif hasattr(model, "recommend"):
                recommended = model.recommend(user_id, n=k)

            if not recommended:
                continue

            ndcg_scores.append(self.calculate_ndcg_at_k(recommended, relevant, k))
            hit_rates.append(self.calculate_hit_rate_at_k(recommended, relevant, k))
            precision_scores.append(self.calculate_precision_at_k(recommended, relevant, k))
            recall_scores.append(self.calculate_recall_at_k(recommended, relevant, k))
            f1_scores.append(self.calculate_f1_at_k(recommended, relevant, k))

        results = {
            "NDCG@10": np.mean(ndcg_scores) if ndcg_scores else 0,
            "Hit_Rate@10": np.mean(hit_rates) if hit_rates else 0,
            "Precision@10": np.mean(precision_scores) if precision_scores else 0,
            "Recall@10": np.mean(recall_scores) if recall_scores else 0,
            "F1@10": np.mean(f1_scores) if f1_scores else 0
        }

        print(f"{model_name} - NDCG@10: {results['NDCG@10']:.3f}, Hit Rate@10: {results['Hit_Rate@10']:.3f}")
        return results

    def compare_models(self, models_dict, train_data, test_data):
        """Compare multiple models"""
        print("=== Model Comparison ===")
        comparison_results = {}

        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            rating_results = self.evaluate_rating_prediction(model, model_name, test_data)
            rec_results = self.evaluate_recommendations(model, model_name, train_data, test_data)
            comparison_results[model_name] = {**rating_results, **rec_results}

        comparison_df = pd.DataFrame(comparison_results).T
        print("\n=== Final Comparison ===")
        print(comparison_df.round(3))
        return comparison_df

    def plot_comparison(self, comparison_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        axes[0, 0].bar(comparison_df.index, comparison_df["RMSE"])
        axes[0, 0].set_title("RMSE (Lower Better)")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].tick_params(axis="x", rotation=45)

        axes[0, 1].bar(comparison_df.index, comparison_df["MAE"])
        axes[0, 1].set_title("MAE (Lower Better)")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].tick_params(axis="x", rotation=45)

        axes[1, 0].bar(comparison_df.index, comparison_df["NDCG@10"])
        axes[1, 0].set_title("NDCG@10 (Higher Better)")
        axes[1, 0].set_ylabel("NDCG@10")
        axes[1, 0].tick_params(axis="x", rotation=45)

        axes[1, 1].bar(comparison_df.index, comparison_df["Hit_Rate@10"])
        axes[1, 1].set_title("Hit Rate@10 (Higher Better)")
        axes[1, 1].set_ylabel("Hit Rate@10")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_report(self, comparison_df, save_path=None):
        report = []
        report.append("=" * 60)
        report.append("MOVIE RECOMMENDATION SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("OVERALL SUMMARY:")
        report.append("-" * 20)

        best_rmse_model = comparison_df["RMSE"].idxmin()
        best_ndcg_model = comparison_df["NDCG@10"].idxmax()
        best_hit_rate_model = comparison_df["Hit_Rate@10"].idxmax()

        report.append(f"Best RMSE: {best_rmse_model} ({comparison_df.loc[best_rmse_model, 'RMSE']:.3f})")
        report.append(f"Best NDCG@10: {best_ndcg_model} ({comparison_df.loc[best_ndcg_model, 'NDCG@10']:.3f})")
        report.append(f"Best Hit Rate@10: {best_hit_rate_model} ({comparison_df.loc[best_hit_rate_model, 'Hit_Rate@10']:.3f})")
        report.append("")

        report.append("DETAILED RESULTS:")
        report.append("-" * 20)
        report.append(comparison_df.round(3).to_string())
        report.append("")

        report.append("=" * 60)

        full_report = "\n".join(report)
        if save_path:
            with open(save_path, "w") as f:
                f.write(full_report)

        print(full_report)
        return full_report


def evaluate_all_models(train_data, test_data, movies, models_dict):
    """Evaluate all models and generate report"""
    print("Starting comprehensive model evaluation...")
    evaluator = RecommendationEvaluator(test_data, movies)
    comparison_df = evaluator.compare_models(models_dict, train_data, test_data)
    evaluator.plot_comparison(comparison_df, save_path="model_comparison.png")
    report = evaluator.generate_report(comparison_df, save_path="evaluation_report.txt")
    comparison_df.to_csv("model_results.csv")
    return comparison_df, report
