# 🎬 Movie Recommendation System

> An academic implementation of multiple recommendation algorithms on the MovieLens 100K dataset — built for IIT Ropar.

---

## Overview

This project explores and compares a range of recommendation techniques, from simple baselines to neural architectures. It predicts user ratings and generates personalized top-N movie recommendations, evaluated across multiple metrics.

- **Dataset**: MovieLens 100K (100K ratings · 1,000 users · 1,700 movies · 1–5 star scale)
- **Goal**: RMSE < 1.0 with at least 3 distinct algorithms
- **Evaluation**: RMSE, MAE, NDCG@10, Hit Rate@10, Precision@10, Recall@10, F1@10

---

## Algorithms Implemented

### Baseline (Non-Personalized)
- **Global Average** — Predicts all ratings as the global mean
- **Popular Movies** — Recommends highest-rated movies by average score

### Content-Based Filtering
- **TF-IDF on Genres** — Cosine similarity over TF-IDF vectors of movie genres and metadata

### Neural Models
- **Matrix Factorization** — User/item embeddings with dot-product prediction (MSE loss, Adam optimizer)
- **Two-Tower Model** — Separate user and item towers; better suited for large-scale retrieval
- **Neural MF with MLP** — Combines MF embeddings with non-linear MLP layers for richer interaction modelling

### Collaborative Filtering *(optional)*
- **SVD** and **KNN (user-based & item-based)** via `scikit-surprise` — requires Visual C++ Build Tools on Windows

---

## Performance Benchmarks

| Metric | Expected Range |
|---|---|
| RMSE | 0.85 – 1.20 |
| MAE | 0.65 – 0.95 |
| NDCG@10 | 0.30 – 0.60 |
| Hit Rate@10 | 0.40 – 0.70 |

**Typical model ranking:** Neural MF > Two-Tower > Matrix Factorization > Content-Based > Baselines

---

## Installation

### Prerequisites
- Python 3.8+
- 4 GB+ RAM
- GPU optional (CUDA supported for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd movie_recommendation_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   python data_downloader.py
   ```
   Or manually place MovieLens 100K files (`u.data`, `u.item`, `u.user`) into `movie/ml-100k/`.

### Optional: Enable Collaborative Filtering

Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Windows only, select the "C++ build tools" workload), then:
```bash
pip install scikit-surprise seaborn
```

---

## Usage

### Without `scikit-surprise`
```bash
python main_simple.py          # Full pipeline, no CF dependencies
python quick_demo.py           # Quick demonstration
python simple_test.py          # Component tests
```

### With All Algorithms
```bash
python main.py                 # Complete pipeline
python recommendation_demo.py  # Interactive demo
python test_system.py          # Full test suite
```

The interactive demo (`recommendation_demo.py`) lets you enter a user ID, compare recommendations across models, and view rating predictions in real time.

---

## Project Structure

```
movie_recommendation_system/
├── movie/ml-100k/             # MovieLens dataset files
├── data_downloader.py         # Dataset download script
├── data_loader.py             # Loading and preprocessing
├── baseline_models.py         # Global average, popularity
├── collaborative_filtering.py # SVD, KNN (requires scikit-surprise)
├── neural_models.py           # MF, Two-Tower, Neural MF
├── evaluation.py              # Metrics and model comparison
├── recommendation_demo.py     # Interactive demo
├── main.py                    # Full pipeline
├── main_simple.py             # Pipeline without CF dependencies
├── quick_demo.py              # Quick demo
├── requirements.txt
└── Generated Files/
    ├── model_results.csv
    ├── model_comparison.png
    ├── evaluation_report.txt
    └── final_project_report.txt
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Microsoft Visual C++ 14.0 or greater is required` | Install Visual C++ Build Tools with C++ workload |
| `ModuleNotFoundError: No module named 'surprise'` | Use `main_simple.py` or install optional dependencies |
| `FileNotFoundError` for dataset | Verify files are in `movie/ml-100k/` |
| `MemoryError` | Reduce batch size or embedding dimensions in `neural_models.py` |

For faster training, set `device = torch.device('cuda')` in the neural model config if a GPU is available.

---

## References

- Koren et al. (2009). *Matrix Factorization Techniques for Recommender Systems*
- He et al. (2017). *Neural Collaborative Filtering*
- Covington et al. (2016). *Deep Neural Networks for YouTube Recommendations*
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) · [PyTorch](https://pytorch.org/) · [Scikit-surprise](http://surpriselib.com/)

---

## License

Created for academic purposes at **IIT Ropar**. Provided as-is for educational use.
