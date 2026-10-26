# 🎓 E-Learning Recommendation System

> A hybrid recommender system for e-learning content, combining content-based filtering, collaborative filtering, session-based patterns, and diversity-aware re-ranking into a single ensemble.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📌 Overview

This project builds a recommendation engine for an e-learning platform, suggesting tutorials and courses tailored to each user's job role, learning history, and browsing sessions. It addresses the real-world challenges of sparse rating matrices, cold-start users, and filter-bubble effects through a multi-signal hybrid architecture.

The system was developed iteratively — a baseline hybrid model was analyzed, its bugs and limitations were documented, and an improved version was built to address each one.

---

## 🧩 Architecture

The system blends four recommendation signals into a single score:

```
User + Item
    │
    ├── Content-Based (30%)      TF-IDF on item descriptions × cosine similarity
    ├── Collaborative (50%)      SVD matrix factorization on explicit/implicit ratings
    ├── Session-Based (10%)      Item transition patterns from browsing sessions
    └── Popularity (10%)         Global item popularity for cold-start fallback
            │
            ▼
    Hybrid Ensemble Score
            │
    Diversity Re-Ranking (MMR)   Maximal Marginal Relevance, α=0.7
            │
            ▼
    Top-N Recommendations
```

**Adaptive weights** — the CB/CF balance shifts automatically based on how much interaction history a user has, so new users lean toward content-based and popular items rather than noisy CF predictions.

---

## ✨ Key Improvements Over Baseline

| Issue | Baseline | Improved |
|---|---|---|
| Temporal decay | Hardcoded 30-day window | Exponential decay (`e^{-0.01 × days}`) on actual item dates |
| Cold start | No fallback | Popularity-weighted fallback |
| User context | Ignored | Job role matched to item target audience |
| Session patterns | Not used | Markov-chain transition model from 60-min sessions |
| Diversity | No control | MMR re-ranking prevents filter bubbles |
| CF model | 100 factors, 20 epochs | 150 factors, 50 epochs (scales with 0.22% density matrix) |

---

## 📊 Results

Evaluated on a 70 / 10 / 20 train-validation-test split.

### Ranking Quality (Test Set)

| Metric | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| Precision | 0.697 | 0.505 | 0.388 | 0.242 |
| Recall | 0.457 | 0.730 | 0.856 | 0.937 |
| NDCG | 0.894 | 0.927 | **0.948** | **0.954** |
| F1 | 0.552 | 0.597 | 0.534 | 0.385 |

### Rating Accuracy

| RMSE | MAE | ROC AUC |
|---|---|---|
| 0.438 | 0.286 | **0.963** |

### Diversity & Coverage

| Intra-List Diversity | Catalog Coverage | Novelty |
|---|---|---|
| 0.930 | 2.83% | 0.730 |

> High intra-list diversity (0.93) confirms that MMR re-ranking successfully prevents the model from recommending near-identical items in the same top-N list.

---

## 🗂️ Project Structure

```
project/
├── data/
│   ├── users.csv                   # User profiles with job roles
│   ├── items.csv                   # Course catalog (title, description, theme, difficulty)
│   ├── explicit_ratings.csv        # Star ratings with watch percentage
│   └── implicit_ratings.csv        # View/interaction events
│
├── src/
│   ├── data_loader.py              # Loads and joins the four data files
│   ├── preprocessor_improved.py    # Recency weights, TF-IDF, session extraction
│   ├── models_improved.py          # CB, CF (SVD), Session, Hybrid, MMR re-ranker
│   ├── engine_improved.py          # Recommendation engine + explanation generator
│   ├── evaluator.py                # Full evaluation suite + plot generation
│   ├── main_improved.py            # Training & evaluation pipeline (CLI)
│   ├── app_improved.py             # Interactive Streamlit app
│   └── debug_scores.py             # Score debugging utilities
│
├── evaluation_results/
│   ├── evaluation_metrics.json     # All metrics (test, validation, diversity, ROC)
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── test_vs_validation.png
│   └── user_interaction_analysis.png
│
├── analyze_users.py                # Standalone user behavior analysis script
└── requirements.txt
```

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:
```
pandas
numpy
scikit-learn
scikit-surprise
```

The Streamlit app also needs:
```bash
pip install streamlit plotly
```

### Run the training & evaluation pipeline

```bash
cd project
python src/main_improved.py
```

This will:
1. Load all four data files from `data/`
2. Preprocess and extract sessions
3. Train CB, CF, session, and hybrid models
4. Run the full evaluation suite
5. Save metrics and plots to `evaluation_results/`
6. Print top-5 sample recommendations for a sample user

### Launch the interactive Streamlit app

```bash
cd project
streamlit run src/app_improved.py
```

---

## 📐 Evaluation Metrics

The evaluator computes a comprehensive set of metrics across multiple cut-offs:

- **Accuracy**: RMSE, MAE
- **Ranking**: Precision@K, Recall@K, NDCG@K, F1@K, HitRate@K — for K ∈ {1, 3, 5, 10, 20}
- **Binary classification**: ROC AUC, Average Precision (ratings ≥ 4 treated as positive)
- **Diversity**: Intra-List Diversity, Catalog Coverage (%), Novelty score
- **Simulated online**: Expected Completion Rate, CTR, Long-Tail Coverage

---

## 📦 Dataset

The dataset represents a real e-learning platform with tutorials in English covering software tools (OneDrive, Office 365, etc.). It is highly sparse — explicit ratings cover ~0.22% of all possible user-item pairs — making cold-start and sparsity handling central challenges.

| File | Records |
|---|---|
| users.csv | ~2,000 users |
| items.csv | ~1,300 items |
| explicit_ratings.csv | ~5,000 ratings |
| implicit_ratings.csv | ~25,000 interactions |

---

## 🔧 Configuration

Key hyperparameters are set in `main_improved.py` and `app_improved.py`:

```python
# Hybrid ensemble weights
w_cb = 0.3          # Content-based
w_cf = 0.5          # Collaborative filtering
w_session = 0.1     # Session-based
w_popularity = 0.1  # Popularity fallback

# SVD parameters
n_factors = 100
n_epochs = 20

# Diversity re-ranking
alpha = 0.7         # 1.0 = pure relevance, 0.0 = pure diversity

# Session extraction
session_gap_minutes = 60
```

---

## 📄 License

MIT
