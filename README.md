# Lazy Librarian - Book Recommendation System

A hybrid book recommendation system combining Collective Matrix Factorization (CMF) with content-based signals for a Swiss library dataset.

**Team:** Swatch
**Kaggle Leaderboard Score:** 15.389 MAP@10

---

## 🚀 Quick Start

Get the Streamlit UI running in 3 steps:

```bash
# 1. Clone the repository
git clone <repository-url>
cd lazy_librarian_kaggle

# 2. Install dependencies using uv (recommended)
uv sync

# 3. Run the Streamlit UI
uv run streamlit run front/main.py
```

The UI will open at **http://localhost:8501** 🎉

### Alternative: Using pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the UI
streamlit run front/main.py
```

---

## Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Running the Project](#-running-the-project)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Baseline Performance](#baseline-performance)
6. [Modern Model](#modern-model)
7. [Performance Gain](#performance-gain)
8. [Examples](#examples)
9. [Tooling & Cost Reflection](#tooling--cost-reflection)
10. [AI-Assisted Coding](#ai-assisted-coding)
11. [Collaboration & Sources](#collaboration--sources)
12. [Leaderboard Score](#leaderboard-score)
13. [Project Structure](#-project-structure)

---

## Exploratory Data Analysis

### Dataset Overview

The data comes from a **Swiss library system** with French and German language materials.

| Dataset | Records | Description |
|---------|---------|-------------|
| Interactions | 87,047 | User-item borrowing events |
| Items | 15,291 | Book metadata |

### Interactions Analysis

| Metric | Value |
|--------|-------|
| Total interactions | 87,047 |
| Unique (user, item) pairs | 64,003 |
| Unique users | 7,838 |
| Unique items (interacted) | 15,109 |
| Date range | Jan 2, 2023 → Oct 14, 2024 |
| **Matrix sparsity** | **99.93%** |

#### User Behavior Distribution

| Interactions per User | Count | Percentage |
|-----------------------|-------|------------|
| 3-5 (light users) | 3,737 | 47.7% |
| 6-10 | 1,884 | 24.0% |
| 11-20 | 1,224 | 15.6% |
| 21-50 | 744 | 9.5% |
| 50+ (power users) | 249 | 3.2% |

**Insight:** Nearly half the users have minimal history (3-5 interactions), making cold-start handling important.

#### Item Popularity Distribution

| Interactions per Item | Count | Percentage |
|-----------------------|-------|------------|
| 1 (long tail) | 692 | 4.6% |
| 2-5 | 9,754 | 64.6% |
| 6-10 | 3,579 | 23.7% |
| 11-20 | 789 | 5.2% |
| 20+ (popular) | 295 | 2.0% |

**Top 5 Most Popular Items:**
1. Le Petit Robert (dictionary) - 380 interactions
2. Demon Slayer (manga) - 357 interactions
3. Vagabond (manga) - 305 interactions
4. Spy x Family (manga) - 257 interactions
5. L'Arabe du futur (graphic novel) - 217 interactions

#### Repeat Interactions

| Metric | Value |
|--------|-------|
| Repeat (u,i) pairs | 10,376 pairs |
| Total repeat interactions | 23,044 (26.5% of all) |

Repeat interactions serve as a strong implicit signal for user preference strength.

### Items Metadata Analysis

| Feature | Coverage | Notes |
|---------|----------|-------|
| Title | 100% | 14,576 unique titles |
| Author | 82.6% | 9,357 unique authors |
| Publisher | 99.8% | 4,337 unique publishers |
| Subjects | 85.5% | 23,305 unique subject tags |

#### Subject Distribution

| Subjects per Item | Count | Percentage |
|-------------------|-------|------------|
| 1 subject | 3,160 | 24% |
| 2-3 subjects | 5,303 | 41% |
| 4-6 subjects | 2,905 | 22% |
| 7+ subjects | 1,700 | 13% |

**Top Subjects:**
1. Bandes dessinées (comics) - 993
2. Schweiz (Switzerland) - 918
3. Suisse (Switzerland FR) - 446
4. Guides pratiques - 351
5. Mangas - 246

### Key Findings

- ✅ Data is clean with no orphan interactions
- ✅ Good subject coverage (85.5% of interacted items have subjects)
- ⚠️ High sparsity (99.93%) - typical for recommender systems
- ⚠️ 26.5% of interactions are repeat borrows (exploitable as implicit signal)
- ⚠️ 14.5% of items missing subjects (cold-start challenge for content-based)
- 🌍 Bilingual collection (French/German)

---

## Baseline Performance

| Model | MAP@10 |
|-------|--------|
| Provided Baseline | **15.283** |

The baseline uses standard ALS (Alternating Least Squares) collaborative filtering on the user-item interaction matrix.

---

## Modern Model

### Architecture: Collective Matrix Factorization (CMF)

We implemented a **Collective Matrix Factorization** approach that jointly factorizes multiple matrices sharing user factors:
```
User-Item:      R ≈ U × V^T        (primary signal)
User-Author:    A ≈ U × Author^T   (side information)
User-Publisher: P ≈ U × Pub^T      (side information)
User-Subject:   S ≈ U × Subject^T  (side information)
```

The key insight is that **user factors U are shared** across all factorizations, enabling knowledge transfer from metadata.

### Design Choices

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Base Algorithm** | ALS (Alternating Least Squares) | Efficient for implicit feedback, handles sparsity well |
| **Side Information** | Author, Publisher, Subject | Available metadata with good coverage (82-99%) |
| **Subject Encoding** | TF-IDF (1,435 features) | Captures subject importance, better than binary multi-hot |
| **Title Embeddings** | OpenAI text-embedding-3-small (512 dims) | Semantic similarity for content-based signal |
| **Score Combination** | Normalized weighted sum | Balances collaborative and content-based signals |

### Model Components

#### 1. Collective Matrix Factorization
- Learns shared user factors across user-item, user-author, user-publisher, and user-subject matrices
- Each auxiliary matrix has its own weight (λ_author, λ_publisher, λ_subject)
- Uses confidence weighting for implicit feedback: `c = 1 + α × interaction_count`

#### 2. TF-IDF Subject Features
- 1,435 subject features extracted from 23,305 raw subject tags
- Filtered to subjects appearing ≥6 times
- TF-IDF weighting captures subject importance per item

#### 3. Title Embeddings
- 512-dimensional embeddings from OpenAI text-embedding-3-small
- User embeddings computed as mean of interacted item embeddings
- Cosine similarity for content-based scoring

#### 4. Hybrid Score Combination
```python
final_score = normalize(CMF_score) + w_embedding × normalize(embedding_score)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| factors | 64-128 | Latent dimension size |
| iterations | 10-20 | ALS iterations |
| alpha | 200-450 | Confidence scaling |
| regularization | 0.05 | L2 regularization |
| λ_author | 0.3 | Author matrix weight |
| λ_publisher | 0.3 | Publisher matrix weight |
| λ_subject | 0.6 | Subject matrix weight |
| w_embedding | 0.15 | Title embedding weight |

---

## Performance Gain

| Model | MAP@10 | Improvement |
|-------|--------|-------------|
| Baseline (ALS) | 15.283 | - |
| **CMF + TF-IDF + Embeddings** | **15.389** | **+0.106 (+0.7%)** |

### Analysis

The improvement is modest but consistent. Contributing factors:

**What Helped:**
- Subject TF-IDF features (λ_subject=0.6 was optimal) - subjects are strong indicators of user preference
- Higher alpha values (200-450) - stronger confidence weighting for implicit feedback
- Publisher information - good coverage (99.8%) and predictive of genre preferences

**What Had Limited Impact:**
- Title embeddings (w_embedding=0.15) - semantic similarity added marginal value on top of subject features
- Author features - many missing values (17.4%) and high cardinality limited effectiveness

**Challenges:**
- High sparsity (99.93%) limits collaborative filtering effectiveness
- Cold-start users (47.7% with ≤5 interactions) difficult to model
- Subject coverage gaps (14.5% missing) affect content-based fallback

---

## Examples

*Section to be completed with specific recommendation examples.*

---

## Tooling & Cost Reflection

### Tools Used

| Tool | Purpose | Cost | Necessity |
|------|---------|------|-----------|
| **OpenAI API** | Title embeddings (text-embedding-3-small) | ~$0.50 | Optional - marginal improvement |
| **implicit** | ALS baseline implementation reference | Free | Helpful for validation |
| **scipy** | Sparse matrix operations | Free | Essential |
| **pandas/numpy** | Data processing | Free | Essential |
| **scikit-learn** | TF-IDF vectorization, preprocessing | Free | Essential |
| **pyarrow** | Parquet file handling | Free | Convenience |

### Cost Summary

| Resource | Cost |
|----------|------|
| OpenAI Embeddings | < $0.50 |
| Compute | Local machine |
| **Total** | **< $1.00** |

The OpenAI embeddings were an experiment to test semantic similarity. Given the marginal improvement (+0.7%), they are **not strictly necessary** - the TF-IDF subject features provide most of the content-based signal.

---

## AI-Assisted Coding

### Experience with Claude

**Initial Phase:** Claude was counterproductive. Following AI suggestions without a clear framework led to confusion and wasted time.

**What Didn't Work:**
- Trying to build the entire system with AI guidance from scratch
- Accepting complex solutions without understanding the fundamentals

**What Worked:**
- Building a working baseline independently first
- Using Claude for **incremental improvements** once the framework existed:
  - Adding new parameters to existing code
  - Debugging index alignment issues
  - Generating boilerplate code for new features
  - Code review and refactoring

**Key Lesson:** AI assistants are most effective as **augmentation tools** when you already understand the problem and have a working solution. They struggle as primary architects for complex systems.

---

## Collaboration & Sources

### References

| Resource | Usage |
|----------|-------|
| [Implicit Feedback ALS (YouTube)](https://www.youtube.com/watch?v=_hf_y-_sj5Y) | Understanding ALS for implicit feedback |

### Collaboration

This project was completed **independently** without collaboration with other teams.

---

## Leaderboard Score

| Metric | Value |
|--------|-------|
| **Team Name** | Swatch |
| **Final MAP@10** | **15.389** |
| **Baseline MAP@10** | 15.283 |
| **Improvement** | +0.7% |

---

## 📦 Installation

### Prerequisites

- **Python** ≥ 3.13
- **RAM**: ~4GB for model training
- **Disk Space**: ~500MB for data and models

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd lazy_librarian_kaggle

# Sync dependencies (creates .venv and installs packages)
uv sync

# Verify installation
uv run python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed!')"
```

### Method 2: Using pip

```bash
# Clone repository
git clone <repository-url>
cd lazy_librarian_kaggle

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -e .

# Verify installation
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed!')"
```

### Verify Installation

After installation, verify all key packages:

```bash
# Using uv
uv run python -c "import streamlit, pandas, numpy, implicit; print('✓ All packages installed')"

# Using activated venv
python -c "import streamlit, pandas, numpy, implicit; print('✓ All packages installed')"
```

---

## 🎮 Running the Project

### 1. Streamlit UI (Interactive Book Recommendations)

The Streamlit UI provides an interactive interface to explore the book recommendation system.

**Using uv (no activation needed):**
```bash
uv run streamlit run front/main.py
```

**Using activated virtual environment:**
```bash
source .venv/bin/activate  # Activate first
streamlit run front/main.py
```

**Custom port:**
```bash
uv run streamlit run front/main.py --server.port 8502
```

**Features:**
- 📊 Browse 15,000+ books with cover images
- 👥 Explore 7,800+ user profiles
- ✨ Get personalized recommendations using CMF model
- 📚 View borrowing history and statistics

**Access:** Open http://localhost:8501 in your browser

### 2. Model Training Pipeline

Train the Collective Matrix Factorization model with hyperparameter tuning.

**Grid Search (Hyperparameter Tuning):**
```bash
# Using uv
uv run python pipe/collective_matrix_factorisation.py

# Using activated venv
python pipe/collective_matrix_factorisation.py
```

**Expected output:**
- Validation MAP@10 scores for different hyperparameter combinations
- Best hyperparameters saved to console
- Training time: ~10-30 minutes depending on grid size

**Full Training (All Data):**
```bash
# Using uv
uv run python pipe/train.py

# Using activated venv
python pipe/train.py
```

**Expected output:**
- Trained model saved to `models/cmf_full_model_enriched.pkl`
- Training time: ~5-15 minutes
- Console output shows iteration progress and loss

### 3. Generate Kaggle Submission

Create submission file for Kaggle competition.

```bash
# Using uv
uv run python pipe/submit.py

# Using activated venv
python pipe/submit.py
```

**Output:**
- `submission_cmf_enriched.csv` in project root
- Format: user_id, recommended item IDs (space-separated)
- Ready for Kaggle submission

---

## 📁 Project Structure

```
lazy_librarian_kaggle/
├── data/
│   ├── interactions.csv             # User-item borrowing events (87K records)
│   ├── items.csv                    # Book metadata (15K books)
│   ├── items_enriched.csv           # Items with additional features
│   └── _/                           # Preprocessed features
│       ├── subject_tfidf.npz        # TF-IDF subject matrix (1,435 features)
│       ├── title_embeddings.npy     # OpenAI embeddings (512 dims)
│       └── items_with_topics.parquet
│
├── front/
│   ├── main.py                      # 🎨 Streamlit UI application (1,330 lines)
│   └── cover_cache.csv              # Cached book cover URLs
│
├── models/
│   ├── cmf_recommender.py           # CMF model wrapper class
│   ├── cmf_full_model_enriched.pkl  # Trained model (generated)
│   └── cmf_full_model_tfidf.pkl     # Alternative model variant
│
├── pipe/
│   ├── collective_matrix_factorisation.py  # 🤖 CMF training with grid search
│   ├── train.py                     # Full training pipeline
│   ├── eval.py                      # MAP@10 evaluation
│   ├── split.py                     # Train/val/test splitting
│   ├── submit.py                    # 📊 Submission file generation
│   ├── hybrid_ranking.py            # Score combination logic
│   └── stats.py                     # Dataset statistics
│
├── data_preprocessing/
│   ├── clean_dataset.py             # Main data cleaning pipeline
│   ├── clean_items.py               # Item metadata cleaning
│   ├── clean_interactions.py        # Interaction data cleaning
│   ├── encode_categorical.py        # Author/Publisher encoding
│   ├── encode_subjects.py           # Subject TF-IDF encoding
│   └── load_data.py                 # Data loading utilities
│
├── pyproject.toml                   # 📦 Project dependencies
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Python 3.13
├── README.md                        # This file
└── submission_cmf_enriched.csv      # Generated submission file
```

### Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `front/main.py` | Streamlit UI | Explore data interactively |
| `pipe/train.py` | Model training | Train on full dataset |
| `pipe/collective_matrix_factorisation.py` | Grid search | Tune hyperparameters |
| `pipe/submit.py` | Generate submission | Create Kaggle submission |
| `models/cmf_recommender.py` | Model wrapper | Load and use trained model |
| `data_preprocessing/clean_dataset.py` | Data preprocessing | Clean raw data |

---

## 💡 Common Workflows

### First Time Setup
```bash
git clone <repo-url>
cd lazy_librarian_kaggle
uv sync
uv run streamlit run front/main.py
```

### Daily Development
```bash
cd lazy_librarian_kaggle
source .venv/bin/activate  # If using pip
streamlit run front/main.py
```

### Creating a Submission
```bash
uv run python pipe/train.py          # Train model
uv run python pipe/submit.py         # Generate submission
# Upload submission_cmf_enriched.csv to Kaggle
```

### Experimenting with Hyperparameters
```bash
# Edit pipe/collective_matrix_factorisation.py
# Modify param_grid dictionary
uv run python pipe/collective_matrix_factorisation.py
# Review output for best MAP@10
```

---

## License

This project was developed for educational purposes as part of a recommendation systems course.