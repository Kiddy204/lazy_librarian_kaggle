import sys
import os

from eval import compute_map_at_k_als
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from implicit.als import AlternatingLeastSquares

from data_preprocessing.clean_interactions import clean_interactions
from data_preprocessing.clean_items import clean_items
from split import create_train_test_split
from submit import generate_submission

"""
Hybrid Re-ranking: ALS + User-User CF + Item-Item CF + User-Subject Signal
Optimized for MAP@10 (implicit feedback)

Requirements:
- pandas
- numpy
- scipy
- implicit
- sklearn
"""

# ============================================================
# Utilities
# ============================================================

def build_interaction_matrix(df, n_users, n_items, weight_col=None):
    data = (
        df[weight_col].astype(np.float32).values
        if weight_col
        else np.ones(len(df), dtype=np.float32)
    )
    return csr_matrix(
        (data, (df.user_idx.values, df.item_idx.values)),
        shape=(n_users, n_items)
    )


def compute_map_at_k(scores, test_matrix, k=10):
    average_precisions = []

    for u in range(scores.shape[0]):
        true_items = test_matrix[u].indices
        if len(true_items) == 0:
            continue

        ranked_items = np.argsort(scores[u])[::-1][:k]

        hits = 0
        score = 0.0
        for i, item in enumerate(ranked_items):
            if item in true_items:
                hits += 1
                score += hits / (i + 1)

        average_precisions.append(score / min(len(true_items), k))

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def zscore_per_user(scores, eps=1e-8):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + eps
    return (scores - mean) / std


def normalize_us_score(us_s, max_overlap):
    return us_s / max_overlap if max_overlap > 0 else 0.0


# ============================================================
# User-User CF
# ============================================================

def user_user_scores(train_matrix):
    """
    Compute user-user collaborative filtering scores.
    Uses cosine similarity between users and propagates preferences.
    """
    sim = cosine_similarity(train_matrix)
    np.fill_diagonal(sim, 0.0)
    return sim @ train_matrix


# ============================================================
# Item-Item CF
# ============================================================

def item_item_scores(train_matrix, top_k_similar=100):
    """
    Compute item-item collaborative filtering scores.
    Uses cosine similarity between items and propagates user preferences.
    
    For each user, the score for an item is the weighted sum of similarities
    to items the user has interacted with.
    
    Parameters:
    -----------
    train_matrix : csr_matrix
        User-item interaction matrix (n_users x n_items)
    top_k_similar : int
        Number of top similar items to consider for each item (for efficiency)
    
    Returns:
    --------
    np.ndarray
        Score matrix (n_users x n_items)
    """
    # Compute item-item similarity (transpose to get item x user, then similarity)
    item_sim = cosine_similarity(train_matrix.T)
    np.fill_diagonal(item_sim, 0.0)
    
    # Optional: Keep only top-k similar items for efficiency and noise reduction
    if top_k_similar is not None and top_k_similar < item_sim.shape[0]:
        for i in range(item_sim.shape[0]):
            # Get indices of items NOT in top-k
            top_k_indices = np.argpartition(item_sim[i], -top_k_similar)[-top_k_similar:]
            mask = np.ones(item_sim.shape[1], dtype=bool)
            mask[top_k_indices] = False
            item_sim[i, mask] = 0.0
    
    # Score = user interactions @ item similarity
    # For each user u and item i: score[u,i] = sum over j of (R[u,j] * sim[j,i])
    return train_matrix @ item_sim


# ============================================================
# User-Subject Matrix
# ============================================================

def build_user_subject_matrix(interactions, items_clean):
    """
    Builds user × subject matrix from item metadata.
    Assumes items_clean['subjects_list'] is a list[str] or NaN.
    """
    # Map item -> list of subjects
    item_subjects = (
        items_clean
        .set_index("i")["subjects_list"]
        .dropna()
    )

    # Build subject vocabulary
    subject_set = set()
    for subs in item_subjects:
        for s in subs:
            subject_set.add(s.strip().lower())

    subject_to_idx = {s: i for i, s in enumerate(sorted(subject_set))}

    rows, cols, data = [], [], []

    for row in interactions.itertuples():
        subjects = item_subjects.get(row.i)
        if subjects is None:
            continue

        for s in subjects:
            s = s.strip().lower()
            idx = subject_to_idx.get(s)
            if idx is not None:
                rows.append(row.user_idx)
                cols.append(idx)
                data.append(1.0)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(
            interactions.user_idx.max() + 1,
            len(subject_to_idx)
        )
    ), subject_to_idx

def build_item_subject_index(items_clean, subject_to_idx, item_id_to_idx):
    """
    Build mapping: item_idx -> list of subject indices
    
    Parameters:
    -----------
    items_clean : pd.DataFrame
        Items dataframe with 'i' (item ID) and 'subjects_list' columns
    subject_to_idx : dict
        Mapping from subject string to subject index
    item_id_to_idx : dict
        Mapping from item ID to item index (0 to n_items-1)
    
    Returns:
    --------
    dict
        Mapping from item_idx to list of subject indices
    """
    item_subject_indices = {}

    for row in items_clean.itertuples():
        if not isinstance(row.subjects_list, list):
            continue
        
        # Convert item ID to item index
        item_idx = item_id_to_idx.get(row.i)
        if item_idx is None:
            continue

        idxs = []
        for s in row.subjects_list:
            s = s.strip().lower()
            if s in subject_to_idx:
                idxs.append(subject_to_idx[s])

        if idxs:
            item_subject_indices[item_idx] = idxs  # Now uses item_idx

    return item_subject_indices
# ============================================================
# ALS Candidate Generator
# ============================================================

# def train_als(train_matrix, factors=512, reg=0.05, alpha=320, iterations=30):
def train_als(train_matrix, factors=192, reg=0.05, alpha=160, iterations=30):

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=42,
        use_gpu=False
    )
    model.fit((train_matrix * alpha).T)

    user_factors = model.item_factors
    item_factors = model.user_factors

    return user_factors @ item_factors.T


# ============================================================
# Hybrid Re-ranking (with Item-Item CF)
# ============================================================
def hybrid_rerank(
    als_scores,
    uu_scores,
    ii_scores,
    us_matrix,
    item_subject_indices,
    top_n=200,
    w_als=0.5,
    w_uu=0.2,
    w_ii=0.2,
    w_us=0.1
):
    """
    Hybrid re-ranking combining multiple signals:
    - ALS (matrix factorization)
    - User-User CF
    - Item-Item CF
    - User-Subject overlap
    
    All input score matrices should be z-score normalized.
    """
    n_users, n_items = als_scores.shape
    final_scores = np.full_like(als_scores, -np.inf)

    us_matrix = us_matrix.tocsr()

    # Precompute user-subject overlap scores
    us_scores = np.zeros((n_users, n_items), dtype=np.float32)
    
    for u in range(n_users):
        user_subjects = set(us_matrix[u].indices)
        if not user_subjects:
            continue
        for i, subs in item_subject_indices.items():
            overlap = len(user_subjects.intersection(subs))
            us_scores[u, i] = overlap
    
    # Z-score normalize to match other signals' scale
    us_scores = zscore_per_user(us_scores)

    for u in range(n_users):
        # Union of top candidates from all signals
        als_candidates = set(np.argsort(als_scores[u])[::-1][:top_n])
        uu_candidates = set(np.argsort(uu_scores[u])[::-1][:top_n])
        ii_candidates = set(np.argsort(ii_scores[u])[::-1][:top_n])
        us_candidates = set(np.argsort(us_scores[u])[::-1][:top_n])
        
        candidates = als_candidates | uu_candidates | ii_candidates | us_candidates

        for i in candidates:
            final_scores[u, i] = (
                w_als * als_scores[u, i] +
                w_uu  * uu_scores[u, i] +
                w_ii  * ii_scores[u, i] +
                w_us  * us_scores[u, i]
            )

    return final_scores
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    items_df = pd.read_csv("data/items.csv")
    interactions_df = pd.read_csv("data/interactions.csv")

    items_clean = clean_items(items_df, interactions_df, remove_subjects=False)
    interactions_clean = clean_interactions(interactions_df)

    users = interactions_clean["u"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(users)}
    interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)

    item_id_to_idx = {i: idx for idx, i in enumerate(items_clean["i"])}
    interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)

    n_users = interactions_clean["user_idx"].nunique()
    n_items = interactions_clean["item_idx"].nunique()

    train_data, test_data, val_data = create_train_test_split(
        interactions_clean, test_size=0.2, val_size=0.2
    )

    train_matrix = build_interaction_matrix(
        train_data, n_users, n_items, weight_col="interaction_rank"
    )
    test_matrix = build_interaction_matrix(
        test_data, n_users, n_items
    )
    val_matrix = build_interaction_matrix(
        val_data, n_users, n_items
    )

    print("Training ALS...")
    als_scores = train_als(train_matrix)

    print("Computing User-User CF...")
    uu_scores = user_user_scores(train_matrix)

    print("Computing Item-Item CF...")
    ii_scores = item_item_scores(train_matrix, top_k_similar=100)

    print("Building User-Subject matrix...")
    us_matrix, subject_to_idx = build_user_subject_matrix(train_data, items_clean)

    item_subject_indices = build_item_subject_index(items_clean, subject_to_idx, item_id_to_idx)


    # Z-score normalize all score matrices
    als_scores = np.asarray(train_als(train_matrix))
    uu_scores = np.asarray(user_user_scores(train_matrix))
    ii_scores = np.asarray(item_item_scores(train_matrix, top_k_similar=100))


    als_scores = zscore_per_user(als_scores)
    uu_scores = zscore_per_user(uu_scores)
    ii_scores = zscore_per_user(ii_scores)
    # Initial hybrid evaluation
    final_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        ii_scores,
        us_matrix,
        item_subject_indices
    )

    map10 = compute_map_at_k(final_scores, test_matrix)
    print(f"\nHybrid MAP@10: {map10:.5f}")

    val_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        ii_scores,
        us_matrix,
        item_subject_indices
    )

    val_map10 = compute_map_at_k(val_scores, val_matrix, k=10)
    print(f"Validation MAP@10: {val_map10:.5f}")

    print("Generating submission...")
    submission_df = generate_submission(
        predictions=final_scores,
        user_index=user_id_to_idx,
        item_index=item_id_to_idx,
        k=10
    )

    submission_path = "submission_hybrid_ranking.csv"
    submission_df.to_csv(submission_path, index=False)

    # ============================================================
    # Grid Search for Best Weights
    # ============================================================
    print("\n" + "=" * 60)
    print("Grid Search for Optimal Weights")
    print("=" * 60)

    best_score = 0
    best_weights = None

    # for w_als in [1, 0.6]:
    #     for w_uu in [0, 0.2, 0.1,0.3]:
    #         for w_ii in [0,0.15, 0.3,0.1,0.2]:
    #             for w_us in [0, 0.15,0.3,0.1,0.1]:
    #                 if w_als + w_uu + w_ii + w_us != 1:
    #                     continue
           
    #                 scores = hybrid_rerank(
    #                     als_scores,
    #                     uu_scores,
    #                     ii_scores,
    #                     us_matrix,
    #                     item_subject_indices,
    #                     w_als=w_als,
    #                     w_uu=w_uu,
    #                     w_ii=w_ii,
    #                     w_us=w_us
    #                 )

    #                 map10 = compute_map_at_k(scores, val_matrix)

    #                 print(
    #                     f"ALS={w_als:.2f}, UU={w_uu:.2f}, II={w_ii:.2f}, US={w_us:.2f} "
    #                     f"=> VAL MAP@10={map10:.5f}"
    #                 )

    #                 if map10 > best_score:
    #                     best_score = map10
    #                     best_weights = (w_als, w_uu, w_ii, w_us)

    # print("\n" + "=" * 60)
    # print(f"Best Weights: ALS={best_weights[0]:.2f}, UU={best_weights[1]:.2f}, "
    #       f"II={best_weights[2]:.2f}, US={best_weights[3]:.2f}")
    # print(f"Best Validation MAP@10: {best_score:.5f}")
    # print("=" * 60)

    # ============================================================
    # Final Model with Best Weights (Optional: Train on full data)
    # ============================================================
    # Uncomment below to retrain on combined train+val data with best weights

    full_train = pd.concat([train_data, val_data])
    
    full_train_matrix = build_interaction_matrix(
        full_train, n_users, n_items, weight_col="interaction_rank"
    )
    
    print("\nRetraining on full data with best weights...")
    als_scores = train_als(full_train_matrix)
    uu_scores = user_user_scores(full_train_matrix)
    ii_scores = item_item_scores(full_train_matrix, top_k_similar=100)
    us_matrix, subject_to_idx = build_user_subject_matrix(full_train, items_clean)
    item_subject_indices = build_item_subject_index(items_clean, subject_to_idx, item_id_to_idx)

    als_scores = zscore_per_user(als_scores)
    uu_scores = zscore_per_user(uu_scores)
    ii_scores = zscore_per_user(ii_scores)
    
    final_scores = hybrid_rerank(
        als_scores,
        uu_scores,
        ii_scores,
        us_matrix,
        item_subject_indices,
        w_als=1,
        w_uu=0,
        w_ii=0,
        w_us=0,
    )
    
    submission_df = generate_submission(
        predictions=final_scores,
        user_index=user_id_to_idx,
        item_index=item_id_to_idx,
        k=10
    )
    
    submission_df.to_csv("submission_als_final.csv", index=False)
    print("Final submission saved to submission_hybrid_final.csv")