"""
Train CMF model on full data and generate submission.
Supports TF-IDF subjects and title embeddings.
"""
import pickle
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.sparse import csr_matrix

from data_preprocessing.clean_interactions import clean_interactions
from data_preprocessing.clean_items import clean_items
from data_preprocessing.encode_categorical import encode_authors, encode_publishers
from submit import generate_submission

from collective_matrix_factorisation import (
    CMFConfig,
    CollectiveMatrixFactorization,
    build_user_author_matrix,
    build_user_publisher_matrix,
    build_user_subject_matrix_tfidf,
    validate_and_align_indices,
    load_and_reindex_tfidf_subjects,
    load_and_reindex_title_embeddings,
    build_user_embedding_matrix,
    predict_with_embeddings
)


def train_on_full_data(
    interactions_df,
    items_df,
    n_users: int,
    n_items: int,
    n_authors: int,
    n_publishers: int,
    config: CMFConfig,
    item_subjects: np.ndarray
):
    """Train CMF on full interaction data (no train/test split)."""
    
    # Build user-item matrix from ALL interactions
    user_item_matrix = csr_matrix(
        (np.ones(len(interactions_df), dtype=np.float32),
         (interactions_df["user_idx"].values, interactions_df["item_idx"].values)),
        shape=(n_users, n_items)
    )
    
    user_author_matrix = build_user_author_matrix(
        interactions_df, items_df, n_users, n_authors
    )
    
    user_publisher_matrix = build_user_publisher_matrix(
        interactions_df, items_df, n_users, n_publishers
    )
    
    user_subject_matrix = build_user_subject_matrix_tfidf(
        interactions_df, item_subjects, n_users
    )
    
    # Train CMF
    model = CollectiveMatrixFactorization(config)
    model.fit(
        user_item_matrix,
        user_author_matrix,
        user_publisher_matrix,
        user_subject_matrix,
        verbose=True
    )
    
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("TRAINING CMF ON FULL DATA")
    print("(TF-IDF Subjects + Title Embeddings)")
    print("=" * 50)
    
    # ============================================================
    # Configuration
    # ============================================================
    TFIDF_PATH = "data/_/subject_tfidf.npz"
    PARQUET_PATH = "data/_/items_with_topics.parquet"
    EMBEDDINGS_PATH = "data/_/title_embeddings.npy"
    
    USE_TFIDF_SUBJECTS = True
    USE_TITLE_EMBEDDINGS = True
    
    # Best embedding weight from grid search (update based on your results)
    W_EMBEDDING = 0.15
    
    # ============================================================
    # Load data
    # ============================================================
    print("\nLoading data...")
    items_df = pd.read_csv("data/items.csv")
    interactions_df = pd.read_csv("data/interactions.csv")
    
    # Preprocess
    items_clean = clean_items(items_df, interactions_df)
    interactions_clean = clean_interactions(interactions_df)
    
    # Encode categorical features
    encoded_authors, author_encoder = encode_authors(items_clean)
    items_clean["author_encoded"] = encoded_authors
    
    encoded_publishers, publisher_encoder = encode_publishers(items_clean)
    items_clean["publisher_encoded"] = encoded_publishers
    
    # Create item index mapping
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items_clean["i"].values)}
    item_idx_to_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    items_clean["item_idx"] = items_clean["i"].map(item_id_to_idx)
    
    # ============================================================
    # Load TF-IDF subjects
    # ============================================================
    if USE_TFIDF_SUBJECTS:
        print("\nLoading TF-IDF subjects...")
        item_subjects, n_subjects = load_and_reindex_tfidf_subjects(
            TFIDF_PATH, PARQUET_PATH, item_id_to_idx
        )
        subject_to_idx = None
    else:
        print("\nUsing multi-hot subjects...")
        from data_preprocessing.encode_subjects import clean_subjects, encode_subjects
        items_clean, valid_subjects = clean_subjects(items_clean, min_count=6)
        item_subjects, subject_to_idx = encode_subjects(items_clean, valid_subjects)
        n_subjects = len(subject_to_idx)
    
    # ============================================================
    # Load title embeddings
    # ============================================================
    item_embeddings = None
    user_embeddings = None
    
    if USE_TITLE_EMBEDDINGS:
        print("\nLoading title embeddings...")
        item_embeddings = load_and_reindex_title_embeddings(
            EMBEDDINGS_PATH, PARQUET_PATH, item_id_to_idx
        )
    
    # Filter interactions to known items
    interactions_clean = interactions_clean[
        interactions_clean["i"].isin(item_id_to_idx)
    ].copy()
    interactions_clean["item_idx"] = interactions_clean["i"].map(item_id_to_idx)
    
    # Create user index mapping
    users = interactions_clean["u"].unique()
    user_id_to_idx = {u: i for i, u in enumerate(users)}
    user_idx_to_id = {i: u for u, i in user_id_to_idx.items()}
    interactions_clean["user_idx"] = interactions_clean["u"].map(user_id_to_idx)
    
    # Dimensions
    n_users = len(users)
    n_items = len(item_id_to_idx)
    n_authors = int(items_clean["author_encoded"].max()) + 1
    n_publishers = int(items_clean["publisher_encoded"].max()) + 1
    
    # Build user embeddings from ALL interactions
    if USE_TITLE_EMBEDDINGS:
        print("\nBuilding user embeddings...")
        user_embeddings = build_user_embedding_matrix(
            interactions_clean, item_embeddings, n_users, aggregation="mean"
        )
        print(f"✓ User embeddings: {user_embeddings.shape}")
    
    # Validate indices
    items_clean = validate_and_align_indices(
        items_clean, interactions_clean, n_items, n_authors, n_publishers,
        item_subjects=item_subjects,
        n_subjects=n_subjects,
        item_embeddings=item_embeddings
    )
    
    print(f"\nDataset: {n_users} users, {n_items} items, {n_authors} authors, {n_publishers} publishers, {n_subjects} subjects")
    if item_embeddings is not None:
        print(f"Embeddings: {item_embeddings.shape[1]} dims")
    print(f"Total interactions: {len(interactions_clean)}")
    
    # Best config from grid search (update these values based on your results)
    best_config = CMFConfig(
        factors=1024,
        iterations=20,
        lambda_author=0.5,
        lambda_publisher=0.5,
        lambda_subject=0.8,
        alpha=380
    )
    
    print(f"\nConfig: factors={best_config.factors}, λ_a={best_config.lambda_author}, "
          f"λ_p={best_config.lambda_publisher}, λ_s={best_config.lambda_subject}, α={best_config.alpha}")
    if USE_TITLE_EMBEDDINGS:
        print(f"Embedding weight: {W_EMBEDDING}")
    
    # Train on full data
    print("\nTraining on full data...")
    model = train_on_full_data(
        interactions_clean, items_clean,
        n_users, n_items, n_authors, n_publishers,
        best_config, item_subjects
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    if USE_TITLE_EMBEDDINGS:
        final_scores = predict_with_embeddings(
            model, items_clean, item_subjects,
            item_embeddings, user_embeddings,
            w_embedding=W_EMBEDDING
        )
    else:
        final_scores = model.predict_scores(items_clean, item_subjects=item_subjects)
    
    # Generate submission
    print("\nGenerating submission...")
    submission_df = generate_submission(
        predictions=final_scores,
        user_index=user_id_to_idx,
        item_index=item_id_to_idx,
        k=10
    )
    
    submission_path = "data/submissions/submission_cmf_full_enriched.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✓ Saved submission to {submission_path}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Users: {submission_df['user_id'].nunique()}")
    print(f"  Sample:\n{submission_df.head(10)}")

    # Save model and artifacts
    print("\nSaving model...")
    model_path = "models/cmf_full_model_enriched.pkl"
    os.makedirs("models", exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "config": best_config,
            "item_subjects": item_subjects,
            "item_embeddings": item_embeddings,
            "user_embeddings": user_embeddings,
            "user_id_to_idx": user_id_to_idx,
            "item_id_to_idx": item_id_to_idx,
            "subject_to_idx": subject_to_idx,
            "use_tfidf": USE_TFIDF_SUBJECTS,
            "use_embeddings": USE_TITLE_EMBEDDINGS,
            "w_embedding": W_EMBEDDING
        }, f)

    print(f"✓ Saved model to {model_path}")