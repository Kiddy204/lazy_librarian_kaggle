"""
Subject encoding with frequency-based filtering.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, List


def clean_subjects(items_df: pd.DataFrame, min_count: int = 6) -> pd.DataFrame:
    """
    Clean subjects by removing those mentioned fewer than min_count times.
    
    Args:
        items_df: DataFrame with 'subjects_list' column (already parsed as list)
        min_count: Minimum frequency threshold
    
    Returns:
        DataFrame with cleaned 'subjects_clean' column
    """
    items_df = items_df.copy()
    
    # Count all subject occurrences
    all_subjects = []
    for subjects in items_df["subjects_list"]:
        if isinstance(subjects, list):
            all_subjects.extend(subjects)
    
    subject_counts = Counter(all_subjects)
    
    # Filter to keep only frequent subjects
    valid_subjects = {s for s, count in subject_counts.items() if count >= min_count}
    
    # Stats
    total_unique = len(subject_counts)
    kept_unique = len(valid_subjects)
    print(f"Subjects: {total_unique} unique -> {kept_unique} kept (min_count={min_count})")
    
    # Filter each row's subjects
    def filter_subjects(subjects_list):
        if not isinstance(subjects_list, list):
            return []
        return [s for s in subjects_list if s in valid_subjects]
    
    items_df["subjects_clean"] = items_df["subjects_list"].apply(filter_subjects)
    
    return items_df, valid_subjects


def encode_subjects(
    items_df: pd.DataFrame,
    valid_subjects: set
) -> Tuple[np.ndarray, dict]:
    """
    Encode subjects as multi-hot vectors using item_idx for alignment.
    """
    subject_to_idx = {s: i for i, s in enumerate(sorted(valid_subjects))}
    n_items = int(items_df["item_idx"].max()) + 1  # Use item_idx range
    n_subjects = len(subject_to_idx)
    
    multi_hot = np.zeros((n_items, n_subjects), dtype=np.float32)
    
    for _, row in items_df.iterrows():
        item_idx = int(row["item_idx"])  # Use item_idx, not row position
        for subj in row["subjects_clean"]:
            if subj in subject_to_idx:
                multi_hot[item_idx, subject_to_idx[subj]] = 1.0
    
    print(f"Encoded subjects: {n_items} items x {n_subjects} subjects")
    
    return multi_hot, subject_to_idx


def build_user_subject_matrix(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    item_subjects: np.ndarray,  # (n_items, n_subjects) multi-hot
    n_users: int
) -> "csr_matrix":
    """
    Build user-subject interaction matrix.
    
    If user interacted with item, they implicitly interacted with all its subjects.
    """
    from scipy.sparse import csr_matrix
    
    n_subjects = item_subjects.shape[1]
    
    # Aggregate: for each user, sum up subjects from all items they interacted with
    user_subject = np.zeros((n_users, n_subjects), dtype=np.float32)
    
    for _, row in interactions_df.iterrows():
        user_idx = row["user_idx"]
        item_idx = row["item_idx"]
        user_subject[user_idx] += item_subjects[item_idx]
    
    return csr_matrix(user_subject)


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    items_df = pd.read_csv("data/items.csv")
    
    # Clean subjects
    items_df, valid_subjects = clean_subjects(items_df, min_count=6)
    
    # Encode as multi-hot
    item_subjects, subject_to_idx = encode_subjects(items_df, valid_subjects)
    
    print(f"\nSample item subjects (first 3 items):")
    for i in range(min(3, len(items_df))):
        print(f"  Item {i}: {items_df['subjects_clean'].iloc[i][:3]}...")  # First 3 subjects
