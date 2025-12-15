"""
Collective Matrix Factorization (CMF) for recommendations with side information.

Jointly factorizes:
    - User-Item matrix:      R ≈ U × V^T
    - User-Author matrix:    A ≈ U × Author^T
    - User-Publisher matrix: P ≈ U × Pub^T
    - User-Subject matrix:   S ≈ U × Subject^T

User factors U are shared, enabling knowledge transfer.
Title embeddings are used as content-based signal combined with CMF scores.
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval import compute_map_at_k_als

import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CMFConfig:
    """Configuration for CMF training."""
    factors: int = 128
    iterations: int = 30
    learning_rate: float = 0.01
    regularization: float = 0.05
    
    # Weights for auxiliary matrices
    lambda_author: float = 0.3
    lambda_publisher: float = 0.2
    lambda_subject: float = 0.2

    # For implicit feedback
    alpha: float = 40.0
    use_confidence: bool = True
    
    random_state: int = 42


class CollectiveMatrixFactorization:
    """
    Collective Matrix Factorization using ALS.
    
    Shares user factors across multiple matrices to incorporate
    side information (authors, publishers, subjects, etc.)
    """
    
    def __init__(self, config: CMFConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_state)
        
        # Learned factors (set during fit)
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.author_factors: Optional[np.ndarray] = None
        self.publisher_factors: Optional[np.ndarray] = None
        self.subject_factors: Optional[np.ndarray] = None

        
    def _init_factors(self, n: int) -> np.ndarray:
        """Initialize factor matrix with small random values."""
        scale = 1.0 / np.sqrt(self.config.factors)
        return self.rng.normal(0, scale, (n, self.config.factors)).astype(np.float32)
    
    def _compute_confidence(self, matrix: csr_matrix) -> csr_matrix:
        """Convert interactions to confidence values: c = 1 + alpha * r"""
        if self.config.use_confidence:
            confidence = matrix.copy()
            confidence.data = 1.0 + self.config.alpha * confidence.data
            return confidence
        return matrix
    
    def fit(
        self,
        user_item_matrix: csr_matrix,
        user_author_matrix: Optional[csr_matrix] = None,
        user_publisher_matrix: Optional[csr_matrix] = None,
        user_subject_matrix: Optional[csr_matrix] = None,
        verbose: bool = True
    ) -> "CollectiveMatrixFactorization":
        """
        Fit the CMF model using Alternating Least Squares.
        
        Args:
            user_item_matrix: Primary interaction matrix (n_users, n_items)
            user_author_matrix: User-author interactions (n_users, n_authors)
            user_publisher_matrix: User-publisher interactions (n_users, n_publishers)
            user_subject_matrix: User-subject interactions (n_users, n_subjects)
            verbose: Print progress
        """
        n_users, n_items = user_item_matrix.shape
        
        # Initialize all factors
        self.user_factors = self._init_factors(n_users)
        self.item_factors = self._init_factors(n_items)
        
        if user_author_matrix is not None:
            n_authors = user_author_matrix.shape[1]
            self.author_factors = self._init_factors(n_authors)
            
        if user_publisher_matrix is not None:
            n_publishers = user_publisher_matrix.shape[1]
            self.publisher_factors = self._init_factors(n_publishers)

        if user_subject_matrix is not None:
            n_subjects = user_subject_matrix.shape[1]
            self.subject_factors = self._init_factors(n_subjects)
            
        # Convert to confidence matrices
        C_item = self._compute_confidence(user_item_matrix)
        C_author = self._compute_confidence(user_author_matrix) if user_author_matrix is not None else None
        C_publisher = self._compute_confidence(user_publisher_matrix) if user_publisher_matrix is not None else None
        C_subject = self._compute_confidence(user_subject_matrix) if user_subject_matrix is not None else None

        # Binary preference matrices (1 if interaction exists)
        P_item = (user_item_matrix > 0).astype(np.float32)
        P_author = (user_author_matrix > 0).astype(np.float32) if user_author_matrix is not None else None
        P_publisher = (user_publisher_matrix > 0).astype(np.float32) if user_publisher_matrix is not None else None
        P_subject = (user_subject_matrix > 0).astype(np.float32) if user_subject_matrix is not None else None

        # ALS iterations
        for iteration in range(self.config.iterations):
            # Update item factors (standard ALS update)
            self.item_factors = self._als_update_factors(
                C_item.T.tocsr(), P_item.T.tocsr(), 
                self.user_factors, self.item_factors,
                weight=1.0
            )
            
            # Update author factors
            if C_author is not None:
                self.author_factors = self._als_update_factors(
                    C_author.T.tocsr(), P_author.T.tocsr(),
                    self.user_factors, self.author_factors,
                    weight=self.config.lambda_author
                )
            
            # Update publisher factors
            if C_publisher is not None:
                self.publisher_factors = self._als_update_factors(
                    C_publisher.T.tocsr(), P_publisher.T.tocsr(),
                    self.user_factors, self.publisher_factors,
                    weight=self.config.lambda_publisher
                )
            
            # Update subject factors
            if C_subject is not None:
                self.subject_factors = self._als_update_factors(
                    C_subject.T.tocsr(), P_subject.T.tocsr(),
                    self.user_factors, self.subject_factors,
                    weight=self.config.lambda_subject
                )

            # Update user factors (considering ALL matrices)
            self.user_factors = self._als_update_user_factors(
                C_item, P_item,
                C_author, P_author,
                C_publisher, P_publisher, 
                C_subject, P_subject
            )
            
            if verbose:
                loss = self._compute_loss(
                    C_item, P_item, C_author, P_author, C_publisher, P_publisher, C_subject, P_subject
                )
                print(f"Iteration {iteration + 1}/{self.config.iterations}, Loss: {loss:.4f}")
        
        return self
    
    def _als_update_factors(
        self,
        C_T: csr_matrix,  # Transposed confidence matrix
        P_T: csr_matrix,  # Transposed preference matrix
        fixed_factors: np.ndarray,  # The factors we're using (not updating)
        updating_factors: np.ndarray,  # The factors we're updating
        weight: float = 1.0
    ) -> np.ndarray:
        """
        ALS update for one set of factors.
        
        For updating item factors given fixed user factors:
            v_i = (U^T C^i U + λI)^{-1} U^T C^i p_i
        """
        n = updating_factors.shape[0]
        k = self.config.factors
        reg = self.config.regularization * weight
        
        # Precompute U^T U
        UTU = fixed_factors.T @ fixed_factors
        
        new_factors = np.zeros_like(updating_factors)
        
        for i in range(n):
            # Get indices and confidence values for this row
            start, end = C_T.indptr[i], C_T.indptr[i + 1]
            indices = C_T.indices[start:end]
            confidence = C_T.data[start:end]
            preferences = P_T.data[start:end] if len(P_T.data) > 0 else np.ones_like(confidence)
            
            if len(indices) == 0:
                # No interactions, just use regularization
                new_factors[i] = 0
                continue
            
            # U_i = user factors for users who interacted with item i
            U_i = fixed_factors[indices]  # (n_interactions, k)
            
            # C^i - I is diagonal with (c_ui - 1) values
            # (U^T C^i U) = U^T U + U_i^T (C^i - I) U_i
            C_minus_I = confidence - 1  # (c - 1) values
            
            # A = U^T U + U_i^T diag(c-1) U_i + λI
            A = UTU + (U_i.T * C_minus_I) @ U_i + reg * np.eye(k)
            
            # b = U_i^T C^i p_i = U_i^T diag(c) p
            b = U_i.T @ (confidence * preferences)
            
            # Solve for new factor
            new_factors[i] = np.linalg.solve(A, b)
        
        return new_factors.astype(np.float32)
    
    def _als_update_user_factors(
        self,
        C_item: csr_matrix,
        P_item: csr_matrix,
        C_author: Optional[csr_matrix],
        P_author: Optional[csr_matrix],
        C_publisher: Optional[csr_matrix],
        P_publisher: Optional[csr_matrix],
        C_subject: Optional[csr_matrix] = None,
        P_subject: Optional[csr_matrix] = None
    ) -> np.ndarray:
        """
        Update user factors considering ALL matrices.
        
        The key insight: user factors appear in all factorizations,
        so we combine gradients from all of them.
        """
        n_users = self.user_factors.shape[0]
        k = self.config.factors
        reg = self.config.regularization
        
        # Precompute F^T F for all factor matrices
        VTV = self.item_factors.T @ self.item_factors
        ATA = self.author_factors.T @ self.author_factors if self.author_factors is not None else None
        PTP = self.publisher_factors.T @ self.publisher_factors if self.publisher_factors is not None else None
        STS = self.subject_factors.T @ self.subject_factors if self.subject_factors is not None else None
        
        new_user_factors = np.zeros_like(self.user_factors)
        
        for u in range(n_users):
            A_combined = reg * np.eye(k)
            b_combined = np.zeros(k)
            
            # Contribution from user-item matrix
            A_u, b_u = self._user_contribution(
                u, C_item, P_item, self.item_factors, VTV, weight=1.0
            )
            A_combined += A_u
            b_combined += b_u
            
            # Contribution from user-author matrix
            if C_author is not None and self.author_factors is not None:
                A_a, b_a = self._user_contribution(
                    u, C_author, P_author, self.author_factors, ATA,
                    weight=self.config.lambda_author
                )
                A_combined += A_a
                b_combined += b_a
            
            # Contribution from user-publisher matrix
            if C_publisher is not None and self.publisher_factors is not None:
                A_p, b_p = self._user_contribution(
                    u, C_publisher, P_publisher, self.publisher_factors, PTP,
                    weight=self.config.lambda_publisher
                )
                A_combined += A_p
                b_combined += b_p
            
            # Contribution from user-subject matrix
            if C_subject is not None and self.subject_factors is not None:
                A_s, b_s = self._user_contribution(
                    u, C_subject, P_subject, self.subject_factors, STS,
                    weight=self.config.lambda_subject
                )
                A_combined += A_s
                b_combined += b_s
                
            # Solve combined system
            new_user_factors[u] = np.linalg.solve(A_combined, b_combined)
        
        return new_user_factors.astype(np.float32)
    
    def _user_contribution(
        self,
        user_idx: int,
        C: csr_matrix,
        P: csr_matrix,
        factors: np.ndarray,
        FTF: np.ndarray,
        weight: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute A and b contribution for a single user from one matrix."""
        k = self.config.factors
        
        start, end = C.indptr[user_idx], C.indptr[user_idx + 1]
        indices = C.indices[start:end]
        confidence = C.data[start:end]
        preferences = P.data[start:end] if end > start else np.array([])
        
        if len(indices) == 0:
            return np.zeros((k, k)), np.zeros(k)
        
        F_u = factors[indices]  # Factors for items/authors/publishers this user interacted with
        C_minus_I = confidence - 1
        
        # A contribution: weight * (F^T F + F_u^T diag(c-1) F_u)
        A = weight * (FTF + (F_u.T * C_minus_I) @ F_u)
        
        # b contribution: weight * F_u^T diag(c) p
        b = weight * (F_u.T @ (confidence * preferences))
        
        return A, b
    
    def _compute_loss(
        self,
        C_item, P_item,
        C_author, P_author,
        C_publisher, P_publisher,
        C_subject, P_subject
    ) -> float:
        """Compute total weighted loss for monitoring convergence."""
        loss = 0.0
        
        # Item loss
        loss += self._matrix_loss(C_item, P_item, self.user_factors, self.item_factors)
        
        # Author loss
        if C_author is not None and self.author_factors is not None:
            loss += self.config.lambda_author * self._matrix_loss(
                C_author, P_author, self.user_factors, self.author_factors
            )
        
        # Publisher loss
        if C_publisher is not None and self.publisher_factors is not None:
            loss += self.config.lambda_publisher * self._matrix_loss(
                C_publisher, P_publisher, self.user_factors, self.publisher_factors
            )
        
        # Subject loss
        if C_subject is not None and self.subject_factors is not None:
            loss += self.config.lambda_subject * self._matrix_loss(
                C_subject, P_subject, self.user_factors, self.subject_factors
            )
            
        # Regularization
        reg = self.config.regularization
        loss += reg * (
            np.sum(self.user_factors ** 2) +
            np.sum(self.item_factors ** 2)
        )
        if self.author_factors is not None:
            loss += reg * np.sum(self.author_factors ** 2)
        if self.publisher_factors is not None:
            loss += reg * np.sum(self.publisher_factors ** 2)
        if self.subject_factors is not None:
            loss += reg * np.sum(self.subject_factors ** 2)
        
        return loss
    
    def _matrix_loss(
        self,
        C: csr_matrix,
        P: csr_matrix,
        U: np.ndarray,
        V: np.ndarray
    ) -> float:
        """Compute weighted squared error for one matrix."""
        loss = 0.0
        for u in range(U.shape[0]):
            start, end = C.indptr[u], C.indptr[u + 1]
            if start == end:
                continue
            indices = C.indices[start:end]
            confidence = C.data[start:end]
            preferences = P.data[start:end]
            
            predictions = U[u] @ V[indices].T
            errors = preferences - predictions
            loss += np.sum(confidence * (errors ** 2))
        
        return loss
    
    def predict_scores(
        self,
        items_df,
        item_subjects: Optional[np.ndarray] = None,
        w_item: float = 1.0,
        w_author: float = 0.3,
        w_publisher: float = 0.2,
        w_subject: float = 0.2
    ) -> np.ndarray:
        """
        Compute final recommendation scores combining all signals.
        
        Args:
            items_df: DataFrame with 'author_encoded' and 'publisher_encoded' columns
            item_subjects: (n_items, n_subjects) matrix (TF-IDF or multi-hot)
            w_item, w_author, w_publisher, w_subject: Combination weights
            
        Returns:
            scores: (n_users, n_items) prediction matrix
        """
        # Direct user-item scores
        scores = w_item * (self.user_factors @ self.item_factors.T)
        
        # Add author-based scores
        if self.author_factors is not None and "author_encoded" in items_df.columns:
            user_author_scores = self.user_factors @ self.author_factors.T
            item_authors = items_df["author_encoded"].values
            scores += w_author * user_author_scores[:, item_authors]
        
        # Add publisher-based scores
        if self.publisher_factors is not None and "publisher_encoded" in items_df.columns:
            user_pub_scores = self.user_factors @ self.publisher_factors.T
            item_publishers = items_df["publisher_encoded"].values
            scores += w_publisher * user_pub_scores[:, item_publishers]
        
        # Add subject-based scores
        if self.subject_factors is not None and item_subjects is not None:
            user_subject_scores = self.user_factors @ self.subject_factors.T
            
            if issparse(item_subjects):
                item_subjects_dense = item_subjects.toarray()
            else:
                item_subjects_dense = item_subjects
            
            subject_contribution = user_subject_scores @ item_subjects_dense.T
            weights_per_item = item_subjects_dense.sum(axis=1) + 1e-8
            subject_contribution = subject_contribution / weights_per_item
            
            scores += w_subject * subject_contribution
        
        return scores


# ============================================================
# Helper functions for building auxiliary matrices
# ============================================================

def build_user_author_matrix(
    interactions_df,
    items_df,
    n_users: int,
    n_authors: int
) -> csr_matrix:
    """Build user-author interaction matrix from user-item interactions."""
    merged = interactions_df.merge(
        items_df[["item_idx", "author_encoded"]].drop_duplicates(),
        on="item_idx",
        how="left"
    )
    
    merged = merged.dropna(subset=["author_encoded"])
    merged["author_encoded"] = merged["author_encoded"].astype(int)
    
    user_author = merged.groupby(["user_idx", "author_encoded"]).size().reset_index(name="count")
    
    return csr_matrix(
        (user_author["count"].astype(np.float32),
         (user_author["user_idx"], user_author["author_encoded"])),
        shape=(n_users, n_authors)
    )


def build_user_publisher_matrix(
    interactions_df,
    items_df,
    n_users: int,
    n_publishers: int
) -> csr_matrix:
    """Build user-publisher interaction matrix from user-item interactions."""
    merged = interactions_df.merge(
        items_df[["item_idx", "publisher_encoded"]].drop_duplicates(),
        on="item_idx",
        how="left"
    )
    
    merged = merged.dropna(subset=["publisher_encoded"])
    merged["publisher_encoded"] = merged["publisher_encoded"].astype(int)
    
    user_pub = merged.groupby(["user_idx", "publisher_encoded"]).size().reset_index(name="count")
    
    return csr_matrix(
        (user_pub["count"].astype(np.float32),
         (user_pub["user_idx"], user_pub["publisher_encoded"])),
        shape=(n_users, n_publishers)
    )


def build_user_subject_matrix_tfidf(
    interactions_df,
    item_subjects: np.ndarray,
    n_users: int
) -> csr_matrix:
    """
    Build user-subject interaction matrix using TF-IDF weights.
    """
    n_subjects = item_subjects.shape[1]
    user_subject = np.zeros((n_users, n_subjects), dtype=np.float32)
    
    user_indices = interactions_df["user_idx"].values
    item_indices = interactions_df["item_idx"].values
    
    for user_idx, item_idx in zip(user_indices, item_indices):
        user_subject[user_idx] += item_subjects[item_idx]
    
    return csr_matrix(user_subject)


# ============================================================
# TF-IDF Subject Loading and Reindexing
# ============================================================

def load_and_reindex_tfidf_subjects(
    tfidf_path: str,
    parquet_path: str,
    item_id_to_idx: dict
) -> Tuple[np.ndarray, int]:
    """
    Load TF-IDF subject matrix and reindex to match item_id_to_idx mapping.
    """
    subject_tfidf = load_npz(tfidf_path)
    n_subjects = subject_tfidf.shape[1]
    
    items_parquet = pd.read_parquet(parquet_path)
    parquet_item_to_row = {item_id: row_idx for row_idx, item_id in enumerate(items_parquet["i"].values)}
    
    n_items = len(item_id_to_idx)
    item_subjects = np.zeros((n_items, n_subjects), dtype=np.float32)
    
    matched = 0
    for item_id, your_idx in item_id_to_idx.items():
        if item_id in parquet_item_to_row:
            parquet_row = parquet_item_to_row[item_id]
            item_subjects[your_idx] = subject_tfidf[parquet_row].toarray().flatten()
            matched += 1
    
    print(f"✓ Loaded TF-IDF subjects: {n_items} items x {n_subjects} subjects")
    print(f"  Matched {matched}/{n_items} items ({matched/n_items*100:.1f}%)")
    
    return item_subjects, n_subjects


# ============================================================
# Title Embeddings Loading and Reindexing
# ============================================================

def load_and_reindex_title_embeddings(
    embeddings_path: str,
    parquet_path: str,
    item_id_to_idx: dict
) -> np.ndarray:
    """
    Load title embeddings and reindex to match item_id_to_idx mapping.
    
    Args:
        embeddings_path: Path to title_embeddings.npy
        parquet_path: Path to items_with_topics.parquet
        item_id_to_idx: Your item_id -> item_idx mapping
        
    Returns:
        item_embeddings: (n_items, embedding_dim) reindexed embeddings
    """
    title_embeddings = np.load(embeddings_path)
    embedding_dim = title_embeddings.shape[1]
    
    items_parquet = pd.read_parquet(parquet_path)
    parquet_item_to_row = {item_id: row_idx for row_idx, item_id in enumerate(items_parquet["i"].values)}
    
    n_items = len(item_id_to_idx)
    item_embeddings = np.zeros((n_items, embedding_dim), dtype=np.float32)
    
    matched = 0
    for item_id, your_idx in item_id_to_idx.items():
        if item_id in parquet_item_to_row:
            parquet_row = parquet_item_to_row[item_id]
            item_embeddings[your_idx] = title_embeddings[parquet_row]
            matched += 1
    
    print(f"✓ Loaded title embeddings: {n_items} items x {embedding_dim} dims")
    print(f"  Matched {matched}/{n_items} items ({matched/n_items*100:.1f}%)")
    
    return item_embeddings


def build_user_embedding_matrix(
    interactions_df,
    item_embeddings: np.ndarray,
    n_users: int,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    Build user embedding matrix by aggregating embeddings of items they interacted with.
    
    Args:
        interactions_df: DataFrame with user_idx, item_idx columns
        item_embeddings: (n_items, embedding_dim) matrix
        n_users: Number of users
        aggregation: "mean" or "sum"
        
    Returns:
        user_embeddings: (n_users, embedding_dim) matrix
    """
    embedding_dim = item_embeddings.shape[1]
    user_embeddings = np.zeros((n_users, embedding_dim), dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.float32)
    
    user_indices = interactions_df["user_idx"].values
    item_indices = interactions_df["item_idx"].values
    
    for user_idx, item_idx in zip(user_indices, item_indices):
        user_embeddings[user_idx] += item_embeddings[item_idx]
        user_counts[user_idx] += 1
    
    if aggregation == "mean":
        user_counts[user_counts == 0] = 1
        user_embeddings = user_embeddings / user_counts[:, np.newaxis]
    
    return user_embeddings


def compute_embedding_scores(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute content-based scores using embedding similarity.
    
    Args:
        user_embeddings: (n_users, embedding_dim)
        item_embeddings: (n_items, embedding_dim)
        normalize: If True, use cosine similarity; else dot product
        
    Returns:
        scores: (n_users, n_items) similarity matrix
    """
    if normalize:
        user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8
        item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8
        user_embeddings = user_embeddings / user_norms
        item_embeddings = item_embeddings / item_norms
    
    scores = user_embeddings @ item_embeddings.T
    
    return scores


def predict_with_embeddings(
    model: CollectiveMatrixFactorization,
    items_df,
    item_subjects: np.ndarray,
    item_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    w_item: float = 1.0,
    w_author: float = 0.3,
    w_publisher: float = 0.2,
    w_subject: float = 0.2,
    w_embedding: float = 0.1
) -> np.ndarray:
    """
    Compute final scores combining CMF and embedding similarity.
    
    Args:
        model: Trained CMF model
        items_df: Items DataFrame
        item_subjects: (n_items, n_subjects) TF-IDF matrix
        item_embeddings: (n_items, embedding_dim) title embeddings
        user_embeddings: (n_users, embedding_dim) aggregated user embeddings
        w_*: Weights for each component
        
    Returns:
        scores: (n_users, n_items) final prediction matrix
    """
    # Get CMF scores
    cmf_scores = model.predict_scores(
        items_df, 
        item_subjects=item_subjects,
        w_item=w_item,
        w_author=w_author,
        w_publisher=w_publisher,
        w_subject=w_subject
    )
    
    # Compute embedding-based scores
    embedding_scores = compute_embedding_scores(user_embeddings, item_embeddings, normalize=True)
    
    # Normalize scores before combining (important for different scales)
    cmf_scores_norm = (cmf_scores - cmf_scores.mean()) / (cmf_scores.std() + 1e-8)
    embedding_scores_norm = (embedding_scores - embedding_scores.mean()) / (embedding_scores.std() + 1e-8)
    
    # Combine
    final_scores = cmf_scores_norm + w_embedding * embedding_scores_norm
    
    return final_scores


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate_cmf(
    train_data,
    test_matrix,
    items_df,
    n_users: int,
    n_items: int,
    n_authors: int,
    n_publishers: int,
    config: CMFConfig,
    item_subjects: Optional[np.ndarray] = None,
    item_embeddings: Optional[np.ndarray] = None,
    user_embeddings: Optional[np.ndarray] = None,
    w_embedding: float = 0.1,
    k: int = 10
):
    """Train CMF and evaluate on test set."""
    
    # Build matrices
    user_item_matrix = csr_matrix(
        (np.ones(len(train_data), dtype=np.float32),
         (train_data["user_idx"].values, train_data["item_idx"].values)),
        shape=(n_users, n_items)
    )
    
    user_author_matrix = build_user_author_matrix(
        train_data, items_df, n_users, n_authors
    )
    
    user_publisher_matrix = build_user_publisher_matrix(
        train_data, items_df, n_users, n_publishers
    )

    user_subject_matrix = None
    if item_subjects is not None:
        user_subject_matrix = build_user_subject_matrix_tfidf(
            train_data, item_subjects, n_users
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
    
    # Predict and evaluate
    if item_embeddings is not None and user_embeddings is not None:
        scores = predict_with_embeddings(
            model, items_df, item_subjects,
            item_embeddings, user_embeddings,
            w_embedding=w_embedding
        )
    else:
        scores = model.predict_scores(items_df, item_subjects=item_subjects)
    
    map_score = compute_map_at_k_als(scores, test_matrix, k=k)
    
    return model, map_score


# ============================================================
# Index Validation & Alignment
# ============================================================

def validate_and_align_indices(
    items_clean,
    interactions_clean,
    n_items: int,
    n_authors: int,
    n_publishers: int,
    item_subjects: Optional[np.ndarray] = None,
    n_subjects: Optional[int] = None,
    item_embeddings: Optional[np.ndarray] = None
):
    """
    Ensure all indices are correctly aligned.
    Returns validated items_df with guaranteed index alignment.
    """
    # Sort items by item_idx
    items_clean = items_clean.sort_values("item_idx").reset_index(drop=True)
    
    # Validate item indices
    assert len(items_clean) == n_items, f"Expected {n_items} items, got {len(items_clean)}"
    assert items_clean["item_idx"].tolist() == list(range(n_items)), "Item indices not contiguous"
    
    # Validate author indices
    max_author = items_clean["author_encoded"].max()
    assert max_author < n_authors, f"Author index {max_author} >= n_authors {n_authors}"
    assert items_clean["author_encoded"].min() >= 0, "Negative author index found"
    
    # Validate publisher indices
    max_publisher = items_clean["publisher_encoded"].max()
    assert max_publisher < n_publishers, f"Publisher index {max_publisher} >= n_publishers {n_publishers}"
    assert items_clean["publisher_encoded"].min() >= 0, "Negative publisher index found"
    
    # Validate subject matrix
    if item_subjects is not None and n_subjects is not None:
        assert item_subjects.shape[0] == n_items, f"item_subjects rows {item_subjects.shape[0]} != n_items {n_items}"
        assert item_subjects.shape[1] == n_subjects, f"item_subjects cols {item_subjects.shape[1]} != n_subjects {n_subjects}"
        assert item_subjects.min() >= 0, "Negative values in item_subjects"
    
    # Validate embeddings
    if item_embeddings is not None:
        assert item_embeddings.shape[0] == n_items, f"item_embeddings rows {item_embeddings.shape[0]} != n_items {n_items}"
    
    print("✓ All indices validated and aligned")
    
    return items_clean


# ============================================================
# MAIN with Validation & Submission
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    from scipy.sparse import csr_matrix
    
    from data_preprocessing.clean_interactions import clean_interactions
    from data_preprocessing.clean_items import clean_items
    from data_preprocessing.encode_categorical import encode_authors, encode_publishers
    from split import create_train_test_split
    from eval import compute_map_at_k_als
    
    # ============================================================
    # Configuration
    # ============================================================
    TFIDF_PATH = "data/_/subject_tfidf.npz"
    PARQUET_PATH = "data/_/items_with_topics.parquet"
    EMBEDDINGS_PATH = "data/_/title_embeddings.npy"
    
    USE_TFIDF_SUBJECTS = True
    USE_TITLE_EMBEDDINGS = True
    
    # Embedding weight (tune this!)
    W_EMBEDDING = 0.15
    
    # ============================================================
    # Load and preprocess
    # ============================================================
    print("Loading data...")
    items_df = pd.read_csv("data/items.csv")
    interactions_df = pd.read_csv("data/interactions.csv")
    
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

    # Validate indices
    items_clean = validate_and_align_indices(
        items_clean, interactions_clean, n_items, n_authors, n_publishers,
        item_subjects=item_subjects, n_subjects=n_subjects,
        item_embeddings=item_embeddings
    )
    
    print(f"\nDataset: {n_users} users, {n_items} items, {n_authors} authors, {n_publishers} publishers, {n_subjects} subjects")
    if item_embeddings is not None:
        print(f"Embeddings: {item_embeddings.shape[1]} dims")
    
    # Train/val/test split
    train_data, test_data, val_data = create_train_test_split(
        interactions_clean, test_size=0.2, val_size=0.2
    )
    
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Build user embeddings from training data only
    user_embeddings = None
    if USE_TITLE_EMBEDDINGS:
        print("\nBuilding user embeddings from training data...")
        user_embeddings = build_user_embedding_matrix(
            train_data, item_embeddings, n_users, aggregation="mean"
        )
        print(f"✓ User embeddings: {user_embeddings.shape}")
    
    # Build test and validation matrices
    test_matrix = csr_matrix(
        (np.ones(len(test_data), dtype=np.float32),
         (test_data["user_idx"].values, test_data["item_idx"].values)),
        shape=(n_users, n_items)
    )
    
    val_matrix = csr_matrix(
        (np.ones(len(val_data), dtype=np.float32),
         (val_data["user_idx"].values, val_data["item_idx"].values)),
        shape=(n_users, n_items)
    )
    
    # ============================================================
    # Grid search (use validation set for tuning)
    # ============================================================
    print("\nRunning CMF hyperparameter search on VALIDATION set...")
    
    best_score = 0.0
    best_config = None
    best_model = None
    best_w_embedding = W_EMBEDDING
    
    param_grid = {
        "factors": [256],              # capacity jump (critical)
        "regularization": [0.05],          # keep fixed (already stable)
        "lambda_author": [0.5],
        "lambda_publisher": [0.5],
        "lambda_subject": [0.5],
        "alpha": [380],
        "w_embedding": [0.0, 0.2, 0.4, 0.6, 0.8] if USE_TITLE_EMBEDDINGS else [0.0],
    }
    
    for factors in param_grid["factors"]:
        for reg in param_grid["regularization"]:  # ADD THIS
            for lambda_a in param_grid["lambda_author"]:
                for lambda_p in param_grid["lambda_publisher"]:
                    for lambda_s in param_grid["lambda_subject"]:
                        for alpha in param_grid["alpha"]:
                            for w_emb in param_grid["w_embedding"]:
                                config = CMFConfig(
                                    factors=factors,
                                    iterations=10,
                                    regularization=reg,  # ADD THIS
                                    lambda_author=lambda_a,
                                    lambda_publisher=lambda_p,
                                    lambda_subject=lambda_s,
                                    alpha=alpha
                                )
                                model, val_score = train_and_evaluate_cmf(
                                    train_data, val_matrix, items_clean,
                                    n_users, n_items, n_authors, n_publishers,
                                    config,
                                    item_subjects=item_subjects,
                                    item_embeddings=item_embeddings,
                                    user_embeddings=user_embeddings,
                                    w_embedding=w_emb,
                                    k=10
                                )
                                
                                emb_str = f", w_emb={w_emb}" if USE_TITLE_EMBEDDINGS else ""
                                print(f"[CMF] f={factors}, λ_a={lambda_a}, λ_p={lambda_p}, λ_s={lambda_s}, α={alpha}{emb_str} => VAL MAP@10={val_score:.5f}")
            
                                if val_score > best_score:
                                    best_score = val_score
                                    best_config = config
                                    best_model = model
                                    best_w_embedding = w_emb
    
    print("\n" + "=" * 50)
    print("BEST CMF RESULT (Validation)")
    print("=" * 50)
    print(f"VAL MAP@10: {best_score:.5f}")
    print(f"Config: factors={best_config.factors}, λ_author={best_config.lambda_author}, "
          f"λ_publisher={best_config.lambda_publisher}, λ_subject={best_config.lambda_subject}, α={best_config.alpha}")
    if USE_TITLE_EMBEDDINGS:
        print(f"Embedding weight: {best_w_embedding}")
    
    # ============================================================
    # Evaluate on TEST set
    # ============================================================
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)
    
    if USE_TITLE_EMBEDDINGS:
        final_scores = predict_with_embeddings(
            best_model, items_clean, item_subjects,
            item_embeddings, user_embeddings,
            w_embedding=best_w_embedding
        )
    else:
        final_scores = best_model.predict_scores(items_clean, item_subjects=item_subjects)
    
    test_map = compute_map_at_k_als(final_scores, test_matrix, k=10)
    print(f"TEST MAP@10: {test_map:.5f}")
    
    # ============================================================
    # Generate submission
    # ============================================================
    print("\nGenerating submission...")
    from submit import generate_submission

    submission_df = generate_submission(
        predictions=final_scores,
        user_index=user_id_to_idx,
        item_index=item_id_to_idx,
        k=10
    )
    
    submission_path = "submission_cmf_enriched.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"✓ Saved submission to {submission_path}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Users: {submission_df['user_id'].nunique()}")
    print(f"  Sample:\n{submission_df.head(10)}")