"""Categorical feature encoding."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Tuple

from data_preprocessing.config import RARE_AUTHOR_THRESHOLD, RARE_PUBLISHER_THRESHOLD, UNKNOWN_AUTHOR, UNKNOWN_PUBLISHER

from data_preprocessing.clean_items import clean_items
from data_preprocessing.load_data import load_interactions, load_items

class FrequencyAwareLabelEncoder:
    """
    Label encoder that groups rare categories.
    Categories below threshold are mapped to a single 'RARE' category.
    """
    def __init__(self, rare_threshold: int, rare_label: str = "__RARE__"):
        self.rare_threshold = rare_threshold
        self.rare_label = rare_label
        self.encoder = LabelEncoder()
        self.frequent_categories = set()
    
    def fit(self, values: pd.Series) -> "FrequencyAwareLabelEncoder":
        # Count frequencies
        freq = values.value_counts()
        self.frequent_categories = set(freq[freq >= self.rare_threshold].index)
        
        # Map rare to rare_label
        mapped = values.apply(
            lambda x: x if x in self.frequent_categories else self.rare_label
        )
        self.encoder.fit(mapped)
        return self
    
    def transform(self, values: pd.Series) -> np.ndarray:
        mapped = values.apply(
            lambda x: x if x in self.frequent_categories else self.rare_label
        )
        return self.encoder.transform(mapped)
    
    def fit_transform(self, values: pd.Series) -> np.ndarray:
        self.fit(values)
        return self.transform(values)
    
    @property
    def n_classes(self) -> int:
        return len(self.encoder.classes_)
    
    @property
    def classes_(self) -> np.ndarray:
        return self.encoder.classes_

def encode_authors(df: pd.DataFrame) -> Tuple[np.ndarray, FrequencyAwareLabelEncoder]:
    """
    Encode author column with rare author grouping.
    
    Returns:
        - encoded array of shape (n_items,)
        - fitted encoder for saving
    """
    encoder = FrequencyAwareLabelEncoder(
        rare_threshold=RARE_AUTHOR_THRESHOLD,
        rare_label="__RARE_AUTHOR__"
    )
    encoded = encoder.fit_transform(df["author_clean"])
    print(f"Authors: {df['author_clean'].nunique()} unique -> {encoder.n_classes} categories")
    return encoded, encoder

def encode_publishers(df: pd.DataFrame) -> Tuple[np.ndarray, FrequencyAwareLabelEncoder]:
    """
    Encode publisher column with rare publisher grouping.
    """
    encoder = FrequencyAwareLabelEncoder(
        rare_threshold=RARE_PUBLISHER_THRESHOLD,
        rare_label="__RARE_PUBLISHER__"
    )
    encoded = encoder.fit_transform(df["publisher_clean"])
    print(f"Publishers: {df['publisher_clean'].nunique()} unique -> {encoder.n_classes} categories")
    return encoded, encoder

def save_encoder(encoder: FrequencyAwareLabelEncoder, path: str) -> None:
    """Save encoder for inference."""
    with open(path, "wb") as f:
        pickle.dump(encoder, f)

def load_encoder(path: str) -> FrequencyAwareLabelEncoder:
    """Load saved encoder."""
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    items = load_items()
    interactions = load_interactions()
    items_clean = clean_items(items, interactions)
    encoded_authors, author_encoder = encode_authors(items_clean)
    items_clean["author_encoded"] = encoded_authors
    encoded_publishers, publisher_encoder = encode_publishers(items_clean)
    items_clean["publisher_encoded"] = encoded_publishers
    # print(items_clean.head())
    # print(author_encoder.classes_)
    # print(publisher_encoder.classes_)
    print(f"author_encoded shape: {encoded_authors.shape}")
    print(f"publisher_encoded shape: {encoded_publishers.shape}")
    print(f"items_clean shape: {items_clean.shape}")
    print(f"author_encoder n_classes: {author_encoder.classes_.shape}")
    print(f"publisher_encoder n_classes: {publisher_encoder.classes_.shape}")
    print(f"author_encoder heaad: {encoded_authors.shape}")
    print(f"publisher_encoder head: {encoded_publishers.shape}")