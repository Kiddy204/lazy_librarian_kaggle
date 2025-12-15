"""Configuration constants for data preprocessing pipeline."""
from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
INTERACTIONS_FILE = DATA_DIR / "interactions.csv"
ITEMS_FILE = DATA_DIR / "items.csv"

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512  # Reduce from 1536 for efficiency
EMBEDDING_BATCH_SIZE = 100  # API batch size

# Feature settings
MIN_SUBJECT_FREQ = 5  # Minimum occurrences to keep a subject
MAX_SUBJECT_FEATURES = 5000  # Cap on subject vocabulary size
UNKNOWN_AUTHOR = "__UNKNOWN__"
UNKNOWN_PUBLISHER = "__UNKNOWN__"
RARE_AUTHOR_THRESHOLD = 2  # Authors with fewer items -> "RARE"
RARE_PUBLISHER_THRESHOLD = 3

# Output files
OUTPUT_FILES = {
    "interactions_clean": OUTPUT_DIR / "interactions_clean.parquet",
    "items_clean": OUTPUT_DIR / "items_clean.parquet",
    "title_embeddings": OUTPUT_DIR / "title_embeddings.npy",
    "subject_matrix": OUTPUT_DIR / "subject_tfidf.npz",
    "subject_vocab": OUTPUT_DIR / "subject_vocab.json",
    "author_encoder": OUTPUT_DIR / "author_encoder.pkl",
    "publisher_encoder": OUTPUT_DIR / "publisher_encoder.pkl",
    "feature_matrix": OUTPUT_DIR / "item_features.npz",
    # Train/val/test splits
    "train": OUTPUT_DIR / "train.parquet",
    "val": OUTPUT_DIR / "val.parquet",
    "test": OUTPUT_DIR / "test.parquet",
}

# Split settings
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2