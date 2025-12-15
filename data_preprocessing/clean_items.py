"""Clean and prepare item metadata."""
import pandas as pd


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import re
from typing import List, Optional
from data_preprocessing.config import UNKNOWN_AUTHOR, UNKNOWN_PUBLISHER
from data_preprocessing.load_data import load_interactions, load_items

def normalize_text(text: Optional[str]) -> str:
    """Normalize text: strip, collapse whitespace."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text

def parse_subjects(subjects_str: Optional[str]) -> List[str]:
    """Parse semicolon-separated subjects into clean list."""
    if pd.isna(subjects_str) or not subjects_str:
        return []
    subjects = [s.strip().lower() for s in str(subjects_str).split(";")]
    subjects = [s for s in subjects if s]  # Remove empty
    return subjects

def clean_author(author: Optional[str]) -> str:
    """
    Clean author field.
    Format: "Lastname, Firstname, birth-death" -> "lastname firstname"
    """
    if pd.isna(author) or not author:
        return UNKNOWN_AUTHOR
    
    author = str(author).strip()
    # Remove birth-death years (e.g., "1947-", "1923-2010")
    author = re.sub(r",?\s*\d{4}-?\d{0,4}\.?$", "", author)
    # Normalize: lowercase, collapse whitespace
    author = re.sub(r"\s+", " ", author.lower().strip())
    
    return author if author else UNKNOWN_AUTHOR

def clean_publisher(publisher: Optional[str]) -> str:
    """Clean publisher field."""
    if pd.isna(publisher) or not publisher:
        return UNKNOWN_PUBLISHER
    return normalize_text(publisher)

def clean_title(title: str) -> str:
    """Clean title for embedding generation."""
    title = normalize_text(title)
    # Remove trailing " /" which appears in many titles
    title = re.sub(r"\s*/\s*$", "", title)
    return title

def clean_items(df: pd.DataFrame, interactions_df: List[int], remove_subjects=False) -> pd.DataFrame:
    """
    Clean all item metadata fields.
    
    Returns DataFrame with cleaned columns:
    - i: item ID (unchanged)
    - title_clean: cleaned title for embeddings
    - title_original: original title (preserved)
    - author_clean: normalized author
    - publisher_clean: normalized publisher  
    - subjects_list: list of subject tags
    - has_subjects: boolean flag
    """
    df = df.copy()
    interacted_items = set(interactions_df["i"].unique())

    # Remove cold-start items
    n_before = len(df)
    df = df[df["i"].isin(interacted_items)]
    n_removed = n_before - len(df)
    print(f"Removed {n_removed} cold-start items ({n_removed/n_before*100:.1f}%)")
    # Clean each field
    df["title_original"] = df["Title"]
    df["title_clean"] = df["Title"].apply(clean_title)
    df["author_clean"] = df["Author"].apply(clean_author)
    df["publisher_clean"] = df["Publisher"].apply(clean_publisher)
    df["subjects_list"] = df["Subjects"].apply(parse_subjects)
    df["has_subjects"] = df["subjects_list"].apply(lambda x: len(x) > 0)
    
    # Select and reorder columns
    df_clean = df[[
        "i", "title_clean", "title_original", 
        "author_clean", "publisher_clean", 
        "subjects_list", "has_subjects"
    ]].copy()
    if remove_subjects:
        all_subjects = df_clean["subjects_list"].explode().unique()
        for subject in all_subjects:
            if df_clean["subjects_list"].apply(lambda x: subject in x).sum() == 1:
                df_clean["subjects_list"] = df_clean["subjects_list"].apply(lambda x: [s for s in x if s != subject])
    # Validate
    assert df_clean["i"].is_unique, "Item IDs must be unique"
    assert df_clean["title_clean"].notna().all(), "All items must have titles"
    
    return df_clean
    
def get_item_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for validation."""
    return {
        "n_items": len(df),
        "n_items_with_subjects": str(round((df["has_subjects"].sum()/len(df)*100), 2))+'%',
        "n_items_with_authors": str(round((len(df[df["author_clean"] != UNKNOWN_AUTHOR])/len(df)*100), 2))+'%',
        "n_items_with_publishers": str(round((len(df[df["publisher_clean"] != UNKNOWN_PUBLISHER])/len(df)*100), 2))+'%',
    }

if __name__ == "__main__":
    items = load_items()
    interactions = load_interactions()
    items_clean = clean_items(items, interactions)
    subects = items_clean.iloc[0]["subjects_list"][0]
    print(f"subjects: {subects}")
    item_stats = get_item_stats(items_clean)
    print(item_stats)
    print(f"columns: {items_clean.columns}")
    print(f"items_clean shape: {items_clean.shape}")