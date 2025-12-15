"""Clean and prepare interaction data."""
import pandas as pd
import numpy as np

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_preprocessing.load_data import load_interactions

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["t"], unit="s")
    df = df.sort_values("t").reset_index(drop=True)
    
    # Add interaction index within each (u, i) pair
    df["interaction_rank"] = df.groupby(["u", "i"]).cumcount() + 1
    
    return df

def get_interaction_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for validation."""

        # Count interactions per (u, i) pair
    pair_counts = df.groupby(["u", "i"]).size()
    
    # Distribution of repeat counts: how many pairs have 1, 2, 3... interactions
    repeat_distribution = pair_counts.value_counts().sort_index().to_dict()
    return {
        "n_interactions": len(df),
        "n_users": df["u"].nunique(),
        "n_items": df["i"].nunique(),
        "date_range": (df["datetime"].min(), df["datetime"].max()),
        "interaction_rank_distribution": repeat_distribution
    }

if __name__ == "__main__":
    interactions = load_interactions()
    interactions_clean = clean_interactions(interactions)
    print(interactions_clean.head())
    print(get_interaction_stats(interactions_clean))