import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_preprocessing.config import INTERACTIONS_FILE, ITEMS_FILE

def load_interactions() -> pd.DataFrame:
    """Load and validate interactions data."""
    df = pd.read_csv(INTERACTIONS_FILE)
    assert list(df.columns) == ["u", "i", "t"], "Unexpected columns in interactions"
    assert df.notna().all().all(), "Unexpected missing values in interactions"
    return df

def load_items() -> pd.DataFrame:
    """Load items data."""
    df = pd.read_csv(ITEMS_FILE)
    expected_cols = ["Title", "Author", "ISBN Valid", "Publisher", "Subjects", "i"]
    assert list(df.columns) == expected_cols, "Unexpected columns in items"
    return df



# def load_data(path='data'):
#     interactions = pd.read_csv(path + '/interactions.csv')
#     items = pd.read_csv(path + '/items.csv')

#     # rename "interractions" columns from ('u','i', 't') to ('user_id', 'item_id', 'timestamp')
#     interactions.rename(columns={'u': 'user_id', 'i': 'item_id', 't': 'timestamp'}, inplace=True)

#     # rename "items" columns from ('i') to ('user_id', 'item_id', 'timestamp')
#     items.rename(columns={'i': 'item_id'}, inplace=True)
#     items_with_interactions = interactions["item_id"].unique()

#     unique_users = interactions["user_id"].unique()
#     unique_items = items_with_interactions
#     user_index = {uid: idx for idx, uid in enumerate(unique_users)}
#     item_index = {iid: idx for idx, iid in enumerate(unique_items)}

#      # ---- Add internal indices to datasets ---- #

#     interactions["user_idx"] = interactions["user_id"].map(user_index)
#     interactions["item_idx"] = interactions["item_id"].map(item_index)

#     items["item_idx"] = items["item_id"].map(item_index)

#     return (
#         interactions,
#         items,
#         user_index,
#         item_index,
#     )

# if __name__ == "__main__":
#     interactions, items, user_index, item_index = load_data()
#     print(f"interactions shape: {interactions.shape}")
#     print(f"items shape: {items.shape}")
#     print(f"User index Length: {len(user_index)}")
#     print(f"User index Max: {len(user_index)}")
#     print(f"Item index Length: {max(item_index)}")
#     print(f"Item index Max: {max(item_index)}")
#     print(f"Unique Items in interactions: {len(interactions['item_id'].unique())}")