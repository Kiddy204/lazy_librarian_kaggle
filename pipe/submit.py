import pandas as pd
import numpy as np

def generate_submission(predictions, user_index, item_index, k=10):
    """
    Generate submission dataframe with top-k recommendations for each user.
    Converts internal indices back to actual user/item IDs for Kaggle submission.
    
    Parameters:
        predictions (numpy array): Predicted scores matrix (n_users x n_items).
        user_index (dict): Mapping from actual user_id to internal user_idx.
        item_index (dict): Mapping from actual item_id to internal item_idx.
        k (int): Number of top recommendations per user.
    
    Returns:
        pandas DataFrame: Submission dataframe with actual user_id and item_id recommendations.
    """
    # Create reverse mappings: internal index -> actual ID
    idx_to_user = {idx: uid for uid, idx in user_index.items()}
    idx_to_item = {idx: iid for iid, idx in item_index.items()}
    
    submissions = []
    n_users = predictions.shape[0]
    
    for user_idx in range(n_users):
        # Get top-k predicted items (sorted by predicted score, descending)
        top_k_item_indices = np.argsort(predictions[user_idx])[::-1][:k]
        
        # Convert item indices back to actual item IDs
        top_k_item_ids = [idx_to_item[idx] for idx in top_k_item_indices]
        
        # Convert to space-separated string of actual item IDs
        recommendation_str = ' '.join(map(str, top_k_item_ids))
        
        # Use actual user_id instead of internal index
        actual_user_id = idx_to_user[user_idx]
        submissions.append({'user_id': actual_user_id, 'recommendation': recommendation_str})
    
    return pd.DataFrame(submissions)