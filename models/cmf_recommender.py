"""
CMF Model-based Recommendation System
Integrates the trained CMF model with the frontend for personalized recommendations.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "pipe"))
sys.path.append(str(Path(__file__).parent.parent / "data_preprocessing"))

try:
    from pipe.submit import generate_submission
    from data_preprocessing.config import INTERACTIONS_FILE, ITEMS_FILE
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")


class CMFRecommender:
    """Simple wrapper around the trained CMF model for frontend integration."""
    
    def __init__(self, model_path: str = "models/cmf_full_model_tfidf.pkl"):
        self.model_path = model_path
        self.model_data = None
        self.is_loaded = False
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained CMF model."""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                print(f"Model file not found: {self.model_path}")
                return False
            
            print(f"Loading CMF model from {self.model_path}...")
            with open(model_file, "rb") as f:
                self.model_data = pickle.load(f)
            
            self.is_loaded = True
            print("✅ CMF model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CMF model: {e}")
            return False
    
    def is_available(self):
        """Check if model is available."""
        return self.is_loaded
    
    def get_recommendations_for_user(self, user_id: int, n_recommendations: int = 10, exclude_items: list = None):
        """
        Get top N recommendations for a specific user using the CMF model.
        
        Args:
            user_id: User ID to get recommendations for
            n_recommendations: Number of recommendations to return
            exclude_items: List of item IDs to exclude (already borrowed)
            
        Returns:
            list: List of recommended item IDs
        """
        if not self.is_loaded:
            print("⚠️ CMF model not available, falling back to popular items")
            return self._get_popular_items(n_recommendations, exclude_items)
        
        try:
            model = self.model_data["model"]
            user_id_to_idx = self.model_data["user_id_to_idx"]
            item_id_to_idx = self.model_data["item_id_to_idx"]
            item_idx_to_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
            
            # Check if user exists in training data
            if user_id not in user_id_to_idx:
                print(f"⚠️ User {user_id} not in training data, using popular items")
                return self._get_popular_items(n_recommendations, exclude_items)
            
            # Get user index
            user_idx = user_id_to_idx[user_id]
            
            # Get user's predicted scores for all items
            # The model stores user and item factors
            user_factors = model.user_factors[user_idx]
            item_factors = model.item_factors
            
            # Calculate scores: user_factors @ item_factors.T
            scores = user_factors @ item_factors.T
            
            # Get top N items (excluding already borrowed)
            if exclude_items is None:
                exclude_items = []
            
            # Convert exclude_items to indices
            exclude_indices = [item_id_to_idx[item_id] for item_id in exclude_items if item_id in item_id_to_idx]
            
            # Set excluded items to very low score
            scores_copy = scores.copy()
            scores_copy[exclude_indices] = -np.inf
            
            # Get top N indices
            top_indices = np.argsort(scores_copy)[::-1][:n_recommendations]
            
            # Convert back to item IDs
            recommended_items = [item_idx_to_id[idx] for idx in top_indices if idx in item_idx_to_id]
            
            return recommended_items
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            return self._get_popular_items(n_recommendations, exclude_items)
    
    def _get_popular_items(self, n_recommendations: int = 10, exclude_items: list = None):
        """Fallback: get popular items based on borrow frequency."""
        try:
            interactions_df = pd.read_csv(INTERACTIONS_FILE)
            
            # Calculate borrow counts
            borrow_counts = interactions_df.groupby('i').size().reset_index(name='count')
            popular_items = borrow_counts.sort_values('count', ascending=False)
            
            # Exclude items if provided
            if exclude_items:
                popular_items = popular_items[~popular_items['i'].isin(exclude_items)]
            
            # Get top N popular items
            top_items = popular_items.head(n_recommendations)['i'].tolist()
            
            return top_items
            
        except Exception as e:
            print(f"❌ Error getting popular items: {e}")
            return []


# Global instance for use in frontend
cmf_recommender = CMFRecommender()