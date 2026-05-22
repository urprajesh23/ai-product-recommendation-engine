"""
Hybrid Recommender System
Combines Collaborative Filtering and Content-Based models
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix

from .collaborative import CollaborativeFilteringModel
from .content_based import ContentBasedModel


class HybridRecommender:
    """
    Hybrid Recommendation System
    
    Combines:
    1. Collaborative Filtering (ALS) - learns from user-item interactions
    2. Content-Based (LightFM) - uses item/user features
    3. Popularity-based fallback - for cold-start scenarios
    """
    
    def __init__(
        self,
        cf_weight=0.6,
        cb_weight=0.4,
        enable_cold_start=True,
        cf_params: Optional[Dict] = None,
        cb_params: Optional[Dict] = None
    ):
        """
        Initialize Hybrid Recommender
        
        Args:
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based (0-1)
            enable_cold_start: Whether to use popularity fallback for new users
            cf_params: Parameters for collaborative filtering model
            cb_params: Parameters for content-based model
        """
        # Validate weights
        if not np.isclose(cf_weight + cb_weight, 1.0):
            raise ValueError("cf_weight + cb_weight must equal 1.0")
        
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.enable_cold_start = enable_cold_start
        
        # Initialize models
        cf_params = cf_params or {}
        cb_params = cb_params or {}
        
        self.cf_model = CollaborativeFilteringModel(**cf_params)
        self.cb_model = ContentBasedModel(**cb_params)
        
        # Cold-start components
        self.popularity_scores = {}
        self.is_fitted = False
        
        # Metadata
        self.n_users = 0
        self.n_items = 0
        
    def fit(
        self,
        train_matrix: csr_matrix,
        interactions_df,
        item_features_dict: Optional[Dict] = None,
        user_features_dict: Optional[Dict] = None,
        cf_epochs=50,
        cb_epochs=30,
        verbose=True
    ):
        """
        Train both collaborative and content-based models
        
        Args:
            train_matrix: User-item interaction matrix (sparse)
            interactions_df: DataFrame with user_idx, item_idx, implicit
            item_features_dict: Item features for content-based model
            user_features_dict: User features for content-based model
            cf_epochs: Training iterations for collaborative filtering
            cb_epochs: Training epochs for content-based
            verbose: Whether to print progress
        """
        print("=" * 70)
        print("🚀 Training Hybrid Recommender System")
        print("=" * 70)
        print(f"   CF Weight: {self.cf_weight:.2f}")
        print(f"   CB Weight: {self.cb_weight:.2f}")
        print(f"   Cold-start enabled: {self.enable_cold_start}")
        print()
        
        self.n_users = train_matrix.shape[0]
        self.n_items = train_matrix.shape[1]
        
        # Train Collaborative Filtering model
        print("📊 Step 1/3: Training Collaborative Filtering Model")
        print("-" * 70)
        self.cf_model.fit(train_matrix, show_progress=verbose)
        print()
        
        # Train Content-Based model
        print("📚 Step 2/3: Training Content-Based Model")
        print("-" * 70)
        
        # Prepare dataset for LightFM
        interactions, weights = self.cb_model.prepare_dataset(
            interactions_df,
            item_features_dict=item_features_dict,
            user_features_dict=user_features_dict
        )
        
        # Train
        self.cb_model.fit(
            interactions,
            epochs=cb_epochs,
            verbose=verbose
        )
        print()
        
        # Calculate popularity scores for cold-start
        if self.enable_cold_start:
            print("⭐ Step 3/3: Calculating Item Popularity Scores")
            print("-" * 70)
            self._calculate_popularity(interactions_df)
        
        self.is_fitted = True
        
        print()
        print("=" * 70)
        print("✅ Hybrid Recommender Training Complete!")
        print("=" * 70)
        print(f"   Users: {self.n_users:,}")
        print(f"   Items: {self.n_items:,}")
        print()
    
    def _calculate_popularity(self, interactions_df):
        """
        Calculate item popularity scores for cold-start
        
        Args:
            interactions_df: DataFrame with item_idx and implicit columns
        """
        # Count positive interactions per item
        item_counts = interactions_df[interactions_df['implicit'] == 1].groupby('item_idx').size()
        
        # Normalize to create probability distribution
        total_interactions = item_counts.sum()
        
        self.popularity_scores = {}
        for item_idx, count in item_counts.items():
            self.popularity_scores[item_idx] = count / total_interactions
        
        print(f"   Calculated popularity for {len(self.popularity_scores):,} items")
    
    def recommend(
        self,
        user_idx: int,
        train_matrix: csr_matrix,
        N=10,
        is_new_user=False,
        filter_already_liked=True,
        return_scores=False
    ) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations for a user
        
        Args:
            user_idx: User index
            train_matrix: Training interaction matrix
            N: Number of recommendations
            is_new_user: Whether user is new (cold-start)
            filter_already_liked: Whether to filter items user has interacted with
            return_scores: Whether to return detailed scores
            
        Returns:
            List of (item_idx, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Handle cold-start users
        if is_new_user or (user_idx >= self.n_users):
            return self._recommend_popular(N)
        
        # Get items user has already interacted with
        user_items = set(train_matrix[user_idx].indices) if filter_already_liked else set()
        
        # Get collaborative filtering recommendations
        cf_recs = self.cf_model.recommend(
            user_idx=user_idx,
            user_item_matrix=train_matrix,
            N=N * 3,  # Get more candidates
            filter_already_liked=filter_already_liked
        )
        
        # Get content-based recommendations
        cb_recs = self.cb_model.recommend(
            user_idx=user_idx,
            n_items=self.n_items,
            filter_items=list(user_items),
            N=N * 3
        )
        
        # Combine recommendations
        hybrid_scores = self._combine_scores(cf_recs, cb_recs)
        
        # Sort and return top N
        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        if return_scores:
            return sorted_recs[:N]
        else:
            return [(item_idx, score) for item_idx, score in sorted_recs[:N]]
    
    def _combine_scores(
        self,
        cf_recs: List[Tuple[int, float]],
        cb_recs: List[Tuple[int, float]]
    ) -> Dict[int, float]:
        """
        Combine scores from collaborative and content-based models
        
        Args:
            cf_recs: Collaborative filtering recommendations
            cb_recs: Content-based recommendations
            
        Returns:
            Dictionary mapping item_idx to combined score
        """
        # Create score dictionaries
        cf_scores = {item: score for item, score in cf_recs}
        cb_scores = {item: score for item, score in cb_recs}
        
        # Get all candidate items
        all_items = set(cf_scores.keys()) | set(cb_scores.keys())
        
        # Normalize scores to [0, 1] range
        cf_max = max(cf_scores.values()) if cf_scores else 1.0
        cb_max = max(cb_scores.values()) if cb_scores else 1.0
        
        # Combine scores
        hybrid_scores = {}
        for item in all_items:
            cf_score = cf_scores.get(item, 0) / cf_max if cf_max > 0 else 0
            cb_score = cb_scores.get(item, 0) / cb_max if cb_max > 0 else 0
            
            # Weighted combination
            hybrid_scores[item] = (
                self.cf_weight * cf_score +
                self.cb_weight * cb_score
            )
        
        return hybrid_scores
    
    def _recommend_popular(self, N=10) -> List[Tuple[int, float]]:
        """
        Recommend popular items (cold-start fallback)
        
        Args:
            N: Number of recommendations
            
        Returns:
            List of (item_idx, popularity_score) tuples
        """
        if not self.popularity_scores:
            # Return empty if no popularity data
            return []
        
        # Sort by popularity
        sorted_items = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items[:N]
    
    def recommend_batch(
        self,
        user_indices: List[int],
        train_matrix: csr_matrix,
        N=10,
        filter_already_liked=True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get recommendations for multiple users
        
        Args:
            user_indices: List of user indices
            train_matrix: Training interaction matrix
            N: Number of recommendations per user
            filter_already_liked: Whether to filter already liked items
            
        Returns:
            Dictionary mapping user_idx to recommendations
        """
        recommendations = {}
        
        for user_idx in user_indices:
            is_new = user_idx >= self.n_users
            
            recs = self.recommend(
                user_idx=user_idx,
                train_matrix=train_matrix,
                N=N,
                is_new_user=is_new,
                filter_already_liked=filter_already_liked
            )
            
            recommendations[user_idx] = recs
        
        return recommendations
    
    def explain_recommendation(
        self,
        user_idx: int,
        item_idx: int,
        train_matrix: csr_matrix
    ) -> Dict:
        """
        Explain why an item was recommended
        
        Args:
            user_idx: User index
            item_idx: Item index
            train_matrix: Training interaction matrix
            
        Returns:
            Dictionary with explanation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating explanations")
        
        # Get scores from both models
        cf_score = self.cf_model.predict_score(user_idx, item_idx)
        
        # Get CB score (need to predict for specific user-item pair)
        cb_score = self.cb_model.predict(
            user_indices=np.array([user_idx]),
            item_indices=np.array([item_idx])
        )[0]
        
        # Normalize
        hybrid_score = self.cf_weight * cf_score + self.cb_weight * cb_score
        
        # Get CF explanation
        cf_explanation = self.cf_model.explain_recommendation(
            user_idx, item_idx, train_matrix, N=3
        )
        
        explanation = {
            'item_idx': item_idx,
            'hybrid_score': float(hybrid_score),
            'cf_score': float(cf_score),
            'cb_score': float(cb_score),
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'contributing_items': cf_explanation['contributing_items'],
            'explanation': (
                f"Score: {hybrid_score:.3f} "
                f"({self.cf_weight:.0%} from collaborative filtering, "
                f"{self.cb_weight:.0%} from content features)"
            )
        }
        
        return explanation
    
    def save(self, filepath: str):
        """
        Save hybrid model to disk
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        cf_path = filepath.parent / f"{filepath.stem}_cf.pkl"
        cb_path = filepath.parent / f"{filepath.stem}_cb.pkl"
        
        self.cf_model.save(cf_path)
        self.cb_model.save(cb_path)
        
        # Save hybrid metadata
        hybrid_data = {
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'enable_cold_start': self.enable_cold_start,
            'popularity_scores': self.popularity_scores,
            'is_fitted': self.is_fitted,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(hybrid_data, f)
        
        print(f"💾 Hybrid model saved to {filepath}")
        print(f"   - CF model: {cf_path}")
        print(f"   - CB model: {cb_path}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load hybrid model from disk
        
        Args:
            filepath: Path to saved model
            
        Returns:
            HybridRecommender instance
        """
        filepath = Path(filepath)
        
        # Load hybrid metadata
        with open(filepath, 'rb') as f:
            hybrid_data = pickle.load(f)
        
        # Load individual models
        cf_path = filepath.parent / f"{filepath.stem}_cf.pkl"
        cb_path = filepath.parent / f"{filepath.stem}_cb.pkl"
        
        # Create instance
        instance = cls(
            cf_weight=hybrid_data['cf_weight'],
            cb_weight=hybrid_data['cb_weight'],
            enable_cold_start=hybrid_data['enable_cold_start']
        )
        
        # Load models
        instance.cf_model = CollaborativeFilteringModel.load(cf_path)
        instance.cb_model = ContentBasedModel.load(cb_path)
        
        # Restore metadata
        instance.popularity_scores = hybrid_data['popularity_scores']
        instance.is_fitted = hybrid_data['is_fitted']
        instance.n_users = hybrid_data['n_users']
        instance.n_items = hybrid_data['n_items']
        
        print(f"📂 Hybrid model loaded from {filepath}")
        
        return instance
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        info = {
            'model_type': 'Hybrid (CF + Content-Based)',
            'cf_weight': self.cf_weight,
            'cb_weight': self.cb_weight,
            'cold_start_enabled': self.enable_cold_start,
            'is_fitted': self.is_fitted,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_popular_items': len(self.popularity_scores)
        }
        
        if self.is_fitted:
            info['cf_model'] = self.cf_model.get_model_info()
            info['cb_model'] = self.cb_model.get_model_info()
        
        return info


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Hybrid Recommender System - Example Usage")
    print("=" * 70)
    
    print("\n📝 Complete workflow example:")
    print("""
from src.models.hybrid import HybridRecommender
from src.data.preprocess import DataPreprocessor

# Load preprocessed data
preprocessor = DataPreprocessor()
train_matrix = load_matrix('data/processed/train_matrix.pkl')
train_df = pd.read_csv('data/processed/train.csv')

# Create hybrid model
hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.4)

# Train
hybrid.fit(
    train_matrix=train_matrix,
    interactions_df=train_df,
    item_features_dict=item_features,
    cf_epochs=50,
    cb_epochs=30
)

# Get recommendations
recommendations = hybrid.recommend(
    user_idx=100,
    train_matrix=train_matrix,
    N=10
)

# Handle new user (cold-start)
new_user_recs = hybrid.recommend(
    user_idx=999999,
    train_matrix=train_matrix,
    N=10,
    is_new_user=True
)

# Save model
hybrid.save('models/hybrid_model.pkl')

# Load model
loaded_model = HybridRecommender.load('models/hybrid_model.pkl')
    """)
    
    print("\n✅ Hybrid recommender ready!")