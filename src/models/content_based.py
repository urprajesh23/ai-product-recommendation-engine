"""
Content-Based Filtering Model using LightFM
Combines collaborative and content features
"""

from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class ContentBasedModel:
    """
    Content-Based Recommender using LightFM
    
    LightFM is a hybrid model that can use both collaborative signals
    and content features (item metadata, user demographics)
    """
    
    def __init__(
        self,
        no_components=128,
        loss='warp',
        learning_rate=0.05,
        item_alpha=0.0,
        user_alpha=0.0,
        random_state=42
    ):
        """
        Initialize LightFM content-based model
        
        Args:
            no_components: Number of latent dimensions
            loss: Loss function ('warp', 'bpr', 'logistic', 'warp-kos')
            learning_rate: Learning rate for SGD
            item_alpha: L2 penalty for item features
            user_alpha: L2 penalty for user features
            random_state: Random seed
        """
        self.no_components = no_components
        self.loss = loss
        self.learning_rate = learning_rate
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.random_state = random_state
        
        # Initialize model
        self.model = LightFM(
            no_components=no_components,
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            random_state=random_state
        )
        
        self.dataset = None
        self.is_fitted = False
        self.item_features_matrix = None
        self.user_features_matrix = None
        
    def prepare_dataset(
        self,
        interactions_df,
        item_features_dict: Optional[Dict] = None,
        user_features_dict: Optional[Dict] = None
    ):
        """
        Prepare LightFM dataset with features
        
        Args:
            interactions_df: DataFrame with user_idx, item_idx, implicit columns
            item_features_dict: Dict mapping item_idx to list of features
            user_features_dict: Dict mapping user_idx to list of features
            
        Returns:
            Tuple of (interactions_matrix, weights)
        """
        print("🔧 Preparing LightFM dataset...")
        
        # Create dataset object
        self.dataset = Dataset()
        
        # Get unique users and items
        users = interactions_df['user_idx'].unique()
        items = interactions_df['item_idx'].unique()
        
        print(f"   Users: {len(users):,}")
        print(f"   Items: {len(items):,}")
        
        # Prepare feature lists
        item_features = None
        if item_features_dict:
            item_features = []
            for item_idx, features in item_features_dict.items():
                for feature in features:
                    item_features.append(f"{feature}")
            item_features = list(set(item_features))
            print(f"   Item features: {len(item_features):,}")
        
        user_features = None
        if user_features_dict:
            user_features = []
            for user_idx, features in user_features_dict.items():
                for feature in features:
                    user_features.append(f"{feature}")
            user_features = list(set(user_features))
            print(f"   User features: {len(user_features):,}")
        
        # Fit the dataset
        self.dataset.fit(
            users=users,
            items=items,
            item_features=item_features,
            user_features=user_features
        )
        
        # Build interactions matrix
        interactions_data = [
            (row['user_idx'], row['item_idx'], row['implicit'])
            for _, row in interactions_df.iterrows()
        ]
        
        interactions_matrix, weights = self.dataset.build_interactions(interactions_data)
        
        # Build feature matrices if features provided
        if item_features_dict:
            item_features_data = [
                (item_idx, features)
                for item_idx, features in item_features_dict.items()
            ]
            self.item_features_matrix = self.dataset.build_item_features(item_features_data)
            print(f"   Item features matrix shape: {self.item_features_matrix.shape}")
        
        if user_features_dict:
            user_features_data = [
                (user_idx, features)
                for user_idx, features in user_features_dict.items()
            ]
            self.user_features_matrix = self.dataset.build_user_features(user_features_data)
            print(f"   User features matrix shape: {self.user_features_matrix.shape}")
        
        print(f"✅ Dataset prepared. Interactions matrix shape: {interactions_matrix.shape}")
        
        return interactions_matrix, weights
    
    def fit(
        self,
        interactions,
        item_features=None,
        user_features=None,
        epochs=30,
        num_threads=4,
        verbose=True
    ):
        """
        Train the LightFM model
        
        Args:
            interactions: Interactions matrix from prepare_dataset
            item_features: Item features matrix (optional)
            user_features: User features matrix (optional)
            epochs: Number of training epochs
            num_threads: Number of parallel threads
            verbose: Whether to print progress
        """
        print(f"🚀 Training Content-Based Model (LightFM)...")
        print(f"   Components: {self.no_components}")
        print(f"   Loss: {self.loss}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {self.learning_rate}")
        
        # Use stored feature matrices if not provided
        if item_features is None:
            item_features = self.item_features_matrix
        if user_features is None:
            user_features = self.user_features_matrix
        
        # Train the model
        self.model.fit(
            interactions=interactions,
            item_features=item_features,
            user_features=user_features,
            epochs=epochs,
            num_threads=num_threads,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        print(f"✅ Training complete!")
    
    def predict(
        self,
        user_indices,
        item_indices,
        item_features=None,
        user_features=None,
        num_threads=4
    ):
        """
        Predict scores for user-item pairs
        
        Args:
            user_indices: Array of user indices
            item_indices: Array of item indices
            item_features: Item features matrix
            user_features: User features matrix
            num_threads: Number of threads
            
        Returns:
            Array of prediction scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if item_features is None:
            item_features = self.item_features_matrix
        if user_features is None:
            user_features = self.user_features_matrix
        
        scores = self.model.predict(
            user_ids=user_indices,
            item_ids=item_indices,
            item_features=item_features,
            user_features=user_features,
            num_threads=num_threads
        )
        
        return scores
    
    def recommend(
        self,
        user_idx: int,
        n_items: int,
        item_features=None,
        user_features=None,
        filter_items: Optional[List[int]] = None,
        N=10
    ) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user
        
        Args:
            user_idx: User index
            n_items: Total number of items in dataset
            item_features: Item features matrix
            user_features: User features matrix
            filter_items: List of item indices to exclude
            N: Number of recommendations
            
        Returns:
            List of (item_idx, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if item_features is None:
            item_features = self.item_features_matrix
        if user_features is None:
            user_features = self.user_features_matrix
        
        # Create arrays for all items
        user_indices = np.array([user_idx] * n_items)
        item_indices = np.arange(n_items)
        
        # Get scores for all items
        scores = self.predict(
            user_indices=user_indices,
            item_indices=item_indices,
            item_features=item_features,
            user_features=user_features
        )
        
        # Filter out specified items
        if filter_items:
            filter_mask = np.ones(len(scores), dtype=bool)
            filter_mask[filter_items] = False
            item_indices = item_indices[filter_mask]
            scores = scores[filter_mask]
        
        # Get top N
        top_indices = np.argsort(-scores)[:N]
        top_items = item_indices[top_indices]
        top_scores = scores[top_indices]
        
        recommendations = list(zip(top_items.tolist(), top_scores.tolist()))
        
        return recommendations
    
    def get_item_representations(self, item_features=None):
        """
        Get item latent representations
        
        Args:
            item_features: Item features matrix
            
        Returns:
            numpy array of item embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing representations")
        
        if item_features is None:
            item_features = self.item_features_matrix
        
        # Get item biases and embeddings
        return self.model.get_item_representations(features=item_features)
    
    def get_user_representations(self, user_features=None):
        """
        Get user latent representations
        
        Args:
            user_features: User features matrix
            
        Returns:
            numpy array of user embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing representations")
        
        if user_features is None:
            user_features = self.user_features_matrix
        
        # Get user biases and embeddings
        return self.model.get_user_representations(features=user_features)
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'no_components': self.no_components,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'item_alpha': self.item_alpha,
            'user_alpha': self.user_alpha,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'dataset': self.dataset,
            'item_features_matrix': self.item_features_matrix,
            'user_features_matrix': self.user_features_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load model from disk
        
        Args:
            filepath: Path to saved model
            
        Returns:
            ContentBasedModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            no_components=model_data['no_components'],
            loss=model_data['loss'],
            learning_rate=model_data['learning_rate'],
            item_alpha=model_data['item_alpha'],
            user_alpha=model_data['user_alpha'],
            random_state=model_data['random_state']
        )
        
        # Restore model state
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.dataset = model_data['dataset']
        instance.item_features_matrix = model_data['item_features_matrix']
        instance.user_features_matrix = model_data['user_features_matrix']
        
        print(f"📂 Model loaded from {filepath}")
        
        return instance
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model parameters and status
        """
        info = {
            'model_type': 'Content-Based (LightFM)',
            'no_components': self.no_components,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'item_alpha': self.item_alpha,
            'user_alpha': self.user_alpha,
            'is_fitted': self.is_fitted
        }
        
        return info


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Content-Based Model (LightFM) - Example Usage")
    print("=" * 70)
    
    print("\n📝 Example: Creating and using the model")
    print("\nfrom src.models.content_based import ContentBasedModel")
    print("\n# Prepare data with features")
    print("model = ContentBasedModel(no_components=128, loss='warp')")
    print("interactions, weights = model.prepare_dataset(interactions_df, item_features_dict)")
    print("\n# Train")
    print("model.fit(interactions, epochs=30)")
    print("\n# Get recommendations")
    print("recs = model.recommend(user_idx=0, n_items=1000, N=10)")
    print("\n✅ Model ready!")