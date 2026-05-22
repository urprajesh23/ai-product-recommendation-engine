"""
Collaborative Filtering Model using ALS Matrix Factorization
Uses implicit library for implicit feedback recommendation
"""

import implicit
import numpy as np
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
from typing import List, Tuple, Optional


class CollaborativeFilteringModel:
    """
    Alternating Least Squares (ALS) Matrix Factorization for Implicit Feedback
    
    This model learns latent user and item factors from interaction data.
    Works well for implicit feedback (clicks, views, purchases) without explicit ratings.
    """
    
    def __init__(
        self, 
        factors=128, 
        regularization=0.01, 
        iterations=50, 
        alpha=40,
        random_state=42
    ):
        """
        Initialize ALS Collaborative Filtering model
        
        Args:
            factors: Number of latent factors (embedding dimensions)
            regularization: L2 regularization parameter
            iterations: Number of training iterations
            alpha: Confidence scaling parameter for implicit feedback
            random_state: Random seed for reproducibility
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Initialize the model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
            calculate_training_loss=True
        )
        
        self.is_fitted = False
        self.training_losses = []
        
    def fit(self, interaction_matrix: csr_matrix, show_progress=True):
        """
        Train the ALS model
        
        Args:
            interaction_matrix: scipy.sparse user-item interaction matrix (users × items)
            show_progress: Whether to show training progress
        """
        print(f"🚀 Training Collaborative Filtering Model...")
        print(f"   Matrix shape: {interaction_matrix.shape}")
        print(f"   Factors: {self.factors}")
        print(f"   Iterations: {self.iterations}")
        # Implicit library >= 0.6 expects user-item matrix (users × items)
        user_item_matrix = interaction_matrix.tocsr()
        
        # Multiply by alpha for confidence weighting
        user_item_matrix = user_item_matrix * self.alpha
        
        # Train the model
        self.model.fit(user_item_matrix, show_progress=show_progress)
        
        self.is_fitted = True
        
        print(f"✅ Training complete!")
        
    def recommend(
        self, 
        user_idx: int, 
        user_item_matrix: csr_matrix,
        N=10, 
        filter_already_liked=True,
        recalculate_user=False
    ) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user
        
        Args:
            user_idx: User index (integer)
            user_item_matrix: User-item interaction matrix
            N: Number of recommendations to return
            filter_already_liked: Whether to filter items user has already interacted with
            recalculate_user: Whether to recalculate user embedding (for new users)
            
        Returns:
            List of (item_idx, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from the model
        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=user_item_matrix[user_idx],
            N=N,
            filter_already_liked_items=filter_already_liked,
            recalculate_user=recalculate_user
        )
        
        # Return as list of tuples
        recommendations = list(zip(item_ids.tolist(), scores.tolist()))
        
        return recommendations
    
    def recommend_batch(
        self,
        user_indices: List[int],
        user_item_matrix: csr_matrix,
        N=10,
        filter_already_liked=True
    ) -> dict:
        """
        Get recommendations for multiple users
        
        Args:
            user_indices: List of user indices
            user_item_matrix: User-item interaction matrix
            N: Number of recommendations per user
            filter_already_liked: Whether to filter already liked items
            
        Returns:
            Dictionary mapping user_idx to list of (item_idx, score) tuples
        """
        recommendations = {}
        
        for user_idx in user_indices:
            try:
                recs = self.recommend(
                    user_idx, 
                    user_item_matrix, 
                    N=N, 
                    filter_already_liked=filter_already_liked
                )
                recommendations[user_idx] = recs
            except Exception as e:
                print(f"⚠️  Warning: Could not generate recommendations for user {user_idx}: {e}")
                recommendations[user_idx] = []
        
        return recommendations
    
    def similar_items(self, item_idx: int, N=10) -> List[Tuple[int, float]]:
        """
        Get similar items based on item embeddings
        
        Args:
            item_idx: Item index
            N: Number of similar items to return
            
        Returns:
            List of (item_idx, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        similar_item_ids, scores = self.model.similar_items(
            itemid=item_idx,
            N=N + 1  # +1 because the item itself will be included
        )
        
        # Remove the item itself (first result)
        similar_items = list(zip(similar_item_ids[1:].tolist(), scores[1:].tolist()))
        
        return similar_items[:N]
    
    def similar_users(self, user_idx: int, N=10) -> List[Tuple[int, float]]:
        """
        Get similar users based on user embeddings
        
        Args:
            user_idx: User index
            N: Number of similar users to return
            
        Returns:
            List of (user_idx, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar users")
        
        similar_user_ids, scores = self.model.similar_users(
            userid=user_idx,
            N=N + 1  # +1 because the user itself will be included
        )
        
        # Remove the user itself (first result)
        similar_users = list(zip(similar_user_ids[1:].tolist(), scores[1:].tolist()))
        
        return similar_users[:N]
    
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """
        Get latent factor embedding for a user
        
        Args:
            user_idx: User index
            
        Returns:
            numpy array of user factors (shape: factors,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing embeddings")
        
        return self.model.user_factors[user_idx]
    
    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """
        Get latent factor embedding for an item
        
        Args:
            item_idx: Item index
            
        Returns:
            numpy array of item factors (shape: factors,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing embeddings")
        
        return self.model.item_factors[item_idx]
    
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """
        Predict score for a user-item pair
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Predicted score (float)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Dot product of user and item embeddings
        score = np.dot(
            self.get_user_embedding(user_idx),
            self.get_item_embedding(item_idx)
        )
        
        return float(score)
    
    def explain_recommendation(
        self, 
        user_idx: int, 
        item_idx: int,
        user_item_matrix: csr_matrix,
        N=5
    ) -> dict:
        """
        Explain why an item was recommended to a user
        
        Args:
            user_idx: User index
            item_idx: Recommended item index
            user_item_matrix: User-item interaction matrix
            N: Number of contributing items to show
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating explanations")
        
        # Get items the user has interacted with
        user_items = user_item_matrix[user_idx].indices
        
        if len(user_items) == 0:
            return {
                'score': 0.0,
                'contributing_items': [],
                'explanation': 'User has no interaction history'
            }
        
        # Calculate contribution of each user item to the recommendation
        item_embedding = self.get_item_embedding(item_idx)
        contributions = []
        
        for user_item_idx in user_items:
            user_item_embedding = self.get_item_embedding(user_item_idx)
            contribution = np.dot(user_item_embedding, item_embedding)
            contributions.append((user_item_idx, float(contribution)))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Get prediction score
        score = self.predict_score(user_idx, item_idx)
        
        return {
            'score': score,
            'contributing_items': contributions[:N],
            'explanation': f'Based on your interaction with {len(user_items)} items, particularly items similar to this one'
        }
    
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
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
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
            CollaborativeFilteringModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            factors=model_data['factors'],
            regularization=model_data['regularization'],
            iterations=model_data['iterations'],
            alpha=model_data['alpha'],
            random_state=model_data['random_state']
        )
        
        # Restore model state
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        
        print(f"📂 Model loaded from {filepath}")
        
        return instance
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model parameters and status
        """
        info = {
            'model_type': 'Collaborative Filtering (ALS)',
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info['n_users'] = self.model.user_factors.shape[0]
            info['n_items'] = self.model.item_factors.shape[0]
        
        return info


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Collaborative Filtering Model - Example Usage")
    print("=" * 70)
    
    # This is just for testing - actual usage requires real data
    print("\n📝 Example: Creating and using the model")
    print("\nfrom src.models.collaborative import CollaborativeFilteringModel")
    print("from src.data.preprocess import DataPreprocessor")
    print("\n# Load preprocessed data")
    print("preprocessor = DataPreprocessor()")
    print("train_matrix = preprocessor.load_matrix('data/processed/train_matrix.pkl')")
    print("\n# Train model")
    print("model = CollaborativeFilteringModel(factors=128, iterations=50)")
    print("model.fit(train_matrix)")
    print("\n# Get recommendations")
    print("recommendations = model.recommend(user_idx=0, user_item_matrix=train_matrix, N=10)")
    print("print(recommendations)")
    print("\n# Save model")
    print("model.save('models/collaborative_model.pkl')")
    
    print("\n✅ Model ready to use!")