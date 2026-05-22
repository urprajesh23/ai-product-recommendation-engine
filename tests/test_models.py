"""
Unit tests for recommendation models
Tests collaborative filtering, content-based, and hybrid models
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.collaborative import CollaborativeFilteringModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridRecommender


# ==================== Fixtures ====================

@pytest.fixture
def sample_interaction_matrix():
    """Create a small sample interaction matrix for testing"""
    # Create a 100x50 sparse matrix (100 users, 50 items)
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_interactions = 500
    
    rows = np.random.randint(0, n_users, n_interactions)
    cols = np.random.randint(0, n_items, n_interactions)
    data = np.ones(n_interactions)
    
    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    return matrix


@pytest.fixture
def sample_dataframe():
    """Create a sample interactions DataFrame"""
    np.random.seed(42)
    n_interactions = 500
    
    df = pd.DataFrame({
        'user_idx': np.random.randint(0, 100, n_interactions),
        'item_idx': np.random.randint(0, 50, n_interactions),
        'implicit': np.ones(n_interactions, dtype=int),
        'timestamp': pd.date_range('2024-01-01', periods=n_interactions, freq='h')
    })
    
    return df


@pytest.fixture
def trained_cf_model(sample_interaction_matrix):
    """Create and train a collaborative filtering model"""
    model = CollaborativeFilteringModel(
        factors=32,
        regularization=0.01,
        iterations=5,  # Fewer iterations for faster tests
        alpha=1.0
    )
    model.fit(sample_interaction_matrix, show_progress=False)
    return model


@pytest.fixture
def trained_cb_model(sample_dataframe):
    """Create and train a content-based model"""
    model = ContentBasedModel(
        no_components=32,
        loss='warp',
        learning_rate=0.05
    )
    interactions, _ = model.prepare_dataset(sample_dataframe)
    model.fit(interactions, epochs=5, verbose=False)
    return model


@pytest.fixture
def trained_hybrid_model(sample_interaction_matrix, sample_dataframe):
    """Create and train a hybrid model"""
    model = HybridRecommender(
        cf_weight=0.6,
        cb_weight=0.4,
        cf_params={'factors': 32, 'iterations': 5, 'alpha': 1.0},
        cb_params={'no_components': 32, 'epochs': 5}
    )
    model.fit(
        train_matrix=sample_interaction_matrix,
        interactions_df=sample_dataframe,
        cf_epochs=5,
        cb_epochs=5,
        verbose=False
    )
    return model


# ==================== Collaborative Filtering Tests ====================

class TestCollaborativeFilteringModel:
    """Test suite for Collaborative Filtering model"""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters"""
        model = CollaborativeFilteringModel(
            factors=64,
            regularization=0.01,
            iterations=10,
            alpha=40
        )
        
        assert model.factors == 64
        assert model.regularization == 0.01
        assert model.iterations == 10
        assert model.alpha == 40
        assert model.is_fitted == False
    
    def test_model_training(self, sample_interaction_matrix):
        """Test that model trains successfully"""
        model = CollaborativeFilteringModel(factors=32, iterations=5)
        model.fit(sample_interaction_matrix, show_progress=False)
        
        assert model.is_fitted == True
        assert model.model.user_factors.shape[0] == sample_interaction_matrix.shape[0]
        assert model.model.item_factors.shape[0] == sample_interaction_matrix.shape[1]
        assert model.model.user_factors.shape[1] == 32  # factors
    
    def test_recommendations(self, trained_cf_model, sample_interaction_matrix):
        """Test that model generates recommendations"""
        user_idx = 0
        n_recs = 10
        
        recs = trained_cf_model.recommend(
            user_idx=user_idx,
            user_item_matrix=sample_interaction_matrix,
            N=n_recs,
            filter_already_liked=True
        )
        
        assert len(recs) <= n_recs
        assert all(isinstance(item, (int, np.integer)) for item, score in recs)
        assert all(isinstance(score, (float, np.floating)) for item, score in recs)
        assert all(score >= 0 for item, score in recs)
    
    def test_similar_items(self, trained_cf_model):
        """Test similar items functionality"""
        item_idx = 0
        n_similar = 5
        
        similar = trained_cf_model.similar_items(item_idx, N=n_similar)
        
        assert len(similar) <= n_similar
        assert all(isinstance(item, (int, np.integer)) for item, score in similar)
        assert item_idx not in [item for item, score in similar]  # Item itself should not be included
    
    def test_user_embedding(self, trained_cf_model):
        """Test user embedding retrieval"""
        user_idx = 0
        embedding = trained_cf_model.get_user_embedding(user_idx)
        
        assert embedding.shape[0] == 32  # factors
        assert isinstance(embedding, np.ndarray)
    
    def test_item_embedding(self, trained_cf_model):
        """Test item embedding retrieval"""
        item_idx = 0
        embedding = trained_cf_model.get_item_embedding(item_idx)
        
        assert embedding.shape[0] == 32  # factors
        assert isinstance(embedding, np.ndarray)
    
    def test_predict_score(self, trained_cf_model):
        """Test score prediction for user-item pair"""
        user_idx = 0
        item_idx = 5
        
        score = trained_cf_model.predict_score(user_idx, item_idx)
        
        assert isinstance(score, float)
    
    def test_model_save_load(self, trained_cf_model, tmp_path):
        """Test model save and load functionality"""
        # Save model
        model_path = tmp_path / "test_cf_model.pkl"
        trained_cf_model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = CollaborativeFilteringModel.load(str(model_path))
        
        assert loaded_model.is_fitted == True
        assert loaded_model.factors == trained_cf_model.factors
        assert np.array_equal(
            loaded_model.get_user_embedding(0),
            trained_cf_model.get_user_embedding(0)
        )
    
    def test_model_info(self, trained_cf_model):
        """Test model info retrieval"""
        info = trained_cf_model.get_model_info()
        
        assert 'model_type' in info
        assert 'factors' in info
        assert 'is_fitted' in info
        assert info['is_fitted'] == True


# ==================== Content-Based Tests ====================

class TestContentBasedModel:
    """Test suite for Content-Based model"""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters"""
        model = ContentBasedModel(
            no_components=64,
            loss='warp',
            learning_rate=0.05
        )
        
        assert model.no_components == 64
        assert model.loss == 'warp'
        assert model.learning_rate == 0.05
        assert model.is_fitted == False
    
    def test_dataset_preparation(self, sample_dataframe):
        """Test dataset preparation"""
        model = ContentBasedModel(no_components=32)
        
        interactions, weights = model.prepare_dataset(sample_dataframe)
        
        assert interactions is not None
        assert model.dataset is not None
    
    def test_model_training(self, sample_dataframe):
        """Test that model trains successfully"""
        model = ContentBasedModel(no_components=32)
        interactions, _ = model.prepare_dataset(sample_dataframe)
        
        model.fit(interactions, epochs=5, verbose=False)
        
        assert model.is_fitted == True
    
    def test_predictions(self, trained_cb_model):
        """Test that model makes predictions"""
        user_indices = np.array([0, 1, 2])
        item_indices = np.array([0, 1, 2])
        
        scores = trained_cb_model.predict(user_indices, item_indices)
        
        assert len(scores) == len(user_indices)
        assert all(isinstance(score, (float, np.floating)) for score in scores)
    
    def test_model_save_load(self, trained_cb_model, tmp_path):
        """Test model save and load functionality"""
        # Save model
        model_path = tmp_path / "test_cb_model.pkl"
        trained_cb_model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = ContentBasedModel.load(str(model_path))
        
        assert loaded_model.is_fitted == True
        assert loaded_model.no_components == trained_cb_model.no_components


# ==================== Hybrid Model Tests ====================

class TestHybridRecommender:
    """Test suite for Hybrid Recommender"""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters"""
        model = HybridRecommender(
            cf_weight=0.7,
            cb_weight=0.3
        )
        
        assert model.cf_weight == 0.7
        assert model.cb_weight == 0.3
        assert model.is_fitted == False
    
    def test_weight_validation(self):
        """Test that weights must sum to 1.0"""
        with pytest.raises(ValueError):
            HybridRecommender(cf_weight=0.5, cb_weight=0.6)
    
    def test_model_training(self, sample_interaction_matrix, sample_dataframe):
        """Test that hybrid model trains successfully"""
        model = HybridRecommender(
            cf_weight=0.6,
            cb_weight=0.4,
            cf_params={'factors': 32, 'iterations': 5},
            cb_params={'no_components': 32, 'epochs': 5}
        )
        
        model.fit(
            train_matrix=sample_interaction_matrix,
            interactions_df=sample_dataframe,
            cf_epochs=5,
            cb_epochs=5,
            verbose=False
        )
        
        assert model.is_fitted == True
        assert model.cf_model.is_fitted == True
        assert model.cb_model.is_fitted == True
    
    def test_recommendations(self, trained_hybrid_model, sample_interaction_matrix):
        """Test that hybrid model generates recommendations"""
        user_idx = 0
        n_recs = 10
        
        recs = trained_hybrid_model.recommend(
            user_idx=user_idx,
            train_matrix=sample_interaction_matrix,
            N=n_recs,
            filter_already_liked=True
        )
        
        assert len(recs) <= n_recs
        assert all(isinstance(item, (int, np.integer)) for item, score in recs)
    
    def test_cold_start_recommendations(self, trained_hybrid_model, sample_interaction_matrix):
        """Test cold-start handling with popularity fallback"""
        new_user_idx = 999  # User not in training data
        n_recs = 10
        
        recs = trained_hybrid_model.recommend(
            user_idx=new_user_idx,
            train_matrix=sample_interaction_matrix,
            N=n_recs,
            is_new_user=True
        )
        
        assert len(recs) <= n_recs
    
    def test_model_save_load(self, trained_hybrid_model, tmp_path):
        """Test model save and load functionality"""
        # Save model
        model_path = tmp_path / "test_hybrid_model.pkl"
        trained_hybrid_model.save(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = HybridRecommender.load(str(model_path))
        
        assert loaded_model.is_fitted == True
        assert loaded_model.cf_weight == trained_hybrid_model.cf_weight
        assert loaded_model.cb_weight == trained_hybrid_model.cb_weight
    
    def test_model_info(self, trained_hybrid_model):
        """Test model info retrieval"""
        info = trained_hybrid_model.get_model_info()
        
        assert 'model_type' in info
        assert 'cf_weight' in info
        assert 'cb_weight' in info
        assert 'is_fitted' in info
        assert info['is_fitted'] == True


# ==================== Integration Tests ====================

class TestModelIntegration:
    """Integration tests across models"""
    
    def test_all_models_produce_recommendations(
        self,
        trained_cf_model,
        trained_hybrid_model,
        sample_interaction_matrix
    ):
        """Test that all models can produce recommendations"""
        user_idx = 0
        n_recs = 5
        
        # CF recommendations
        cf_recs = trained_cf_model.recommend(
            user_idx, sample_interaction_matrix, N=n_recs
        )
        
        # Hybrid recommendations
        hybrid_recs = trained_hybrid_model.recommend(
            user_idx, sample_interaction_matrix, N=n_recs
        )
        
        assert len(cf_recs) > 0
        assert len(hybrid_recs) > 0
    
    def test_recommendation_consistency(self, trained_cf_model, sample_interaction_matrix):
        """Test that repeated recommendations are consistent"""
        user_idx = 0
        n_recs = 10
        
        recs1 = trained_cf_model.recommend(user_idx, sample_interaction_matrix, N=n_recs)
        recs2 = trained_cf_model.recommend(user_idx, sample_interaction_matrix, N=n_recs)
        
        # Recommendations should be identical
        assert recs1 == recs2


# ==================== Run Tests ====================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])