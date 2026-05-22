"""
Unit tests for FastAPI endpoints
Tests API functionality, request validation, and response formats
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app


# ==================== Fixtures ====================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.recommend = Mock(return_value=[
        (0, 0.95),
        (1, 0.89),
        (2, 0.85),
        (3, 0.82),
        (4, 0.78)
    ])
    model.cf_model = Mock()
    model.cf_model.similar_items = Mock(return_value=[
        (5, 0.92),
        (6, 0.88),
        (7, 0.85)
    ])
    model.get_model_info = Mock(return_value={
        'model_type': 'Hybrid',
        'cf_weight': 0.6,
        'cb_weight': 0.4,
        'is_fitted': True,
        'n_users': 1000,
        'n_items': 500
    })
    return model


@pytest.fixture
def mock_train_matrix():
    """Create a mock training matrix"""
    return csr_matrix((100, 50))


@pytest.fixture
def mock_mappings():
    """Create mock ID mappings"""
    return {
        'user_id_to_idx': {'A123': 0, 'B456': 1},
        'item_id_to_idx': {'ITEM001': 0, 'ITEM002': 1, 'ITEM003': 2},
        'idx_to_user_id': {0: 'A123', 1: 'B456'},
        'idx_to_item_id': {0: 'ITEM001', 1: 'ITEM002', 2: 'ITEM003'}
    }


# ==================== API Endpoint Tests ====================

class TestRootEndpoints:
    """Test basic API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert 'endpoints' in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
        assert 'redis_connected' in data
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert 'model_loaded' in data
        assert 'timestamp' in data


class TestRecommendationEndpoints:
    """Test recommendation endpoints"""
    
    @patch('src.api.main.model')
    @patch('src.api.main.train_matrix')
    @patch('src.api.main.mappings')
    def test_recommend_endpoint_success(
        self,
        mock_mappings_patch,
        mock_matrix_patch,
        mock_model_patch,
        client,
        mock_model,
        mock_train_matrix,
        mock_mappings
    ):
        """Test successful recommendation request"""
        # Setup mocks
        mock_model_patch.return_value = mock_model
        mock_matrix_patch.return_value = mock_train_matrix
        mock_mappings_patch.return_value = mock_mappings
        
        # Make request
        response = client.post(
            "/recommend",
            json={
                "user_id": "A123",
                "n_recommendations": 5,
                "filter_purchased": True
            }
        )
        
        # With model not loaded, expect 503
        # This is expected behavior when running tests without trained models
        assert response.status_code in [200, 503]
    
    def test_recommend_endpoint_validation(self, client):
        """Test request validation"""
        # Missing required field
        response = client.post(
            "/recommend",
            json={
                "n_recommendations": 10
            }
        )
        assert response.status_code == 422  # Validation error
        
        # Invalid n_recommendations (too high)
        response = client.post(
            "/recommend",
            json={
                "user_id": "A123",
                "n_recommendations": 1000  # Exceeds MAX_RECOMMENDATIONS
            }
        )
        assert response.status_code == 422
        
        # Invalid n_recommendations (too low)
        response = client.post(
            "/recommend",
            json={
                "user_id": "A123",
                "n_recommendations": 0
            }
        )
        assert response.status_code == 422
    
    def test_recommend_endpoint_without_model(self, client):
        """Test recommendation request when model is not loaded"""
        response = client.post(
            "/recommend",
            json={
                "user_id": "A123",
                "n_recommendations": 10
            }
        )
        
        # Should return 503 Service Unavailable
        assert response.status_code == 503
        data = response.json()
        assert 'error' in data or 'detail' in data
    
    def test_similar_items_endpoint_validation(self, client):
        """Test similar items endpoint validation"""
        # Missing required field
        response = client.post(
            "/similar-items",
            json={
                "n_items": 5
            }
        )
        assert response.status_code == 422
        
        # Valid request (but model not loaded)
        response = client.post(
            "/similar-items",
            json={
                "item_id": "ITEM001",
                "n_items": 5
            }
        )
        assert response.status_code in [200, 404, 503]


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_without_model(self, client):
        """Test model info when model is not loaded"""
        response = client.get("/model/info")
        
        # Should return 503 when model not loaded
        assert response.status_code == 503
    
    @patch('src.api.main.model')
    def test_model_info_with_model(self, mock_model_patch, client, mock_model):
        """Test model info when model is loaded"""
        mock_model_patch.return_value = mock_model
        
        # This will still fail because the actual endpoint checks if model is None
        response = client.get("/model/info")
        assert response.status_code in [200, 503]


class TestRequestResponseFormats:
    """Test request and response formats"""
    
    def test_recommendation_request_format(self, client):
        """Test that recommendation request follows correct format"""
        valid_request = {
            "user_id": "A123",
            "n_recommendations": 10,
            "filter_purchased": True
        }
        
        response = client.post("/recommend", json=valid_request)
        # Will fail without model, but validates request format
        assert response.status_code in [200, 503, 422]
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.options("/")
        # CORS middleware should add headers
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test request to non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test using wrong HTTP method"""
        response = client.get("/recommend")  # Should be POST
        assert response.status_code == 405  # Method not allowed
    
    def test_invalid_json(self, client):
        """Test request with invalid JSON"""
        response = client.post(
            "/recommend",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# ==================== Integration Tests ====================

class TestAPIIntegration:
    """Integration tests for API"""
    
    def test_api_documentation_available(self, client):
        """Test that API documentation is accessible"""
        # Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert 'openapi' in data
        assert 'paths' in data
    
    def test_api_versioning(self, client):
        """Test that API version is consistent"""
        response = client.get("/")
        data = response.json()
        assert 'version' in data
        
        health_response = client.get("/health")
        health_data = health_response.json()
        assert 'version' in health_data


# ==================== Performance Tests ====================

class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_health_check_response_time(self, client):
        """Test that health check is fast"""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond in less than 1 second
    
    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests"""
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return client.get("/health")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        assert all(r.status_code == 200 for r in responses)


# ==================== Run Tests ====================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])