"""
Models module for Product Recommender System
Contains collaborative filtering, content-based, and hybrid models
"""

from .collaborative import CollaborativeFilteringModel
from .content_based import ContentBasedModel
from .hybrid import HybridRecommender

__all__ = [
    'CollaborativeFilteringModel',
    'ContentBasedModel',
    'HybridRecommender',
]