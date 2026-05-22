"""
Evaluation module for Product Recommender System
Contains metrics and evaluation utilities for recommendation models
"""

from .metrics import RecommenderMetrics, evaluate_model, cross_validate_model

__all__ = [
    'RecommenderMetrics',
    'evaluate_model',
    'cross_validate_model',
]