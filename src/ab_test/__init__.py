"""
A/B Testing Module for Product Recommender System
Simulates A/B tests to compare recommendation strategies
"""

from .simulator import ABTestSimulator, ABTestMetrics, simulate_user_session

__all__ = [
    'ABTestSimulator',
    'ABTestMetrics',
    'simulate_user_session',
]