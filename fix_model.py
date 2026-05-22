import os
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from src.models.collaborative import CollaborativeFilteringModel
from src.models.hybrid import HybridRecommender

def fix_models():
    print("Fixing models...")
    
    # Load training matrix
    with open('data/processed/train_matrix.pkl', 'rb') as f:
        train_matrix = pickle.load(f)
    print(f"Loaded train_matrix with shape {train_matrix.shape}")
    
    # 1. Fix Collaborative Model
    print("Training Collaborative Filtering Model...")
    cf_model = CollaborativeFilteringModel(factors=128, iterations=50)
    cf_model.fit(train_matrix, show_progress=False)
    cf_model.save('models/collaborative_model.pkl')
    
    # 2. Fix Hybrid Model
    print("Training Hybrid Model...")
    hybrid_model = HybridRecommender(cf_params={'factors': 128, 'iterations': 50})
    # The hybrid model fits both CF and CB models
    # Wait, HybridRecommender has a fit() method that we need to call
    # But it also requires item_features for the CB part, if any. 
    # Let's check if we can just re-initialize the HybridRecommender and fit it.
    
    try:
        # Check if item_features exist
        if os.path.exists('data/processed/item_features.pkl'):
            with open('data/processed/item_features.pkl', 'rb') as f:
                item_features = pickle.load(f)
            hybrid_model.fit(train_matrix, item_features=item_features)
        else:
            hybrid_model.fit(train_matrix)
        hybrid_model.save('models/hybrid_model.pkl')
    except Exception as e:
        print(f"Failed to fix hybrid model: {e}")
        
    print("Done!")

if __name__ == '__main__':
    fix_models()
