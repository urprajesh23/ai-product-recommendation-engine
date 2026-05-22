"""
Data Preprocessing Module
Cleans and prepares data for recommendation models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import pickle
from typing import Tuple, Optional
import json


class DataPreprocessor:
    """
    Preprocess Amazon reviews data for recommendation system
    """
    
    def __init__(self, min_user_interactions=5, min_item_interactions=5):
        """
        Initialize preprocessor
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Mappings for later use
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        
    def load_data(self, filepath, file_format='csv'):
        """
        Load data from file
        
        Args:
            filepath: Path to data file
            file_format: 'csv' or 'json'
            
        Returns:
            pandas DataFrame
        """
        filepath = Path(filepath)
        
        print(f"📖 Loading data from {filepath}...")
        
        if file_format == 'csv':
            df = pd.read_csv(filepath)
        elif file_format == 'json':
            df = pd.read_json(filepath, lines=True)
        else:
            raise ValueError("file_format must be 'csv' or 'json'")
        
        print(f"✅ Loaded {len(df):,} rows")
        
        return df
    
    def clean_data(self, df):
        """
        Clean and standardize column names
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("🧹 Cleaning data...")
        
        # Standardize column names based on Amazon dataset structure
        column_mapping = {
            'reviewerID': 'user_id',
            'asin': 'item_id',
            'overall': 'rating',
            'unixReviewTime': 'timestamp'
        }
        
        # Keep only relevant columns
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df = df[available_cols].copy()
        df = df.rename(columns=column_mapping)
        
        # Remove duplicates (same user reviewing same item multiple times)
        initial_len = len(df)
        df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        removed = initial_len - len(df)
        if removed > 0:
            print(f"   Removed {removed:,} duplicate user-item pairs")
        
        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna(subset=['user_id', 'item_id', 'rating'])
        removed = initial_len - len(df)
        if removed > 0:
            print(f"   Removed {removed:,} rows with missing values")
        
        print(f"✅ Data cleaned. {len(df):,} rows remaining")
        
        return df
    
    def filter_sparse_data(self, df):
        """
        Remove users and items with too few interactions
        
        Args:
            df: DataFrame with user_id, item_id columns
            
        Returns:
            Filtered DataFrame
        """
        print(f"🔍 Filtering sparse users/items...")
        print(f"   Min user interactions: {self.min_user_interactions}")
        print(f"   Min item interactions: {self.min_item_interactions}")
        
        initial_len = len(df)
        initial_users = df['user_id'].nunique()
        initial_items = df['item_id'].nunique()
        
        # Iteratively filter until no more removals
        iteration = 0
        while True:
            iteration += 1
            prev_len = len(df)
            
            # Count interactions
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            
            # Filter users and items
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            
            df = df[df['user_id'].isin(valid_users)]
            df = df[df['item_id'].isin(valid_items)]
            
            # Break if no change
            if len(df) == prev_len:
                break
            
            print(f"   Iteration {iteration}: {len(df):,} rows remaining")
        
        final_users = df['user_id'].nunique()
        final_items = df['item_id'].nunique()
        
        print(f"✅ Filtering complete:")
        print(f"   Rows: {initial_len:,} → {len(df):,} ({(len(df)/initial_len)*100:.1f}%)")
        print(f"   Users: {initial_users:,} → {final_users:,}")
        print(f"   Items: {initial_items:,} → {final_items:,}")
        
        return df
    
    def create_implicit_feedback(self, df, threshold=4.0):
        """
        Convert explicit ratings to implicit feedback
        
        Args:
            df: DataFrame with rating column
            threshold: Rating threshold for positive feedback
            
        Returns:
            DataFrame with implicit column
        """
        print(f"🔄 Converting to implicit feedback (threshold={threshold})...")
        
        df = df.copy()
        df['implicit'] = (df['rating'] >= threshold).astype(int)
        
        positive = df['implicit'].sum()
        total = len(df)
        
        print(f"   Positive interactions: {positive:,} ({(positive/total)*100:.1f}%)")
        print(f"   Negative interactions: {total-positive:,} ({((total-positive)/total)*100:.1f}%)")
        
        return df
    
    def encode_ids(self, df):
        """
        Encode user and item IDs to sequential integers
        
        Args:
            df: DataFrame with user_id and item_id columns
            
        Returns:
            DataFrame with user_idx and item_idx columns
        """
        print("🔢 Encoding IDs to integers...")
        
        df = df.copy()
        
        # Encode users
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        
        # Encode items
        df['item_idx'] = self.item_encoder.fit_transform(df['item_id'])
        
        # Create mapping dictionaries
        self.user_id_to_idx = dict(zip(
            self.user_encoder.classes_,
            range(len(self.user_encoder.classes_))
        ))
        self.item_id_to_idx = dict(zip(
            self.item_encoder.classes_,
            range(len(self.item_encoder.classes_))
        ))
        self.idx_to_user_id = {v: k for k, v in self.user_id_to_idx.items()}
        self.idx_to_item_id = {v: k for k, v in self.item_id_to_idx.items()}
        
        print(f"✅ Encoded {df['user_idx'].nunique():,} users and {df['item_idx'].nunique():,} items")
        
        return df
    
    def train_test_split(self, df, test_size=0.2, random_state=42):
        """
        Time-based train/test split
        
        Args:
            df: DataFrame with timestamp column
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"📊 Splitting data (test_size={test_size})...")
        
        df = df.sort_values('timestamp').copy()
        
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"✅ Split complete:")
        print(f"   Train: {len(train_df):,} rows ({(len(train_df)/len(df))*100:.1f}%)")
        print(f"   Test: {len(test_df):,} rows ({(len(test_df)/len(df))*100:.1f}%)")
        
        return train_df, test_df
    
    def create_interaction_matrix(self, df):
        """
        Create sparse user-item interaction matrix
        
        Args:
            df: DataFrame with user_idx, item_idx, implicit columns
            
        Returns:
            scipy.sparse.csr_matrix
        """
        print("🔨 Creating interaction matrix...")
        
        n_users = df['user_idx'].max() + 1
        n_items = df['item_idx'].max() + 1
        
        matrix = csr_matrix(
            (df['implicit'].values, (df['user_idx'].values, df['item_idx'].values)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        sparsity = 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        
        print(f"✅ Matrix created:")
        print(f"   Shape: {matrix.shape} (users × items)")
        print(f"   Non-zero entries: {matrix.nnz:,}")
        print(f"   Sparsity: {sparsity:.2f}%")
        
        return matrix
    
    def save_processed_data(self, train_df, test_df, train_matrix, output_dir='data/processed'):
        """
        Save processed data and mappings
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            train_matrix: Training interaction matrix
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving processed data to {output_dir}...")
        
        # Save DataFrames
        train_df.to_csv(output_dir / 'train.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        # Save matrix
        with open(output_dir / 'train_matrix.pkl', 'wb') as f:
            pickle.dump(train_matrix, f)
        
        # Save encoders and mappings
        mappings = {
            'user_id_to_idx': self.user_id_to_idx,
            'item_id_to_idx': self.item_id_to_idx,
            'idx_to_user_id': self.idx_to_user_id,
            'idx_to_item_id': self.idx_to_item_id,
        }
        
        with open(output_dir / 'mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        with open(output_dir / 'mappings.json', 'w') as f:
            # Convert all keys to strings for JSON
            json_mappings = {
                k: {str(kk): vv for kk, vv in v.items()}
                for k, v in mappings.items()
            }
            json.dump(json_mappings, f, indent=2)
        
        print("✅ All data saved successfully!")
        print(f"   - train.csv")
        print(f"   - test.csv")
        print(f"   - train_matrix.pkl")
        print(f"   - mappings.pkl")
        print(f"   - mappings.json")
    
    def get_statistics(self, df):
        """
        Get dataset statistics
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_interactions': len(df),
            'n_users': df['user_id'].nunique(),
            'n_items': df['item_id'].nunique(),
            'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['item_id'].nunique())),
            'avg_interactions_per_user': len(df) / df['user_id'].nunique(),
            'avg_interactions_per_item': len(df) / df['item_id'].nunique(),
            'min_rating': df['rating'].min(),
            'max_rating': df['rating'].max(),
            'avg_rating': df['rating'].mean(),
        }
        
        return stats


def main():
    """Main preprocessing pipeline"""
    print("=" * 70)
    print("Amazon Product Reviews - Data Preprocessing Pipeline")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        min_user_interactions=5,
        min_item_interactions=5
    )
    
    # Load data
    df = preprocessor.load_data('data/raw/reviews_electronics.csv', file_format='csv')
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Get initial statistics
    print("\n" + "=" * 70)
    print("Initial Dataset Statistics")
    print("=" * 70)
    stats = preprocessor.get_statistics(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Filter sparse data
    df = preprocessor.filter_sparse_data(df)
    
    # Create implicit feedback
    df = preprocessor.create_implicit_feedback(df, threshold=4.0)
    
    # Encode IDs
    df = preprocessor.encode_ids(df)
    
    # Train/test split
    train_df, test_df = preprocessor.train_test_split(df, test_size=0.2)
    
    # Create interaction matrix
    train_matrix = preprocessor.create_interaction_matrix(train_df)
    
    # Get final statistics
    print("\n" + "=" * 70)
    print("Final Dataset Statistics")
    print("=" * 70)
    stats = preprocessor.get_statistics(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    # Save processed data
    preprocessor.save_processed_data(train_df, test_df, train_matrix)
    
    print("\n" + "=" * 70)
    print("✅ Preprocessing Complete!")
    print("=" * 70)
    print("Next steps:")
    print("1. Train collaborative filtering model")
    print("2. Train content-based model")
    print("3. Build hybrid recommender")


if __name__ == "__main__":
    main()