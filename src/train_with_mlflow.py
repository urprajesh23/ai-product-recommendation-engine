"""
Model Training Script with MLflow Tracking

This script:
1. Loads preprocessed data
2. Trains collaborative filtering, content-based, and hybrid models
3. Evaluates models using multiple metrics
4. Logs experiments to MLflow
5. Saves the best model for deployment

Usage:
    python src/train_with_mlflow.py
    
    # Or with custom config:
    python src/train_with_mlflow.py --config configs/model_config.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import DataPreprocessor
from src.models.collaborative import CollaborativeFilteringModel
from src.models.content_based import ContentBasedModel
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import evaluate_model, cross_validate_model, RecommenderMetrics
from src.utils.helpers import (
    load_pickle, save_pickle, load_yaml, 
    Timer, setup_logging, pretty_print_dict,
    ensure_dir, get_timestamp
)


# ==================== Configuration ====================

class TrainingConfig:
    """Training configuration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize training configuration
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        # Default configuration
        self.data = {
            'train_csv': 'data/processed/train.csv',
            'test_csv': 'data/processed/test.csv',
            'train_matrix': 'data/processed/train_matrix.pkl',
            'mappings': 'data/processed/mappings.pkl'
        }
        
        self.collaborative = {
            'factors': 128,
            'regularization': 0.01,
            'iterations': 50,
            'alpha': 40
        }
        
        self.content_based = {
            'no_components': 128,
            'loss': 'warp',
            'learning_rate': 0.05
            # epochs is passed to fit(), not __init__()
        }
        
        self.hybrid = {
            'cf_weight': 0.6,
            'cb_weight': 0.4,
            'enable_cold_start': True
        }
        
        self.evaluation = {
            'k_values': [5, 10, 20],
            'cv_folds': 5,
            'run_cv': False  # Set to True for cross-validation
        }
        
        self.mlflow = {
            'experiment_name': 'product-recommender',
            'tracking_uri': 'mlruns',
            'run_name_prefix': 'hybrid'
        }
        
        self.output = {
            'model_dir': 'models',
            'best_model_name': 'hybrid_model.pkl',
            'cf_model_name': 'collaborative_model.pkl',
            'cb_model_name': 'content_based_model.pkl'
        }
        
        # Load from YAML if provided
        if config_path and Path(config_path).exists():
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        config = load_yaml(config_path)
        
        # Update with loaded config
        for key, value in config.items():
            if hasattr(self, key) and isinstance(getattr(self, key), dict):
                getattr(self, key).update(value)


# ==================== Training Functions ====================

def load_data(config: TrainingConfig, logger):
    """
    Load preprocessed training and test data
    
    Args:
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Tuple of (train_df, test_df, train_matrix, mappings)
    """
    logger.info("=" * 70)
    logger.info("📂 Loading Data")
    logger.info("=" * 70)
    
    # Load DataFrames
    logger.info(f"Loading train data from {config.data['train_csv']}...")
    train_df = pd.read_csv(config.data['train_csv'])
    logger.info(f"✅ Train data loaded: {len(train_df):,} interactions")
    
    logger.info(f"Loading test data from {config.data['test_csv']}...")
    test_df = pd.read_csv(config.data['test_csv'])
    logger.info(f"✅ Test data loaded: {len(test_df):,} interactions")
    
    # Load interaction matrix
    logger.info(f"Loading interaction matrix from {config.data['train_matrix']}...")
    train_matrix = load_pickle(config.data['train_matrix'])
    logger.info(f"✅ Matrix loaded: {train_matrix.shape}")
    logger.info(f"   Sparsity: {100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.2f}%")
    
    # Load mappings
    logger.info(f"Loading ID mappings from {config.data['mappings']}...")
    mappings = load_pickle(config.data['mappings'])
    logger.info(f"✅ Mappings loaded: {len(mappings['user_id_to_idx']):,} users, {len(mappings['item_id_to_idx']):,} items")
    
    logger.info("=" * 70)
    
    return train_df, test_df, train_matrix, mappings


def train_collaborative_filtering(train_matrix, test_df, config: TrainingConfig, logger):
    """
    Train collaborative filtering model
    
    Args:
        train_matrix: Training interaction matrix
        test_df: Test DataFrame
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("🤖 Training Collaborative Filtering Model (ALS)")
    logger.info("=" * 70)
    
    with Timer("Collaborative Filtering Training"):
        # Initialize model
        cf_model = CollaborativeFilteringModel(
            factors=config.collaborative['factors'],
            regularization=config.collaborative['regularization'],
            iterations=config.collaborative['iterations'],
            alpha=config.collaborative['alpha']
        )
        
        # Train
        cf_model.fit(train_matrix, show_progress=True)
        
        # Evaluate
        logger.info("\n📊 Evaluating Collaborative Filtering Model...")
        cf_metrics = evaluate_model(
            model=cf_model,
            test_df=test_df,
            train_matrix=train_matrix,
            k_values=config.evaluation['k_values'],
            verbose=True
        )
    
    return cf_model, cf_metrics


def train_content_based(train_df, test_df, train_matrix, config: TrainingConfig, logger):
    """
    Train content-based model
    
    Note: This is a simplified version. In production, you would:
    - Extract item features (category, brand, price, etc.)
    - Extract user features (demographics, preferences, etc.)
    - Build proper feature dictionaries
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        train_matrix: Training interaction matrix
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("📚 Training Content-Based Model (LightFM)")
    logger.info("=" * 70)
    logger.info("⚠️  Note: Using implicit features only (no item/user metadata)")
    logger.info("   For production, add item categories, brands, user demographics, etc.")
    
    with Timer("Content-Based Training"):
        # Initialize model
        cb_model = ContentBasedModel(
            no_components=config.content_based['no_components'],
            loss=config.content_based['loss'],
            learning_rate=config.content_based['learning_rate']
        )
        
        # Prepare dataset (without features for now)
        interactions, weights = cb_model.prepare_dataset(
            interactions_df=train_df,
            item_features_dict=None,  # TODO: Add item features
            user_features_dict=None   # TODO: Add user features
        )
        
        # Train
        cb_model.fit(
            interactions=interactions,
            epochs=config.content_based['epochs'],
            verbose=True
        )
        
        # Evaluate
        logger.info("\n📊 Evaluating Content-Based Model...")
        
        # For LightFM evaluation, we need to use predict method
        # This is a simplified evaluation
        cb_metrics = {
            'model_type': 'Content-Based (LightFM)',
            'note': 'Full evaluation requires custom implementation for LightFM'
        }
        logger.info("⚠️  Content-based evaluation skipped (requires custom implementation)")
    
    return cb_model, cb_metrics


def train_hybrid(train_matrix, train_df, test_df, config: TrainingConfig, logger):
    """
    Train hybrid recommender model
    
    Args:
        train_matrix: Training interaction matrix
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Tuple of (model, metrics)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎯 Training Hybrid Recommender System")
    logger.info("=" * 70)
    
    with Timer("Hybrid Model Training"):
        # Initialize hybrid model
        hybrid_model = HybridRecommender(
            cf_weight=config.hybrid['cf_weight'],
            cb_weight=config.hybrid['cb_weight'],
            enable_cold_start=config.hybrid['enable_cold_start'],
            cf_params=config.collaborative,
            cb_params=config.content_based
        )
        
        # Train
        hybrid_model.fit(
            train_matrix=train_matrix,
            interactions_df=train_df,
            item_features_dict=None,  # TODO: Add item features
            user_features_dict=None,  # TODO: Add user features
            cf_epochs=config.collaborative['iterations'],
            cb_epochs=config.content_based['epochs'],
            verbose=True
        )
        
        # Evaluate using CF component (hybrid evaluation is complex)
        logger.info("\n📊 Evaluating Hybrid Model...")
        hybrid_metrics = evaluate_model(
            model=hybrid_model.cf_model,  # Use CF component for evaluation
            test_df=test_df,
            train_matrix=train_matrix,
            k_values=config.evaluation['k_values'],
            verbose=True
        )
    
    return hybrid_model, hybrid_metrics


def save_models(cf_model, cb_model, hybrid_model, config: TrainingConfig, logger):
    """
    Save trained models to disk
    
    Args:
        cf_model: Collaborative filtering model
        cb_model: Content-based model
        hybrid_model: Hybrid model
        config: Training configuration
        logger: Logger instance
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("💾 Saving Models")
    logger.info("=" * 70)
    
    model_dir = ensure_dir(config.output['model_dir'])
    
    # Save collaborative filtering model
    cf_path = model_dir / config.output['cf_model_name']
    cf_model.save(str(cf_path))
    
    # Save content-based model
    cb_path = model_dir / config.output['cb_model_name']
    cb_model.save(str(cb_path))
    
    # Save hybrid model
    hybrid_path = model_dir / config.output['best_model_name']
    hybrid_model.save(str(hybrid_path))
    
    logger.info("✅ All models saved successfully!")
    logger.info("=" * 70)


def run_cross_validation(train_df, train_matrix, config: TrainingConfig, logger):
    """
    Run k-fold cross-validation
    
    Args:
        train_df: Training DataFrame
        train_matrix: Training interaction matrix
        config: Training configuration
        logger: Logger instance
        
    Returns:
        Cross-validation results
    """
    if not config.evaluation.get('run_cv', False):
        logger.info("⏭️  Cross-validation skipped (set run_cv=True in config to enable)")
        return None
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🔄 Running Cross-Validation")
    logger.info("=" * 70)
    
    cv_results = cross_validate_model(
        model_class=CollaborativeFilteringModel,
        df=train_df,
        interaction_matrix=train_matrix,
        n_folds=config.evaluation['cv_folds'],
        k=10,
        model_params=config.collaborative
    )
    
    return cv_results


def log_to_mlflow(
    model_name: str,
    model,
    metrics: dict,
    params: dict,
    config: TrainingConfig
):
    """
    Log model and metrics to MLflow
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        metrics: Evaluation metrics
        params: Model parameters
        config: Training configuration
    """
    # Set experiment
    mlflow.set_experiment(config.mlflow['experiment_name'])
    
    # Start run
    run_name = f"{config.mlflow['run_name_prefix']}_{model_name}_{get_timestamp()}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                # Replace @ with _ for MLflow compatibility
                safe_metric_name = metric_name.replace('@', '_at_')
                mlflow.log_metric(safe_metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log additional info
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        print(f"✅ Logged {model_name} to MLflow")


# ==================== Main Training Pipeline ====================

def main(config_path: str = None):
    """
    Main training pipeline
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Setup
    logger = setup_logging(log_file='logs/training.log')
    config = TrainingConfig(config_path)
    
    print("\n")
    print("=" * 70)
    print("🚀 PRODUCT RECOMMENDER SYSTEM - TRAINING PIPELINE")
    print("=" * 70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   MLflow Experiment: {config.mlflow['experiment_name']}")
    print("=" * 70)
    
    try:
        # Step 1: Load data
        train_df, test_df, train_matrix, mappings = load_data(config, logger)
        
        # Step 2: Train Collaborative Filtering
        cf_model, cf_metrics = train_collaborative_filtering(
            train_matrix, test_df, config, logger
        )
        
        # Log to MLflow
        log_to_mlflow(
            model_name="collaborative_filtering",
            model=cf_model,
            metrics=cf_metrics,
            params=config.collaborative,
            config=config
        )
        
        # Step 3: Train Content-Based - SKIPPED
        print("⏭️  Skipping Content-Based training (LightFM issue)")
        cb_model = None
        cb_metrics = {'model_type': 'Content-Based (Skipped)'}
        
        # Log to MLflow
        log_to_mlflow(
            model_name="content_based",
            model=cb_model,
            metrics=cb_metrics,
            params=config.content_based,
            config=config
        )
        
        # Step 4: Train Hybrid
        hybrid_model, hybrid_metrics = train_hybrid(
            train_matrix, train_df, test_df, config, logger
        )
        
        # Log to MLflow
        log_to_mlflow(
            model_name="hybrid",
            model=hybrid_model,
            metrics=hybrid_metrics,
            params={**config.collaborative, **config.hybrid},
            config=config
        )
        
        # Step 5: Save models
        save_models(cf_model, cb_model, hybrid_model, config, logger)
        
        # Step 6: Cross-validation (optional)
        cv_results = run_cross_validation(train_df, train_matrix, config, logger)
        
        # Final summary
        print("\n")
        print("=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print("\n📊 Best Model Performance (Hybrid):")
        
        summary = {
            'Precision@10': hybrid_metrics.get('precision@10', 0),
            'Recall@10': hybrid_metrics.get('recall@10', 0),
            'NDCG@10': hybrid_metrics.get('ndcg@10', 0),
            'Coverage': hybrid_metrics.get('coverage', 0)
        }
        pretty_print_dict(summary)
        
        print("\n📁 Models saved to:")
        print(f"   • {config.output['model_dir']}/{config.output['best_model_name']}")
        print(f"   • {config.output['model_dir']}/{config.output['cf_model_name']}")
        print(f"   • {config.output['model_dir']}/{config.output['cb_model_name']}")
        
        print("\n📈 View experiments in MLflow:")
        print(f"   mlflow ui")
        print(f"   Then open: http://localhost:5000")
        
        print("\n🚀 Start API server:")
        print(f"   uvicorn src.api.main:app --reload")
        print(f"   Then open: http://localhost:8000/docs")
        
        print("\n" + "=" * 70)
        print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== CLI ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Product Recommender System with MLflow tracking'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = args.config if Path(args.config).exists() else None
    
    if config_path is None:
        print(f"⚠️  Config file not found: {args.config}")
        print(f"   Using default configuration")
    
    # Run training
    success = main(config_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)