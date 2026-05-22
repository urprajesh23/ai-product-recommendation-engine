"""
Helper utilities for Product Recommender System
Contains common functions used across the project
"""

import pickle
import json
import yaml
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


# ==================== File I/O Functions ====================

def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(data: Any, filepath: str, verbose: bool = True):
    """
    Save data to pickle file
    
    Args:
        data: Object to save
        filepath: Path to save pickle file
        verbose: Whether to print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        print(f"💾 Saved to {filepath} ({file_size:.2f} MB)")


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_json(data: Dict, filepath: str, indent: int = 2, verbose: bool = True):
    """
    Save data to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        indent: Indentation for pretty printing
        verbose: Whether to print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    if verbose:
        print(f"💾 Saved to {filepath}")


def load_yaml(filepath: str) -> Dict:
    """
    Load data from YAML file
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary with loaded data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def save_yaml(data: Dict, filepath: str, verbose: bool = True):
    """
    Save data to YAML file
    
    Args:
        data: Dictionary to save
        filepath: Path to save YAML file
        verbose: Whether to print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    if verbose:
        print(f"💾 Saved to {filepath}")


# ==================== Mapping Functions ====================

def load_mappings(filepath: str = 'data/processed/mappings.pkl') -> Dict[str, Dict]:
    """
    Load user/item ID mappings
    
    Args:
        filepath: Path to mappings pickle file
        
    Returns:
        Dictionary with mappings:
        {
            'user_id_to_idx': {...},
            'item_id_to_idx': {...},
            'idx_to_user_id': {...},
            'idx_to_item_id': {...}
        }
    """
    return load_pickle(filepath)


def save_mappings(mappings: Dict[str, Dict], filepath: str = 'data/processed/mappings.pkl'):
    """
    Save user/item ID mappings
    
    Args:
        mappings: Dictionary with mappings
        filepath: Path to save mappings
    """
    save_pickle(mappings, filepath)


def get_user_idx(user_id: str, mappings: Dict[str, Dict]) -> Optional[int]:
    """
    Convert user ID to internal index
    
    Args:
        user_id: Original user ID (string)
        mappings: Mappings dictionary
        
    Returns:
        User index (int) or None if not found
    """
    return mappings['user_id_to_idx'].get(user_id)


def get_item_idx(item_id: str, mappings: Dict[str, Dict]) -> Optional[int]:
    """
    Convert item ID to internal index
    
    Args:
        item_id: Original item ID (string)
        mappings: Mappings dictionary
        
    Returns:
        Item index (int) or None if not found
    """
    return mappings['item_id_to_idx'].get(item_id)


def get_user_id(user_idx: int, mappings: Dict[str, Dict]) -> Optional[str]:
    """
    Convert user index to original ID
    
    Args:
        user_idx: Internal user index (int)
        mappings: Mappings dictionary
        
    Returns:
        User ID (string) or None if not found
    """
    return mappings['idx_to_user_id'].get(user_idx)


def get_item_id(item_idx: int, mappings: Dict[str, Dict]) -> Optional[str]:
    """
    Convert item index to original ID
    
    Args:
        item_idx: Internal item index (int)
        mappings: Mappings dictionary
        
    Returns:
        Item ID (string) or None if not found
    """
    return mappings['idx_to_item_id'].get(item_idx)


# ==================== Sparse Matrix Functions ====================

def save_sparse_matrix(matrix: csr_matrix, filepath: str, verbose: bool = True):
    """
    Save sparse matrix to file
    
    Args:
        matrix: Scipy sparse matrix
        filepath: Path to save file
        verbose: Whether to print confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_npz(filepath, matrix)
    
    if verbose:
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        print(f"💾 Saved sparse matrix to {filepath} ({file_size:.2f} MB)")


def load_sparse_matrix(filepath: str) -> csr_matrix:
    """
    Load sparse matrix from file
    
    Args:
        filepath: Path to sparse matrix file
        
    Returns:
        Scipy sparse matrix
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return load_npz(filepath)


def convert_sparse_matrix_to_dict(matrix: csr_matrix) -> Dict[Tuple[int, int], float]:
    """
    Convert sparse matrix to dictionary format
    
    Args:
        matrix: Scipy sparse matrix
        
    Returns:
        Dictionary mapping (row, col) to value
    """
    cx = matrix.tocoo()
    return {(row, col): val for row, col, val in zip(cx.row, cx.col, cx.data)}


# ==================== Directory Functions ====================

def ensure_dir(directory: str) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# ==================== Configuration Functions ====================

def get_config(config_name: str = 'model_config', config_dir: str = 'configs') -> Dict:
    """
    Load configuration file
    
    Args:
        config_name: Name of config file (without extension)
        config_dir: Directory containing config files
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_dir) / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return load_yaml(config_path)


# ==================== Logging Functions ====================

def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (default: INFO)
        format_string: Custom format string
        
    Returns:
        Logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('recommender')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


# ==================== Timer Class ====================

class Timer:
    """
    Context manager for timing code execution
    
    Example:
        with Timer("Training model"):
            model.fit(data)
        
        # Output: Training model: 45.23 seconds
    """
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer
        
        Args:
            name: Name of operation being timed
            verbose: Whether to print timing info
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer"""
        if self.verbose:
            print(f"⏱️  {self.name} started...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """Stop timer and print elapsed time"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
        if self.verbose:
            print(f"✅ {self.name} completed in {self.format_time(self.elapsed)}")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format seconds into human-readable string
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string (e.g., "1h 23m 45s")
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"


# ==================== Display Functions ====================

def pretty_print_dict(d: Dict, indent: int = 0, title: Optional[str] = None):
    """
    Pretty print a dictionary
    
    Args:
        d: Dictionary to print
        indent: Indentation level
        title: Optional title to print first
    """
    if title:
        print("=" * 70)
        print(title)
        print("=" * 70)
    
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            pretty_print_dict(value, indent + 1)
        elif isinstance(value, (list, tuple)) and len(value) > 5:
            print("  " * indent + f"{key}: [{len(value)} items]")
        elif isinstance(value, float):
            print("  " * indent + f"{key}: {value:.4f}")
        else:
            print("  " * indent + f"{key}: {value}")
    
    if title and indent == 0:
        print("=" * 70)


# ==================== System Info Functions ====================

def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage
    
    Returns:
        Dictionary with memory statistics (in MB)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def print_system_info():
    """Print system information"""
    print("=" * 70)
    print("💻 System Information")
    print("=" * 70)
    
    # CPU
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Memory
    mem = psutil.virtual_memory()
    print(f"Memory Total: {mem.total / (1024**3):.2f} GB")
    print(f"Memory Available: {mem.available / (1024**3):.2f} GB")
    print(f"Memory Usage: {mem.percent}%")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    print(f"Disk Free: {disk.free / (1024**3):.2f} GB")
    print(f"Disk Usage: {disk.percent}%")
    
    print("=" * 70)


# ==================== Data Validation Functions ====================

def validate_recommendations(
    recommendations: List[Tuple[int, float]],
    n_items: int,
    check_duplicates: bool = True
) -> bool:
    """
    Validate recommendations format and content
    
    Args:
        recommendations: List of (item_idx, score) tuples
        n_items: Total number of items in catalog
        check_duplicates: Whether to check for duplicate items
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(recommendations, list):
        print("❌ Recommendations must be a list")
        return False
    
    if len(recommendations) == 0:
        print("⚠️  Warning: Empty recommendations list")
        return True
    
    # Check format
    for i, rec in enumerate(recommendations):
        if not isinstance(rec, (tuple, list)) or len(rec) != 2:
            print(f"❌ Invalid format at position {i}: {rec}")
            return False
        
        item_idx, score = rec
        
        # Check item index
        if not isinstance(item_idx, (int, np.integer)):
            print(f"❌ Item index must be integer at position {i}: {item_idx}")
            return False
        
        if item_idx < 0 or item_idx >= n_items:
            print(f"❌ Item index out of range at position {i}: {item_idx} (max: {n_items-1})")
            return False
        
        # Check score
        if not isinstance(score, (int, float, np.number)):
            print(f"❌ Score must be numeric at position {i}: {score}")
            return False
    
    # Check for duplicates
    if check_duplicates:
        item_indices = [rec[0] for rec in recommendations]
        if len(item_indices) != len(set(item_indices)):
            print("❌ Duplicate items found in recommendations")
            return False
    
    return True


# ==================== Utility Functions ====================

def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as string
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)


def generate_run_id() -> str:
    """
    Generate unique run ID for experiments
    
    Returns:
        Run ID string (e.g., "run_20240115_143022")
    """
    return f"run_{get_timestamp()}"


def chunks(lst: List, n: int):
    """
    Yield successive n-sized chunks from list
    
    Args:
        lst: List to chunk
        n: Chunk size
        
    Yields:
        Chunks of size n
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default value if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Utilities Module - Example Usage")
    print("=" * 70)
    
    # Timer example
    print("\n⏱️  Timer Example:")
    with Timer("Sample operation"):
        time.sleep(2)
    
    # Memory usage
    print("\n💾 Memory Usage:")
    mem = memory_usage()
    print(f"   RSS: {mem['rss_mb']:.2f} MB")
    print(f"   VMS: {mem['vms_mb']:.2f} MB")
    print(f"   Percent: {mem['percent']:.2f}%")
    
    # Pretty print
    print("\n📄 Pretty Print Example:")
    sample_dict = {
        'model': 'hybrid',
        'params': {
            'cf_weight': 0.6,
            'cb_weight': 0.4
        },
        'metrics': {
            'precision@10': 0.41,
            'recall@10': 0.38
        }
    }
    pretty_print_dict(sample_dict, title="Model Configuration")
    
    # Timestamp
    print(f"\n🕐 Current timestamp: {get_timestamp()}")
    print(f"   Run ID: {generate_run_id()}")
    
    print("\n" + "=" * 70)
    print("✅ Utilities module ready!")
    print("=" * 70)