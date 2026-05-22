"""
Evaluation Metrics for Recommendation Systems
Implements industry-standard metrics: Precision, Recall, NDCG, MAP, Coverage, etc.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from scipy.sparse import csr_matrix
from collections import defaultdict
import warnings


class RecommenderMetrics:
    """
    Comprehensive metrics for evaluating recommendation systems
    
    Implements:
    - Precision@K: Proportion of recommended items that are relevant
    - Recall@K: Proportion of relevant items that are recommended
    - NDCG@K: Normalized Discounted Cumulative Gain (position-aware)
    - MAP@K: Mean Average Precision
    - Hit Rate@K: Proportion of users with at least one relevant item
    - Coverage: Proportion of items that get recommended
    - Diversity: How diverse the recommendations are
    """
    
    @staticmethod
    def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        Precision@K: What fraction of recommended items are relevant?
        
        Formula: (# of recommended items that are relevant) / k
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            Precision score (0.0 to 1.0)
            
        Example:
            recommended = [1, 2, 3, 4, 5]
            relevant = [2, 4, 6, 8]
            precision_at_k(recommended, relevant, k=5) = 2/5 = 0.4
        """
        if k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = len(set(recommended_k) & relevant_set)
        
        return hits / k
    
    @staticmethod
    def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        Recall@K: What fraction of relevant items are recommended?
        
        Formula: (# of recommended items that are relevant) / (total # of relevant items)
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            Recall score (0.0 to 1.0)
            
        Example:
            recommended = [1, 2, 3, 4, 5]
            relevant = [2, 4, 6, 8]
            recall_at_k(recommended, relevant, k=5) = 2/4 = 0.5
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        hits = len(set(recommended_k) & relevant_set)
        
        return hits / len(relevant_set)
    
    @staticmethod
    def f1_score_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        F1 Score@K: Harmonic mean of Precision and Recall
        
        Formula: 2 * (precision * recall) / (precision + recall)
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        precision = RecommenderMetrics.precision_at_k(recommended_items, relevant_items, k)
        recall = RecommenderMetrics.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Accounts for the position of relevant items. Items ranked higher get more weight.
        
        Formula: DCG@K / IDCG@K
        where DCG = sum(rel_i / log2(i + 2)) for i in top K
        
        Args:
            recommended_items: List of recommended item indices (ordered by rank)
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            NDCG score (0.0 to 1.0)
            
        Example:
            recommended = [1, 2, 3, 4, 5]  # ranked by relevance
            relevant = [2, 4, 6]
            
            Relevance vector: [0, 1, 0, 1, 0]
            DCG = 1/log2(3) + 1/log2(5) = 0.63 + 0.43 = 1.06
            
            Ideal order: [2, 4, 1, 3, 5]
            IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
            
            NDCG = 1.06 / 1.63 = 0.65
        """
        if len(relevant_items) == 0:
            return 0.0
        
        # Create binary relevance vector
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        relevance = [1 if item in relevant_set else 0 for item in recommended_k]
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
        
        # Calculate IDCG (Ideal DCG - if items were perfectly ranked)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def average_precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        Average Precision@K
        
        Considers both precision and the position of relevant items.
        
        Formula: (1/min(k, |relevant|)) * sum(P@i * rel_i) for i=1 to k
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            AP score (0.0 to 1.0)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_k, start=1):
            if item in relevant_set:
                num_hits += 1
                precision_at_i = num_hits / i
                score += precision_at_i
        
        if num_hits == 0:
            return 0.0
        
        return score / min(len(relevant_set), k)
    
    @staticmethod
    def hit_rate_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        Hit Rate@K: Binary metric indicating if at least one relevant item is in top-K
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            1.0 if at least one hit, 0.0 otherwise
        """
        recommended_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        
        return 1.0 if len(recommended_k & relevant_set) > 0 else 0.0
    
    @staticmethod
    def mrr_at_k(recommended_items: List[int], relevant_items: List[int], k: int = 10) -> float:
        """
        Mean Reciprocal Rank@K
        
        The reciprocal of the rank of the first relevant item.
        
        Args:
            recommended_items: List of recommended item indices
            relevant_items: List of ground truth relevant item indices
            k: Number of top recommendations to consider
            
        Returns:
            MRR score (0.0 to 1.0)
            
        Example:
            If first relevant item is at position 3: MRR = 1/3 = 0.333
        """
        recommended_k = recommended_items[:k]
        relevant_set = set(relevant_items)
        
        for idx, item in enumerate(recommended_k, start=1):
            if item in relevant_set:
                return 1.0 / idx
        
        return 0.0
    
    @staticmethod
    def coverage(all_recommended_items: Set[int], total_items: int) -> float:
        """
        Catalog Coverage: What fraction of items ever get recommended?
        
        Args:
            all_recommended_items: Set of all items that were recommended to any user
            total_items: Total number of items in catalog
            
        Returns:
            Coverage score (0.0 to 1.0)
            
        Example:
            If 500 out of 1000 items were recommended: coverage = 0.5
        """
        if total_items == 0:
            return 0.0
        
        return len(all_recommended_items) / total_items
    
    @staticmethod
    def diversity(recommended_items: List[int], item_similarity_matrix: Optional[np.ndarray] = None) -> float:
        """
        Intra-list Diversity: How diverse are the recommendations?
        
        Measures average dissimilarity between all pairs of recommended items.
        
        Args:
            recommended_items: List of recommended item indices
            item_similarity_matrix: Pairwise item similarity matrix (optional)
            
        Returns:
            Diversity score (0.0 to 1.0, higher is more diverse)
        """
        if len(recommended_items) <= 1:
            return 0.0
        
        if item_similarity_matrix is None:
            # If no similarity matrix, assume all items are different
            return 1.0
        
        # Calculate average dissimilarity
        total_dissimilarity = 0.0
        count = 0
        
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item_i = recommended_items[i]
                item_j = recommended_items[j]
                
                # Dissimilarity = 1 - similarity
                similarity = item_similarity_matrix[item_i, item_j]
                total_dissimilarity += (1 - similarity)
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_dissimilarity / count
    
    @staticmethod
    def novelty(recommended_items: List[int], item_popularity: Dict[int, float]) -> float:
        """
        Novelty: How novel/surprising are the recommendations?
        
        Novel items are those that are less popular.
        
        Args:
            recommended_items: List of recommended item indices
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Novelty score (higher means more novel)
        """
        if len(recommended_items) == 0:
            return 0.0
        
        # Novelty = -log2(popularity)
        # More popular items have lower novelty
        novelties = []
        for item in recommended_items:
            pop = item_popularity.get(item, 1e-6)  # Small value if item not found
            novelties.append(-np.log2(pop + 1e-10))  # Add small value to avoid log(0)
        
        return np.mean(novelties)


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    train_matrix: csr_matrix,
    k_values: List[int] = [5, 10, 20],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a recommendation model on test data
    
    Args:
        model: Trained recommendation model with recommend() method
        test_df: Test DataFrame with user_idx, item_idx columns
        train_matrix: Training interaction matrix (to pass to model)
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        verbose: Whether to print progress
        
    Returns:
        Dictionary with metric results
        
    Example:
        results = evaluate_model(
            model=cf_model,
            test_df=test_df,
            train_matrix=train_matrix,
            k_values=[10, 20]
        )
        # Results: {'precision@10': 0.41, 'recall@10': 0.38, ...}
    """
    print("=" * 70)
    print("📊 Evaluating Recommendation Model")
    print("=" * 70)
    
    # Group test data by user
    user_test_items = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    users = list(user_test_items.keys())
    
    print(f"   Test users: {len(users):,}")
    print(f"   K values: {k_values}")
    print()
    
    # Initialize metrics storage
    metrics = defaultdict(list)
    
    # Track items recommended (for coverage)
    all_recommended_items = set()
    
    # Evaluate each user
    total_users = len(users)
    failed_users = 0
    
    for idx, user_idx in enumerate(users):
        if verbose and (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{total_users} users evaluated...", end='\r')
        
        # Get ground truth
        relevant_items = user_test_items[user_idx]
        
        if len(relevant_items) == 0:
            continue
        

            
        # Get recommendations
        try:
            # Check if user exists in training matrix
            if user_idx >= train_matrix.shape[0]:
                if verbose and failed_users == 0:
                    print(f"\n⚠️  Warning: User {user_idx} not in training matrix")
                failed_users += 1
                continue
            
            # Get max K recommendations
            max_k = max(k_values)
            recs = model.recommend(user_idx, train_matrix, N=max_k, filter_already_liked=True)
            recommended_items = [item for item, score in recs]

            # Track for coverage
            all_recommended_items.update(recommended_items)
            
        except Exception as e:
            if verbose and failed_users == 0:
                print(f"\n⚠️  Warning: Could not get recommendations for user {user_idx}: {e}")
            failed_users += 1
            continue
        
        # Calculate metrics for each K
        for k in k_values:
            metrics[f'precision@{k}'].append(
                RecommenderMetrics.precision_at_k(recommended_items, relevant_items, k)
            )
            metrics[f'recall@{k}'].append(
                RecommenderMetrics.recall_at_k(recommended_items, relevant_items, k)
            )
            metrics[f'ndcg@{k}'].append(
                RecommenderMetrics.ndcg_at_k(recommended_items, relevant_items, k)
            )
            metrics[f'hit_rate@{k}'].append(
                RecommenderMetrics.hit_rate_at_k(recommended_items, relevant_items, k)
            )
            metrics[f'map@{k}'].append(
                RecommenderMetrics.average_precision_at_k(recommended_items, relevant_items, k)
            )
            metrics[f'mrr@{k}'].append(
                RecommenderMetrics.mrr_at_k(recommended_items, relevant_items, k)
            )
    
    if verbose:
        print()  # New line after progress
    
    # Calculate averages
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = np.mean(values)
    
    # Add coverage
    total_items = train_matrix.shape[1]
    results['coverage'] = RecommenderMetrics.coverage(all_recommended_items, total_items)
    
    # Add metadata
    results['n_users_evaluated'] = len(users) - failed_users
    results['n_users_failed'] = failed_users
    
    # Print results
    print()
    print("=" * 70)
    print("📈 Evaluation Results")
    print("=" * 70)
    
    for k in k_values:
        print(f"\n📍 Metrics @ K={k}:")
        print(f"   Precision@{k}:  {results.get(f'precision@{k}', 0):.4f}")
        print(f"   Recall@{k}:     {results.get(f'recall@{k}', 0):.4f}")
        print(f"   NDCG@{k}:       {results.get(f'ndcg@{k}', 0):.4f}")
        print(f"   Hit Rate@{k}:   {results.get(f'hit_rate@{k}', 0):.4f}")
        print(f"   MAP@{k}:        {results.get(f'map@{k}', 0):.4f}")
        print(f"   MRR@{k}:        {results.get(f'mrr@{k}', 0):.4f}")
    
    print(f"\n📦 Coverage: {results['coverage']:.4f}")
    print(f"   ({len(all_recommended_items):,} / {total_items:,} items recommended)")
    
    print(f"\n✅ Evaluated {results['n_users_evaluated']:,} users successfully")
    if failed_users > 0:
        print(f"⚠️  Failed for {failed_users:,} users")
    
    print("=" * 70)
    
    return results


def cross_validate_model(
    model_class,
    df: pd.DataFrame,
    interaction_matrix: csr_matrix,
    n_folds: int = 5,
    k: int = 10,
    model_params: Optional[Dict] = None,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    K-Fold Cross Validation for recommendation models
    
    Args:
        model_class: Model class to instantiate (e.g., CollaborativeFilteringModel)
        df: Full DataFrame with user_idx, item_idx
        interaction_matrix: Full interaction matrix
        n_folds: Number of folds
        k: K value for metrics
        model_params: Parameters to pass to model constructor
        random_state: Random seed
        
    Returns:
        Dictionary with lists of metric values for each fold
    """
    print("=" * 70)
    print(f"🔄 {n_folds}-Fold Cross Validation")
    print("=" * 70)
    
    model_params = model_params or {}
    np.random.seed(random_state)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    fold_size = len(df) // n_folds
    
    metrics_per_fold = defaultdict(list)
    
    for fold in range(n_folds):
        print(f"\n📂 Fold {fold + 1}/{n_folds}")
        print("-" * 70)
        
        # Split data
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(df)
        
        test_df = df.iloc[test_start:test_end]
        train_df = pd.concat([df.iloc[:test_start], df.iloc[test_end:]])
        
        print(f"   Train: {len(train_df):,} interactions")
        print(f"   Test:  {len(test_df):,} interactions")
        
        # Create train matrix
        from scipy.sparse import csr_matrix
        train_matrix = csr_matrix(
            (train_df['implicit'].values, (train_df['user_idx'].values, train_df['item_idx'].values)),
            shape=interaction_matrix.shape
        )
        
        # Train model
        model = model_class(**model_params)
        model.fit(train_matrix, show_progress=False)
        
        # Evaluate
        results = evaluate_model(model, test_df, train_matrix, k_values=[k], verbose=False)
        
        # Store results
        for metric_name, value in results.items():
            if metric_name not in ['n_users_evaluated', 'n_users_failed']:
                metrics_per_fold[metric_name].append(value)
        
        print(f"   Precision@{k}: {results[f'precision@{k}']:.4f}")
        print(f"   Recall@{k}:    {results[f'recall@{k}']:.4f}")
        print(f"   NDCG@{k}:      {results[f'ndcg@{k}']:.4f}")
    
    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("📊 Cross-Validation Summary")
    print("=" * 70)
    
    summary = {}
    for metric_name, values in metrics_per_fold.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[f'{metric_name}_mean'] = mean_val
        summary[f'{metric_name}_std'] = std_val
        
        print(f"{metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("=" * 70)
    
    return dict(metrics_per_fold)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Recommender Metrics - Example Usage")
    print("=" * 70)
    
    # Example data
    recommended = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    relevant = [2, 4, 5, 11, 12]
    
    print("\n📝 Example:")
    print(f"   Recommended items: {recommended}")
    print(f"   Relevant items:    {relevant}")
    print()
    
    # Calculate metrics
    k = 10
    
    precision = RecommenderMetrics.precision_at_k(recommended, relevant, k)
    recall = RecommenderMetrics.recall_at_k(recommended, relevant, k)
    ndcg = RecommenderMetrics.ndcg_at_k(recommended, relevant, k)
    hit_rate = RecommenderMetrics.hit_rate_at_k(recommended, relevant, k)
    map_score = RecommenderMetrics.average_precision_at_k(recommended, relevant, k)
    mrr = RecommenderMetrics.mrr_at_k(recommended, relevant, k)
    
    print(f"📊 Metrics @ K={k}:")
    print(f"   Precision@{k}:  {precision:.4f}")
    print(f"   Recall@{k}:     {recall:.4f}")
    print(f"   NDCG@{k}:       {ndcg:.4f}")
    print(f"   Hit Rate@{k}:   {hit_rate:.4f}")
    print(f"   MAP@{k}:        {map_score:.4f}")
    print(f"   MRR@{k}:        {mrr:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Metrics module ready!")
    print("=" * 70)
    
    print("\n📖 Usage in evaluation:")
    print("""
from src.evaluation.metrics import evaluate_model
from src.models.collaborative import CollaborativeFilteringModel

# Train model
model = CollaborativeFilteringModel()
model.fit(train_matrix)

# Evaluate
results = evaluate_model(
    model=model,
    test_df=test_df,
    train_matrix=train_matrix,
    k_values=[5, 10, 20]
)

print(f"Precision@10: {results['precision@10']:.4f}")
print(f"NDCG@10: {results['ndcg@10']:.4f}")
    """)