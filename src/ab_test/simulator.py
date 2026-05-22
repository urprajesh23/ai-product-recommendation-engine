"""
A/B Test Simulator for Recommendation Systems

Simulates user sessions and compares different recommendation strategies:
- Click-Through Rate (CTR)
- Conversion Rate
- Revenue per session
- User engagement metrics

Usage:
    from src.ab_test import ABTestSimulator
    
    simulator = ABTestSimulator(model_a, model_b, traffic_split=0.5)
    results = simulator.run_test(test_users, test_df, n_sessions=10000)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ABTestMetrics:
    """Container for A/B test metrics"""
    variant: str
    n_sessions: int
    ctr: float
    conversion_rate: float
    avg_revenue: float
    total_revenue: float
    avg_items_clicked: float
    avg_session_duration: float
    
    def __str__(self):
        return f"""
        Variant: {self.variant}
        Sessions: {self.n_sessions}
        CTR: {self.ctr:.4f}
        Conversion Rate: {self.conversion_rate:.4f}
        Avg Revenue: ${self.avg_revenue:.2f}
        Total Revenue: ${self.total_revenue:.2f}
        Avg Items Clicked: {self.avg_items_clicked:.2f}
        """


class ABTestSimulator:
    """
    A/B Test Simulator for Recommendation Systems
    
    Simulates user sessions to compare two recommendation models/strategies.
    Tracks CTR, conversion rate, revenue, and other engagement metrics.
    """
    
    def __init__(
        self,
        model_a,
        model_b,
        traffic_split: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize A/B test simulator
        
        Args:
            model_a: Control model (baseline)
            model_b: Treatment model (new variant)
            traffic_split: Proportion of traffic to model_b (0-1)
            random_state: Random seed for reproducibility
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Simulation parameters
        self.base_ctr = 0.15  # Base click-through rate
        self.base_conversion_rate = 0.05  # Base conversion rate
        self.avg_item_price = 25.0  # Average item price
        self.price_std = 15.0  # Price standard deviation
        
    def assign_variant(self) -> str:
        """
        Assign user to variant (A or B)
        
        Returns:
            'A' or 'B'
        """
        return 'B' if np.random.random() < self.traffic_split else 'A'
    
    def simulate_user_session(
        self,
        user_idx: int,
        relevant_items: List[int],
        train_matrix,
        n_recommendations: int = 10
    ) -> Dict[str, Any]:
        """
        Simulate a single user session
        
        Args:
            user_idx: User index
            relevant_items: Ground truth items user is interested in
            train_matrix: Training interaction matrix
            n_recommendations: Number of recommendations to show
            
        Returns:
            Dictionary with session metrics
        """
        # Assign to variant
        variant = self.assign_variant()
        model = self.model_b if variant == 'B' else self.model_a
        
        # Get recommendations
        try:
            is_new_user = user_idx >= train_matrix.shape[0]
            
            if hasattr(model, 'recommend'):
                recs = model.recommend(
                    user_idx=user_idx,
                    train_matrix=train_matrix,
                    N=n_recommendations,
                    is_new_user=is_new_user,
                    filter_already_liked=True
                )
            else:
                # Fallback for models without recommend method
                recs = []
            
            recommended_items = [item for item, score in recs]
            
        except Exception as e:
            # Fallback to empty recommendations
            recommended_items = []
        
        if len(recommended_items) == 0:
            # No recommendations - return zero metrics
            return {
                'variant': variant,
                'n_recommendations': 0,
                'n_clicks': 0,
                'n_conversions': 0,
                'revenue': 0.0,
                'ctr': 0.0,
                'conversion_rate': 0.0,
                'session_duration': 0.0
            }
        
        # Calculate hits (relevant items in recommendations)
        relevant_set = set(relevant_items)
        hits = set(recommended_items) & relevant_set
        n_hits = len(hits)
        
        # Simulate clicks (clicks depend on relevance)
        # Higher probability of clicking relevant items
        clicks = []
        for item in recommended_items:
            if item in relevant_set:
                # Relevant item - higher click probability
                click_prob = self.base_ctr * 3.0  # 3x more likely to click
            else:
                # Non-relevant item - base click probability
                click_prob = self.base_ctr
            
            if np.random.random() < click_prob:
                clicks.append(item)
        
        n_clicks = len(clicks)
        
        # Simulate conversions (purchases)
        # Only clicked items can be converted
        conversions = []
        for item in clicks:
            if item in relevant_set:
                # Relevant item - higher conversion probability
                conv_prob = self.base_conversion_rate * 4.0
            else:
                # Non-relevant item - base conversion probability
                conv_prob = self.base_conversion_rate
            
            if np.random.random() < conv_prob:
                conversions.append(item)
        
        n_conversions = len(conversions)
        
        # Calculate revenue
        # Each conversion generates revenue based on item price
        revenue = 0.0
        for _ in conversions:
            # Sample item price from normal distribution
            price = max(1.0, np.random.normal(self.avg_item_price, self.price_std))
            revenue += price
        
        # Simulate session duration (in seconds)
        # More clicks = longer session
        base_duration = 30  # seconds
        duration_per_click = 15  # seconds per click
        session_duration = base_duration + (n_clicks * duration_per_click)
        session_duration += np.random.normal(0, 10)  # Add noise
        session_duration = max(5, session_duration)  # Minimum 5 seconds
        
        # Calculate metrics
        ctr = n_clicks / n_recommendations if n_recommendations > 0 else 0.0
        conversion_rate = n_conversions / n_clicks if n_clicks > 0 else 0.0
        
        return {
            'variant': variant,
            'user_idx': user_idx,
            'n_recommendations': n_recommendations,
            'n_relevant_shown': n_hits,
            'n_clicks': n_clicks,
            'n_conversions': n_conversions,
            'revenue': revenue,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'session_duration': session_duration,
            'recommended_items': recommended_items,
            'clicked_items': clicks,
            'converted_items': conversions
        }
    
    def run_test(
        self,
        test_users: List[int],
        test_df: pd.DataFrame,
        train_matrix,
        n_sessions: int = 1000,
        n_recommendations: int = 10,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, ABTestMetrics]]:
        """
        Run A/B test simulation
        
        Args:
            test_users: List of test user indices
            test_df: Test DataFrame with user_idx, item_idx columns
            train_matrix: Training interaction matrix
            n_sessions: Number of sessions to simulate
            n_recommendations: Number of recommendations per session
            verbose: Whether to print progress
            
        Returns:
            Tuple of (results_df, summary_metrics)
        """
        if verbose:
            print("=" * 70)
            print("🧪 Running A/B Test Simulation")
            print("=" * 70)
            print(f"   Model A (Control): {type(self.model_a).__name__}")
            print(f"   Model B (Treatment): {type(self.model_b).__name__}")
            print(f"   Traffic Split: {(1-self.traffic_split)*100:.0f}% A / {self.traffic_split*100:.0f}% B")
            print(f"   Sessions: {n_sessions:,}")
            print(f"   Recommendations per session: {n_recommendations}")
            print()
        
        # Group test data by user
        user_test_items = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        
        # Run simulations
        results = []
        
        for i in range(n_sessions):
            if verbose and (i + 1) % 100 == 0:
                print(f"   Progress: {i + 1}/{n_sessions} sessions...", end='\r')
            
            # Sample a random user
            user_idx = np.random.choice(test_users)
            
            # Get ground truth items
            relevant_items = user_test_items.get(user_idx, [])
            
            if len(relevant_items) == 0:
                continue  # Skip users with no test interactions
            
            # Simulate session
            session_result = self.simulate_user_session(
                user_idx=user_idx,
                relevant_items=relevant_items,
                train_matrix=train_matrix,
                n_recommendations=n_recommendations
            )
            
            results.append(session_result)
        
        if verbose:
            print()  # New line after progress
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary metrics by variant
        summary = {}
        
        for variant in ['A', 'B']:
            variant_data = results_df[results_df['variant'] == variant]
            
            if len(variant_data) == 0:
                continue
            
            metrics = ABTestMetrics(
                variant=variant,
                n_sessions=len(variant_data),
                ctr=variant_data['ctr'].mean(),
                conversion_rate=variant_data['conversion_rate'].mean(),
                avg_revenue=variant_data['revenue'].mean(),
                total_revenue=variant_data['revenue'].sum(),
                avg_items_clicked=variant_data['n_clicks'].mean(),
                avg_session_duration=variant_data['session_duration'].mean()
            )
            
            summary[variant] = metrics
        
        # Print summary
        if verbose:
            self._print_summary(summary, results_df)
        
        return results_df, summary
    
    def _print_summary(self, summary: Dict[str, ABTestMetrics], results_df: pd.DataFrame):
        """Print A/B test summary"""
        print()
        print("=" * 70)
        print("📊 A/B Test Results")
        print("=" * 70)
        
        for variant in ['A', 'B']:
            if variant in summary:
                print(f"\n{'🅰️  VARIANT A (Control)' if variant == 'A' else '🅱️  VARIANT B (Treatment)'}")
                print("-" * 70)
                metrics = summary[variant]
                print(f"   Sessions: {metrics.n_sessions:,}")
                print(f"   CTR: {metrics.ctr:.4f} ({metrics.ctr*100:.2f}%)")
                print(f"   Conversion Rate: {metrics.conversion_rate:.4f} ({metrics.conversion_rate*100:.2f}%)")
                print(f"   Avg Revenue/Session: ${metrics.avg_revenue:.2f}")
                print(f"   Total Revenue: ${metrics.total_revenue:.2f}")
                print(f"   Avg Items Clicked: {metrics.avg_items_clicked:.2f}")
                print(f"   Avg Session Duration: {metrics.avg_session_duration:.1f}s")
        
        # Calculate improvements
        if 'A' in summary and 'B' in summary:
            print("\n" + "=" * 70)
            print("📈 Treatment vs Control")
            print("=" * 70)
            
            ctr_lift = ((summary['B'].ctr - summary['A'].ctr) / summary['A'].ctr) * 100 if summary['A'].ctr > 0 else 0
            conv_lift = ((summary['B'].conversion_rate - summary['A'].conversion_rate) / summary['A'].conversion_rate) * 100 if summary['A'].conversion_rate > 0 else 0
            rev_lift = ((summary['B'].avg_revenue - summary['A'].avg_revenue) / summary['A'].avg_revenue) * 100 if summary['A'].avg_revenue > 0 else 0
            
            print(f"   CTR Lift: {ctr_lift:+.2f}%")
            print(f"   Conversion Rate Lift: {conv_lift:+.2f}%")
            print(f"   Revenue Lift: {rev_lift:+.2f}%")
            
            # Statistical significance test
            print("\n📊 Statistical Significance (t-test):")
            
            # CTR t-test
            ctr_a = results_df[results_df['variant'] == 'A']['ctr']
            ctr_b = results_df[results_df['variant'] == 'B']['ctr']
            t_stat, p_value = stats.ttest_ind(ctr_a, ctr_b)
            
            print(f"   CTR p-value: {p_value:.4f} {'✅ Significant' if p_value < 0.05 else '❌ Not significant'}")
            
            # Revenue t-test
            rev_a = results_df[results_df['variant'] == 'A']['revenue']
            rev_b = results_df[results_df['variant'] == 'B']['revenue']
            t_stat, p_value = stats.ttest_ind(rev_a, rev_b)
            
            print(f"   Revenue p-value: {p_value:.4f} {'✅ Significant' if p_value < 0.05 else '❌ Not significant'}")
        
        print("=" * 70)
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot A/B test results
        
        Args:
            results_df: Results DataFrame from run_test
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('A/B Test Results Comparison', fontsize=16, fontweight='bold')
        
        # CTR comparison
        axes[0, 0].set_title('Click-Through Rate (CTR)')
        results_df.boxplot(column='ctr', by='variant', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Variant')
        axes[0, 0].set_ylabel('CTR')
        
        # Conversion rate comparison
        axes[0, 1].set_title('Conversion Rate')
        results_df.boxplot(column='conversion_rate', by='variant', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Variant')
        axes[0, 1].set_ylabel('Conversion Rate')
        
        # Revenue comparison
        axes[0, 2].set_title('Revenue per Session')
        results_df.boxplot(column='revenue', by='variant', ax=axes[0, 2])
        axes[0, 2].set_xlabel('Variant')
        axes[0, 2].set_ylabel('Revenue ($)')
        
        # Number of clicks
        axes[1, 0].set_title('Clicks per Session')
        results_df.boxplot(column='n_clicks', by='variant', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Variant')
        axes[1, 0].set_ylabel('Number of Clicks')
        
        # Session duration
        axes[1, 1].set_title('Session Duration')
        results_df.boxplot(column='session_duration', by='variant', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Variant')
        axes[1, 1].set_ylabel('Duration (seconds)')
        
        # Revenue distribution
        axes[1, 2].set_title('Revenue Distribution')
        for variant in ['A', 'B']:
            variant_data = results_df[results_df['variant'] == variant]['revenue']
            axes[1, 2].hist(variant_data, alpha=0.5, label=f'Variant {variant}', bins=30)
        axes[1, 2].set_xlabel('Revenue ($)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()


def simulate_user_session(
    user_idx: int,
    model,
    relevant_items: List[int],
    train_matrix,
    n_recommendations: int = 10
) -> Dict[str, Any]:
    """
    Standalone function to simulate a single user session
    
    Args:
        user_idx: User index
        model: Recommendation model
        relevant_items: Ground truth relevant items
        train_matrix: Training interaction matrix
        n_recommendations: Number of recommendations
        
    Returns:
        Session metrics dictionary
    """
    simulator = ABTestSimulator(model, model, traffic_split=0.0)
    return simulator.simulate_user_session(
        user_idx, relevant_items, train_matrix, n_recommendations
    )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("A/B Test Simulator - Example Usage")
    print("=" * 70)
    
    print("""
    📖 Example: Running an A/B test
    
    from src.ab_test import ABTestSimulator
    from src.models.collaborative import CollaborativeFilteringModel
    from src.models.hybrid import HybridRecommender
    
    # Load models
    model_a = CollaborativeFilteringModel.load('models/collaborative_model.pkl')
    model_b = HybridRecommender.load('models/hybrid_model.pkl')
    
    # Initialize simulator
    simulator = ABTestSimulator(
        model_a=model_a,  # Control
        model_b=model_b,  # Treatment
        traffic_split=0.5  # 50/50 split
    )
    
    # Run test
    results_df, summary = simulator.run_test(
        test_users=test_users,
        test_df=test_df,
        train_matrix=train_matrix,
        n_sessions=10000,
        n_recommendations=10
    )
    
    # Plot results
    simulator.plot_results(results_df, save_path='ab_test_results.png')
    
    # Access metrics
    print(f"Variant A CTR: {summary['A'].ctr:.4f}")
    print(f"Variant B CTR: {summary['B'].ctr:.4f}")
    print(f"Revenue Lift: {((summary['B'].avg_revenue - summary['A'].avg_revenue) / summary['A'].avg_revenue) * 100:.2f}%")
    """)
    
    print("\n" + "=" * 70)
    print("✅ A/B Test Simulator ready!")
    print("=" * 70)