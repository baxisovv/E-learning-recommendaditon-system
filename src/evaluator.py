import numpy as np
import pandas as pd
from surprise import accuracy
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation system with:
    - Train/Test/Validation metrics
    - Multiple evaluation metrics (RMSE, MAE, Precision@K, Recall@K, NDCG@K, etc.)
    - ROC curves
    - Precision-Recall curves
    - Catalog coverage
    - Diversity metrics
    - Online metrics simulation
    """
    
    def __init__(self, users_df, items_df, hybrid_model, cf_model, interactions_df):
        self.users = users_df.set_index('user_id')
        self.items = items_df.set_index('item_id')
        self.hybrid = hybrid_model
        self.cf_model = cf_model
        self.interactions = interactions_df
        
        # For catalog coverage tracking
        self.all_item_ids = set(items_df['item_id'].values)
        self.recommended_items = set()
        
    def evaluate_offline_metrics(self, dataset_name='test'):
        """
        Evaluate comprehensive offline metrics
        
        Args:
            dataset_name: 'test', 'validation', or 'train'
            
        Returns:
            Dictionary of metrics
        """
        # Select appropriate dataset
        if dataset_name == 'test':
            testset = self.cf_model.testset
        elif dataset_name == 'validation':
            testset = self.cf_model.validationset
        else:
            testset = self.cf_model.trainset.build_testset()
        
        # Get predictions
        predictions = self.cf_model.algo.test(testset)
        
        # 1. RMSE and MAE
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        # 2. Prepare for ranking metrics
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
        
        # 3. Calculate Precision@K, Recall@K, NDCG@K for multiple K values
        k_values = [1, 3, 5, 10, 20]
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        ndcg_at_k = {k: [] for k in k_values}
        hit_rate_at_k = {k: 0 for k in k_values}
        
        relevance_threshold = 3.5  # Items with rating >= 3.5 considered relevant
        
        for uid, user_ratings in user_est_true.items():
            # Sort by estimated rating
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # Identify relevant items (ground truth)
            relevant_items = set([i for i, (_, true_r) in enumerate(user_ratings) if true_r >= relevance_threshold])
            n_relevant = len(relevant_items)
            
            if n_relevant == 0:
                continue
            
            for k in k_values:
                top_k = user_ratings[:k]
                
                # Precision@K
                recommended_relevant = sum(1 for i, (_, true_r) in enumerate(top_k) if true_r >= relevance_threshold)
                precision_at_k[k].append(recommended_relevant / k)
                
                # Recall@K
                recall_at_k[k].append(recommended_relevant / n_relevant if n_relevant > 0 else 0)
                
                # Hit Rate@K (at least one relevant item in top-K)
                if recommended_relevant > 0:
                    hit_rate_at_k[k] += 1
                
                # NDCG@K
                dcg = sum((true_r / np.log2(idx + 2)) for idx, (_, true_r) in enumerate(top_k))
                ideal_ratings = sorted([true_r for (_, true_r) in user_ratings], reverse=True)[:k]
                idcg = sum((true_r / np.log2(idx + 2)) for idx, true_r in enumerate(ideal_ratings))
                ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0)
        
        # Aggregate metrics
        metrics = {
            'RMSE': float(rmse),
            'MAE': float(mae),
        }
        
        for k in k_values:
            metrics[f'Precision@{k}'] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0
            metrics[f'Recall@{k}'] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0
            metrics[f'NDCG@{k}'] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0
            metrics[f'HitRate@{k}'] = float(hit_rate_at_k[k] / len(user_est_true)) if user_est_true else 0
        
        # F1 Score at K
        for k in k_values:
            p = metrics[f'Precision@{k}']
            r = metrics[f'Recall@{k}']
            metrics[f'F1@{k}'] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0
        
        return metrics
    
    def calculate_diversity_metrics(self, user_sample_size=100, k=10):
        """
        Calculate diversity metrics:
        - Intra-list diversity (diversity within each user's recommendations)
        - Coverage (% of catalog recommended)
        - Novelty (average popularity of recommended items)
        """
        # Sample users for efficiency
        sample_users = self.users.index[:user_sample_size].tolist()
        
        diversity_scores = []
        novelty_scores = []
        recommended_items_all = set()
        
        for user_id in sample_users:
            # Get user history
            user_history = self.interactions[
                self.interactions['user_id'] == user_id
            ]['item_id'].tolist()
            
            # Get recommendations
            scores = []
            for item_id in self.all_item_ids:
                if item_id not in user_history:
                    h_score, _ = self.hybrid.predict(user_id, item_id, user_history)
                    scores.append((item_id, h_score))
            
            # Top K recommendations
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_items = [item_id for item_id, _ in scores[:k]]
            recommended_items_all.update(top_k_items)
            
            # Calculate intra-list diversity
            if len(top_k_items) > 1:
                sim_sum = 0
                pairs = 0
                for i in range(len(top_k_items)):
                    for j in range(i + 1, len(top_k_items)):
                        try:
                            sim = self.hybrid.cb.get_similarity_score(top_k_items[i], top_k_items[j])
                            sim_sum += sim
                            pairs += 1
                        except:
                            pass
                
                if pairs > 0:
                    diversity_scores.append(1 - (sim_sum / pairs))
            
            # Calculate novelty (inverse popularity)
            if 'nb_views' in self.items.columns:
                item_popularities = []
                for item_id in top_k_items:
                    try:
                        views = self.items.loc[item_id, 'nb_views']
                        item_popularities.append(views)
                    except:
                        pass
                
                if item_popularities:
                    # Novelty: items with lower views are more novel
                    max_views = self.items['nb_views'].max()
                    novelty = 1 - (np.mean(item_popularities) / max_views)
                    novelty_scores.append(novelty)
        
        # Catalog coverage
        coverage = len(recommended_items_all) / len(self.all_item_ids) * 100
        
        return {
            'Intra-List Diversity': float(np.mean(diversity_scores)) if diversity_scores else 0,
            'Catalog Coverage (%)': float(coverage),
            'Novelty': float(np.mean(novelty_scores)) if novelty_scores else 0,
            'Unique Items Recommended': len(recommended_items_all)
        }
    
    def generate_roc_curves(self, output_dir='evaluation_results'):
        """
        Generate ROC curves for rating prediction
        Binary classification: rating >= 4 vs. rating < 4
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = self.cf_model.algo.test(self.cf_model.testset)
        
        # Extract true and predicted ratings
        y_true = []
        y_pred = []
        
        for uid, iid, true_r, est, _ in predictions:
            # Binary classification: rating >= 4 is "positive"
            y_true.append(1 if true_r >= 4.0 else 0)
            y_pred.append(est)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Rating Prediction (Positive: Rating ≥ 4)', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'ROC_AUC': float(roc_auc)}
    
    def generate_precision_recall_curves(self, output_dir='evaluation_results'):
        """Generate Precision-Recall curves"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = self.cf_model.algo.test(self.cf_model.testset)
        
        # Extract true and predicted ratings
        y_true = []
        y_pred = []
        
        for uid, iid, true_r, est, _ in predictions:
            y_true.append(1 if true_r >= 4.0 else 0)
            y_pred.append(est)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {'Average_Precision': float(avg_precision)}
    
    def simulate_online_metrics(self, sample_size=100, k=10):
        """
        Simulate online metrics:
        - Click-Through Rate (CTR)
        - Expected completion rate
        - Coverage of long-tail items
        """
        # Sample users
        sample_users = self.users.index[:sample_size].tolist()
        
        clicks = 0
        total_recommendations = 0
        expected_completions = 0
        long_tail_recommendations = 0
        
        # Define long-tail items (bottom 50% by popularity)
        if 'nb_views' in self.items.columns:
            median_views = self.items['nb_views'].median()
            long_tail_items = set(
                self.items[self.items['nb_views'] <= median_views].index
            )
        else:
            long_tail_items = set()
        
        for user_id in sample_users:
            user_history = self.interactions[
                self.interactions['user_id'] == user_id
            ]['item_id'].tolist()
            
            # Get recommendations
            scores = []
            for item_id in self.all_item_ids:
                if item_id not in user_history:
                    h_score, _ = self.hybrid.predict(user_id, item_id, user_history)
                    scores.append((item_id, h_score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_items = [item_id for item_id, _ in scores[:k]]
            
            total_recommendations += len(top_k_items)
            
            # Simulate CTR (items with score > 3.5 are likely to be clicked)
            for item_id, score in scores[:k]:
                if score > 3.5:
                    clicks += 1
                
                # Simulate completion (higher score = higher completion probability)
                completion_prob = min(score / 5.0, 1.0)
                expected_completions += completion_prob
                
                # Track long-tail recommendations
                if item_id in long_tail_items:
                    long_tail_recommendations += 1
        
        ctr = clicks / total_recommendations if total_recommendations > 0 else 0
        completion_rate = expected_completions / total_recommendations if total_recommendations > 0 else 0
        long_tail_coverage = long_tail_recommendations / total_recommendations if total_recommendations > 0 else 0
        
        return {
            'Simulated CTR': float(ctr),
            'Expected Completion Rate': float(completion_rate),
            'Long-Tail Coverage': float(long_tail_coverage)
        }
    
    def generate_evaluation_report(self, output_dir='evaluation_results'):
        """
        Generate comprehensive evaluation report with all metrics
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Offline Metrics - Test Set
        print("\n" + "=" * 80)
        print("1. OFFLINE METRICS - TEST SET")
        print("=" * 80)
        test_metrics = self.evaluate_offline_metrics('test')
        for metric, value in test_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        
        # 2. Offline Metrics - Validation Set
        print("\n" + "=" * 80)
        print("2. OFFLINE METRICS - VALIDATION SET")
        print("=" * 80)
        val_metrics = self.evaluate_offline_metrics('validation')
        for metric, value in val_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        
        # 3. Diversity and Coverage Metrics
        print("\n" + "=" * 80)
        print("3. DIVERSITY AND COVERAGE METRICS")
        print("=" * 80)
        diversity_metrics = self.calculate_diversity_metrics()
        for metric, value in diversity_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        
        # 4. ROC Curve Analysis
        print("\n" + "=" * 80)
        print("4. ROC CURVE ANALYSIS")
        print("=" * 80)
        roc_metrics = self.generate_roc_curves(output_dir)
        for metric, value in roc_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        print(f"ROC curve saved to: {output_dir}/roc_curve.png")
        
        # 5. Precision-Recall Analysis
        print("\n" + "=" * 80)
        print("5. PRECISION-RECALL ANALYSIS")
        print("=" * 80)
        pr_metrics = self.generate_precision_recall_curves(output_dir)
        for metric, value in pr_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        print(f"PR curve saved to: {output_dir}/precision_recall_curve.png")
        
        # 6. Simulated Online Metrics
        print("\n" + "=" * 80)
        print("6. SIMULATED ONLINE METRICS")
        print("=" * 80)
        online_metrics = self.simulate_online_metrics()
        for metric, value in online_metrics.items():
            print(f"{metric:.<50} {value:.4f}")
        
        # 7. Save all metrics to JSON
        all_metrics = {
            'test_metrics': test_metrics,
            'validation_metrics': val_metrics,
            'diversity_metrics': diversity_metrics,
            'roc_metrics': roc_metrics,
            'precision_recall_metrics': pr_metrics,
            'online_metrics': online_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"Complete metrics saved to: {output_dir}/evaluation_metrics.json")
        print("=" * 80)
        
        return all_metrics
    
    def generate_comparison_plots(self, output_dir='evaluation_results'):
        """Generate comparison visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Compare test vs validation metrics
        test_metrics = self.evaluate_offline_metrics('test')
        val_metrics = self.evaluate_offline_metrics('validation')
        
        # Extract comparable metrics
        comparable_metrics = ['Precision@5', 'Recall@5', 'NDCG@5', 'F1@5']
        
        test_values = [test_metrics.get(m, 0) for m in comparable_metrics]
        val_values = [val_metrics.get(m, 0) for m in comparable_metrics]
        
        # Plot comparison
        x = np.arange(len(comparable_metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, test_values, width, label='Test Set', color='skyblue')
        bars2 = ax.bar(x + width/2, val_values, width, label='Validation Set', color='lightcoral')
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Test vs. Validation Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(comparable_metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/test_vs_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {output_dir}/test_vs_validation.png")
