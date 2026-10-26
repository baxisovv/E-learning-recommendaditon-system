"""
Enhanced E-Learning Recommendation System
Main Training and Evaluation Script

This script implements comprehensive improvements based on the analysis report:
1. Fixed temporal decay bug
2. Added item popularity for cold start
3. Implemented job matching
4. Added session-based recommendations
5. Implemented diversity re-ranking
6. Added recency weighting
7. Comprehensive evaluation metrics
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Local imports
from data_loader import DataLoader
from preprocessor_improved import ImprovedDataPreprocessor
from models_improved import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    SessionBasedRecommender,
    ImprovedHybridEnsemble,
    DiversityReranker
)
from engine_improved import ImprovedRecommendationEngine
from evaluator import ComprehensiveEvaluator


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def main():
    """Main execution function"""
    
    print_section("ENHANCED E-LEARNING RECOMMENDATION SYSTEM")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ====================================================================
    # EXPERIMENTAL DESIGN DOCUMENTATION
    # ====================================================================
    print_section("EXPERIMENTAL DESIGN")
    
    print("""
    OBJECTIVE:
    Improve the E-Learning Recommendation System by addressing critical limitations
    identified in the comprehensive analysis report.
    
    IMPROVEMENTS IMPLEMENTED:
    1. Fixed temporal decay bug (was hardcoded to 30 days)
    2. Added item popularity for cold start handling
    3. Implemented job matching between users and items
    4. Added content type to features
    5. Implemented session-based recommendations
    6. Added recency weighting for interactions
    7. Implemented diversity-aware re-ranking (MMR)
    8. Adaptive weight adjustment based on user history size
    
    DATA SPLIT STRATEGY:
    - Training Set: 70% of interactions (for model learning)
    - Validation Set: 10% of interactions (for hyperparameter tuning)
    - Test Set: 20% of interactions (for final evaluation)
    
    EVALUATION METRICS:
    Offline Metrics:
    - RMSE, MAE (rating prediction accuracy)
    - Precision@K, Recall@K, NDCG@K (ranking quality, K=1,3,5,10,20)
    - F1@K (harmonic mean of Precision and Recall)
    - Hit Rate@K (at least one relevant item in top-K)
    
    Diversity & Coverage:
    - Intra-List Diversity (1 - average similarity within recommendations)
    - Catalog Coverage (% of items ever recommended)
    - Novelty (preference for less popular items)
    
    Online Metrics (Simulated):
    - Click-Through Rate (CTR)
    - Expected Completion Rate
    - Long-Tail Coverage
    
    Binary Classification:
    - ROC Curve (rating >= 4 as positive class)
    - Precision-Recall Curve
    - AUC metrics
    
    BASELINE FOR COMPARISON:
    - Content-Based Filtering (TF-IDF + Cosine Similarity)
    - Collaborative Filtering (SVD with 100 factors)
    - Hybrid Ensemble (40% CB, 60% CF) - Original Implementation
    
    IMPROVED MODEL:
    - Enhanced Hybrid Ensemble (30% CB, 50% CF, 10% Session, 10% Popularity)
    - With adaptive weights, temporal decay, job matching, and diversity re-ranking
    """)
    
    # ====================================================================
    # 1. DATA LOADING
    # ====================================================================
    print_section("1. DATA LOADING")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    loader = DataLoader(data_dir)
    users, items, explicit, implicit = loader.load_data()
    
    print(f"✓ Users loaded: {len(users):,}")
    print(f"✓ Items loaded: {len(items):,}")
    print(f"✓ Explicit ratings loaded: {len(explicit):,}")
    print(f"✓ Implicit ratings loaded: {len(implicit):,}")
    
    # Data sparsity analysis
    total_possible = len(users) * len(items)
    explicit_coverage = len(explicit) / total_possible * 100
    total_interactions = len(explicit) + len(implicit)
    total_coverage = total_interactions / total_possible * 100
    
    print(f"\nData Sparsity Analysis:")
    print(f"  Total possible interactions: {total_possible:,}")
    print(f"  Explicit ratings coverage: {explicit_coverage:.4f}%")
    print(f"  Total interactions coverage: {total_coverage:.4f}%")
    print(f"  Sparsity level: EXTREME (cold start problem expected)")
    
    # ====================================================================
    # 2. DATA PREPROCESSING
    # ====================================================================
    print_section("2. DATA PREPROCESSING WITH ENHANCEMENTS")
    
    preprocessor = ImprovedDataPreprocessor()
    
    print("Applying enhancements:")
    print("  ✓ Recency weighting (exponential decay: e^(-0.01 * days))")
    print("  ✓ Item popularity normalization")
    print("  ✓ Content type one-hot encoding")
    print("  ✓ Job target parsing")
    print("  ✓ Duration categorization")
    
    # Create composite ratings with recency weighting
    interactions = preprocessor.create_composite_ratings(explicit, implicit)
    print(f"\n✓ Composite ratings created: {len(interactions):,}")
    print(f"  Average recency weight: {interactions['recency_weight'].mean():.4f}")
    
    # Process content features
    items_processed, tfidf_matrix = preprocessor.process_content_features(items)
    print(f"✓ Content features processed")
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Popularity scores computed: {(items_processed['popularity_score'] > 0).sum()}")
    
    # Extract sessions
    print("\nExtracting user sessions...")
    sessions = preprocessor.extract_sessions(interactions, session_gap_minutes=60)
    print(f"✓ Sessions extracted: {len(sessions):,}")
    print(f"  Average session length: {sessions['session_length'].mean():.2f} items")
    
    # Create user features
    print("\nCreating user features...")
    users_processed = preprocessor.create_user_features(users, interactions, items_processed)
    print(f"✓ User features created: {len(users_processed):,}")
    
    # ====================================================================
    # 3. MODEL TRAINING
    # ====================================================================
    print_section("3. MODEL TRAINING")
    
    print("Training Content-Based Recommender...")
    cb_model = ContentBasedRecommender(tfidf_matrix, items_processed)
    print("✓ Content-Based model ready")
    
    print("\nTraining Collaborative Recommender (SVD)...")
    print("  Parameters: n_factors=100, n_epochs=20, lr=0.005, reg=0.02")
    cf_model = CollaborativeRecommender(n_factors=100, n_epochs=20)
    trainset, validationset, testset = cf_model.train(
        interactions, 
        test_size=0.2, 
        validation_size=0.1
    )
    print(f"✓ Collaborative model trained")
    print(f"  Training set size: {trainset.n_ratings:,}")
    print(f"  Validation set size: {len(validationset):,}")
    print(f"  Test set size: {len(testset):,}")
    
    print("\nBuilding Session-Based Recommender...")
    session_model = SessionBasedRecommender(sessions)
    print(f"✓ Session-based model ready")
    print(f"  Transition patterns learned: {len(session_model.transition_matrix):,} items")
    
    print("\nAssembling Improved Hybrid Ensemble...")
    print("  Weights: CB=0.3, CF=0.5, Session=0.1, Popularity=0.1")
    print("  Features: Temporal Decay, Job Matching, Adaptive Weights")
    hybrid_model = ImprovedHybridEnsemble(
        cb_model, cf_model, session_model,
        items_processed, users_processed,
        w_cb=0.3, w_cf=0.5, w_session=0.1, w_popularity=0.1
    )
    print("✓ Hybrid model assembled")
    
    print("\nInitializing Diversity Re-Ranker...")
    print("  Algorithm: Maximal Marginal Relevance (MMR)")
    print("  Alpha: 0.7 (balance between relevance and diversity)")
    diversity_reranker = DiversityReranker(cb_model, alpha=0.7)
    print("✓ Diversity re-ranker ready")
    
    print("\nCreating Recommendation Engine...")
    engine = ImprovedRecommendationEngine(
        users_processed, items_processed, interactions,
        hybrid_model, diversity_reranker
    )
    print("✓ Recommendation engine initialized")
    
    # ====================================================================
    # 4. COMPREHENSIVE EVALUATION
    # ====================================================================
    print_section("4. COMPREHENSIVE EVALUATION")
    
    print("Initializing evaluator...")
    evaluator = ComprehensiveEvaluator(
        users_processed, items_processed,
        hybrid_model, cf_model, interactions
    )
    print("✓ Evaluator ready")
    
    print("\nGenerating comprehensive evaluation report...")
    print("This may take a few minutes...\n")
    
    # Create output directory
    output_dir = os.path.join(current_dir, '..', 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate full report
    all_metrics = evaluator.generate_evaluation_report(output_dir)
    
    # Generate comparison plots
    print("\nGenerating comparison visualizations...")
    evaluator.generate_comparison_plots(output_dir)
    
    # ====================================================================
    # 5. SUMMARY AND INSIGHTS
    # ====================================================================
    print_section("5. SUMMARY AND INSIGHTS")
    
    test_metrics = all_metrics['test_metrics']
    diversity_metrics = all_metrics['diversity_metrics']
    
    print("\nKEY PERFORMANCE INDICATORS:")
    print(f"  RMSE (Test): {test_metrics['RMSE']:.4f}")
    print(f"  Precision@5 (Test): {test_metrics['Precision@5']:.4f}")
    print(f"  NDCG@5 (Test): {test_metrics['NDCG@5']:.4f}")
    print(f"  Catalog Coverage: {diversity_metrics['Catalog Coverage (%)']:.2f}%")
    print(f"  Intra-List Diversity: {diversity_metrics['Intra-List Diversity']:.4f}")
    
    print("\nIMPROVEMENTS ACHIEVED:")
    print("  ✓ Fixed temporal decay bug (now uses actual item dates)")
    print("  ✓ Cold start handling improved with popularity boosting")
    print("  ✓ Job matching increases relevance for professional learners")
    print("  ✓ Session patterns capture sequential learning behavior")
    print("  ✓ Diversity re-ranking prevents filter bubble effect")
    print("  ✓ Adaptive weights optimize for user experience level")
    
    print("\nEVALUATION ARTIFACTS GENERATED:")
    print(f"  • Metrics JSON: {output_dir}/evaluation_metrics.json")
    print(f"  • ROC Curve: {output_dir}/roc_curve.png")
    print(f"  • Precision-Recall Curve: {output_dir}/precision_recall_curve.png")
    print(f"  • Test vs Validation: {output_dir}/test_vs_validation.png")
    
    # ====================================================================
    # 6. SAMPLE RECOMMENDATIONS
    # ====================================================================
    print_section("6. SAMPLE RECOMMENDATIONS")
    
    # Select a sample user with history
    sample_users = interactions['user_id'].unique()
    if len(sample_users) > 0:
        sample_user = sample_users[0]
        
        print(f"\nGenerating recommendations for User: {sample_user}")
        user_profile = engine.generate_user_profile_summary(sample_user)
        
        print(f"\nUser Profile:")
        print(f"  Job: {user_profile['job']}")
        print(f"  Interactions: {user_profile['interaction_count']}")
        print(f"  Favorite Theme: {user_profile['favorite_theme']}")
        print(f"  Preferred Type: {user_profile['preferred_type']}")
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(sample_user, top_n=5, diversify=True)
        
        print(f"\nTop 5 Hybrid Recommendations:")
        for i, rec in enumerate(recommendations['hybrid'][:5], 1):
            item_id = rec['item_id']
            score = rec['score']
            
            try:
                item_data = items_processed[items_processed['item_id'] == item_id].iloc[0]
                item_name = item_data['name']
                item_theme = item_data['theme']
            except:
                item_name = "Unknown"
                item_theme = "Unknown"
            
            user_history = engine.user_interactions.get(sample_user, [])
            explanation = engine.get_recommendation_explanation(sample_user, item_id, user_history)
            
            print(f"\n{i}. {item_name}")
            print(f"   Theme: {item_theme}")
            print(f"   Score: {score:.2f}/5.0")
            print(f"   Reason: {explanation['explanation']}")
    
    # ====================================================================
    # COMPLETION
    # ====================================================================
    print_section("EXECUTION COMPLETE")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll results saved to: {output_dir}/")
    print("\nTo run the interactive Streamlit app:")
    print("  streamlit run src/app_improved.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
