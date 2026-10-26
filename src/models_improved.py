import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
from datetime import datetime

class ContentBasedRecommender:
    """Enhanced Content-Based Recommender with metadata awareness"""
    
    def __init__(self, tfidf_matrix, items_df):
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.items_df = items_df.set_index('item_id')
        self.indices = pd.Series(
            range(len(items_df)), 
            index=items_df['item_id']
        )

    def get_similarity_score(self, item_id1, item_id2):
        """Get cosine similarity between two items"""
        try:
            idx1 = self.indices[item_id1]
            idx2 = self.indices[item_id2]
            return self.cosine_sim[idx1][idx2]
        except KeyError:
            return 0.0
    
    def get_similar_items(self, item_id, top_k=10):
        """Find K most similar items"""
        try:
            idx = self.indices[item_id]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_k+1]  # Exclude self
            item_indices = [i[0] for i in sim_scores]
            return self.items_df.iloc[item_indices].index.tolist()
        except KeyError:
            return []


class CollaborativeRecommender:
    """Enhanced Collaborative Filtering with configurable parameters"""
    
    def __init__(self, n_factors=150, n_epochs=50, lr_all=0.005, reg_all=0.02):
        # FIX (Cause 4): increased n_factors 100→150 and n_epochs 20→50 to
        # extract more signal from the 0.22% density matrix.
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.trainset = None
        self.testset = None
        self.validationset = None

    def train(self, interactions_df, test_size=0.2, validation_size=0.1):
        """
        Train with train/validation/test split
        
        Args:
            interactions_df: DataFrame with user_id, item_id, scaled_score
            test_size: Proportion for test set
            validation_size: Proportion for validation set (from total data)
        """
        from sklearn.model_selection import train_test_split as sk_split
        
        # First, split at DataFrame level to create train+val vs test
        train_val_df, test_df = sk_split(
            interactions_df, 
            test_size=test_size, 
            random_state=42
        )
        
        # Second, split train+val into train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        train_df, val_df = sk_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=42
        )
        
        # Create Surprise datasets
        reader = Reader(rating_scale=(1, 5))
        
        # Training set
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'item_id', 'scaled_score']], 
            reader
        )
        self.trainset = train_data.build_full_trainset()
        
        # Validation set (as testset format)
        val_data = Dataset.load_from_df(
            val_df[['user_id', 'item_id', 'scaled_score']], 
            reader
        )
        val_full = val_data.build_full_trainset()
        self.validationset = val_full.build_testset()
        
        # Test set (as testset format)
        test_data = Dataset.load_from_df(
            test_df[['user_id', 'item_id', 'scaled_score']], 
            reader
        )
        test_full = test_data.build_full_trainset()
        self.testset = test_full.build_testset()
        
        # Train on training set
        self.algo.fit(self.trainset)
        
        return self.trainset, self.validationset, self.testset

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        return self.algo.predict(user_id, item_id).est


class SessionBasedRecommender:
    """Session-based recommendations using transition patterns"""
    
    def __init__(self, sessions_df):
        self.sessions = sessions_df
        self.transition_matrix = self._build_transition_matrix()
        self.item_frequency = self._calculate_item_frequency()
    
    def _build_transition_matrix(self):
        """Build item-to-item transition probabilities"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for _, session in self.sessions.iterrows():
            items = session['items']
            for i in range(len(items) - 1):
                transitions[items[i]][items[i+1]] += 1
        
        # Normalize to probabilities
        transition_probs = {}
        for item, next_items in transitions.items():
            total = sum(next_items.values())
            transition_probs[item] = {
                next_item: count / total 
                for next_item, count in next_items.items()
            }
        
        return transition_probs
    
    def _calculate_item_frequency(self):
        """Calculate how often each item appears in sessions"""
        all_items = []
        for _, session in self.sessions.iterrows():
            all_items.extend(session['items'])
        
        from collections import Counter
        return Counter(all_items)
    
    def predict_next(self, last_item_id, top_n=5):
        """Predict next items given last viewed item"""
        if last_item_id not in self.transition_matrix:
            return []
        
        next_items = self.transition_matrix[last_item_id]
        sorted_items = sorted(
            next_items.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [item_id for item_id, prob in sorted_items[:top_n]]
    
    def get_session_score(self, last_item_id, candidate_item_id):
        """Get transition probability score"""
        if last_item_id not in self.transition_matrix:
            return 0.0
        
        return self.transition_matrix[last_item_id].get(candidate_item_id, 0.0)


class ImprovedHybridEnsemble:
    """
    Enhanced Hybrid Model with:
    - Fixed temporal decay
    - Item popularity boost
    - Job matching
    - Session-based signals
    - Adaptive weighting
    """
    
    def __init__(self, cb_model, cf_model, session_model, items_df, users_df,
                 w_cb=0.3, w_cf=0.5, w_session=0.1, w_popularity=0.1):
        self.cb = cb_model
        self.cf = cf_model
        self.session = session_model
        self.items = items_df.set_index('item_id')
        self.users = users_df.set_index('user_id')
        
        # Base weights
        self.w_cb = w_cb
        self.w_cf = w_cf
        self.w_session = w_session
        self.w_popularity = w_popularity
        
        # Precompute statistics
        self.max_views = self.items['nb_views'].max() if 'nb_views' in self.items.columns else 1
        self.median_views = self.items['nb_views'].median() if 'nb_views' in self.items.columns else 0

    def predict(self, user_id, item_id, user_history_items, adaptive_weights=True):
        """
        Enhanced prediction with multiple signals
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            user_history_items: List of items user has interacted with
            adaptive_weights: Whether to adapt weights based on user history size
            
        Returns:
            tuple: (hybrid_score, component_scores_dict)
        """
        # Get user and item data
        try:
            user_job = self.users.loc[user_id, 'job_clean'] if user_id in self.users.index else None
        except:
            user_job = None
        
        try:
            item_data = self.items.loc[item_id]
        except:
            # Item not in index, return low score
            return 1.0, {'cb': 0, 'cf': 1.0, 'session': 0, 'popularity': 0, 'job': 0, 'temporal': 1.0}
        
        # 1. Collaborative Score
        cf_score = self.cf.predict(user_id, item_id)
        
        # 2. Content-Based Score
        # FIX (Cause 6): when there is no history the CB score previously hard-coded
        # to 1.0 (minimum). Use 3.0 (neutral mid-scale) instead so it doesn't
        # drag down the hybrid for cold-start users.
        if user_history_items:
            cb_scores = [
                self.cb.get_similarity_score(item_id, hist_item) 
                for hist_item in user_history_items
            ]
            cb_score = np.mean(cb_scores) if cb_scores else 0.0
            cb_scaled = (cb_score * 4) + 1  # cosine [0,1] → [1,5]
        else:
            cb_score = 0.0
            cb_scaled = 3.0  # neutral mid-scale for cold-start
        
        # 3. Session-Based Score
        # FIX (Cause 6): same treatment — 3.0 instead of 1.0 when no session data
        if user_history_items:
            last_item = user_history_items[-1]
            session_prob = self.session.get_session_score(last_item, item_id)
            session_score = session_prob * 5.0 if session_prob > 0 else 3.0
        else:
            session_score = 3.0  # neutral mid-scale for cold-start
        
        # 4. Popularity Score
        # FIX (Cause 5): preprocessor already stores a p95-capped popularity_score
        # (0–1). Multiply by 5 to bring into [0,5] range. The old formula divided
        # by max_views (1,114) which made median items score only 0.26.
        if 'popularity_score' in item_data.index:
            popularity_score = float(item_data['popularity_score']) * 5.0
        elif 'nb_views' in item_data.index:
            # Fallback: clip by p95 of entire catalogue
            p95 = self.items['nb_views'].quantile(0.95) if 'nb_views' in self.items.columns else 1
            cap = p95 if p95 > 0 else 1
            popularity_score = min(float(item_data['nb_views']) / cap, 1.0) * 5.0
        else:
            popularity_score = 3.0  # neutral if no view data
        
        # 5. Job Match Score
        job_boost = 0.0
        if user_job and 'target_jobs' in item_data.index:
            try:
                target_jobs = item_data['target_jobs']
                if isinstance(target_jobs, list) and user_job in target_jobs:
                    job_boost = 0.5  # Significant boost for job match
            except:
                pass
        
        # 6. Temporal Decay (FIXED: Use actual item age with high floor for e-learning)
        if 'created_at' in item_data.index:
            try:
                item_date = pd.to_datetime(item_data['created_at'])
                current_date = pd.Timestamp.now()
                days_old = (current_date - item_date).days
                
                # E-learning content stays relevant much longer than news/social media
                # Use very gentle decay with high floor
                decay_rate = 0.00005  # Extremely gentle (half of previous)
                
                # Calculate decay with HIGH minimum floor of 0.9 (90% of original score)
                raw_decay = np.exp(-decay_rate * max(0, days_old))
                decay_factor = max(0.9, raw_decay)  # Never go below 90%
                
                # Special handling: content < 2 years old gets full score
                if days_old < 730:  # 2 years
                    decay_factor = 1.0
                    
                # Freshness boost: very recent content (< 6 months) gets slight boost
                if days_old < 180:
                    decay_factor = 1.05  # 5% boost for very fresh content
                    
            except:
                decay_factor = 1.0
        else:
            decay_factor = 1.0
        
        # Adaptive Weighting based on user history size
        if adaptive_weights:
            history_size = len(user_history_items)
            
            if history_size == 0:
                # Full cold start: rely on popularity + CF item biases
                w_pop = 0.5
                w_cb = 0.0
                w_cf = 0.5
                w_session = 0.0
            elif history_size < 5:
                # Partial cold start
                w_pop = 0.3
                w_cb = 0.2
                w_cf = 0.4
                w_session = 0.1
            else:
                # Established user: use configured weights
                w_pop = self.w_popularity
                w_cb = self.w_cb
                w_cf = self.w_cf
                w_session = self.w_session
        else:
            w_pop = self.w_popularity
            w_cb = self.w_cb
            w_cf = self.w_cf
            w_session = self.w_session
        
        # Combine scores
        hybrid_score = (
            w_cb * cb_scaled + 
            w_cf * cf_score + 
            w_session * session_score +
            w_pop * popularity_score +
            job_boost
        ) * decay_factor
        
        # Ensure score is in reasonable range [1, 5]
        # The raw hybrid score should already be roughly 1-5, but enforce bounds
        hybrid_score = np.clip(hybrid_score, 1.0, 5.0)
        
        # Return score and components for debugging
        components = {
            'cb': cb_scaled,
            'cf': cf_score,
            'session': session_score,
            'popularity': popularity_score,
            'job': job_boost,
            'temporal': decay_factor,
            'raw_hybrid': (w_cb * cb_scaled + w_cf * cf_score + w_session * session_score + w_pop * popularity_score + job_boost),
            'weights': {'cb': w_cb, 'cf': w_cf, 'session': w_session, 'pop': w_pop}
        }
        
        return hybrid_score, components


class DiversityReranker:
    """Maximal Marginal Relevance (MMR) for diversity-aware re-ranking"""
    
    def __init__(self, cb_model, alpha=0.7):
        """
        Args:
            cb_model: ContentBasedRecommender for similarity calculation
            alpha: Tradeoff between relevance (1.0) and diversity (0.0)
        """
        self.cb = cb_model
        self.alpha = alpha
    
    def rerank(self, candidate_scores, top_k=10):
        """
        Re-rank candidates for diversity
        
        Args:
            candidate_scores: List of (item_id, score) tuples sorted by score
            top_k: Number of items to return
            
        Returns:
            List of re-ranked item_ids
        """
        if len(candidate_scores) <= 1:
            return [item_id for item_id, _ in candidate_scores[:top_k]]
        
        selected = []
        remaining = list(candidate_scores)
        
        # Select top item by relevance
        selected.append(remaining[0][0])
        remaining = remaining[1:]
        
        # Iteratively select items balancing relevance and diversity
        while remaining and len(selected) < top_k:
            mmr_scores = []
            
            for item_id, relevance_score in remaining:
                # Calculate similarity to already selected items
                similarities = []
                for selected_id in selected:
                    sim = self.cb.get_similarity_score(item_id, selected_id)
                    similarities.append(sim)
                
                max_similarity = max(similarities) if similarities else 0
                
                # MMR formula: balance relevance vs. novelty
                mmr = self.alpha * relevance_score - (1 - self.alpha) * max_similarity
                mmr_scores.append((item_id, mmr))
            
            # Select item with highest MMR
            best_item = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_item[0])
            remaining = [(id, score) for id, score in remaining if id != best_item[0]]
        
        return selected
