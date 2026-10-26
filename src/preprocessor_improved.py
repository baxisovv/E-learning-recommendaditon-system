import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import ast

class ImprovedDataPreprocessor:
    """
    Enhanced data preprocessor with:
    - Recency weighting for interactions
    - Item popularity features
    - Job matching features
    - Content type inclusion
    - Session extraction
    - Duration features
    """
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.scaler = MinMaxScaler(feature_range=(1, 5))
        
    def create_composite_ratings(self, explicit, implicit):
        """
        Fixed composite ratings addressing:
        - Cause 3: watch_percentage = 0 for 75%+ of rows collapses rating contribution
        - Cause 4: implicit-only rows have no watch_pct signal at all
        - Recency weighting using actual timestamps
        """
        # Aggregate implicit interactions: view count + most recent timestamp
        implicit_agg = implicit.groupby(['user_id', 'item_id']).agg(
            views=('item_id', 'count'),
            last_interaction=('created_at', 'max')
        ).reset_index()

        # Merge explicit and implicit
        interactions = pd.merge(
            explicit,
            implicit_agg,
            on=['user_id', 'item_id'],
            how='outer'
        )

        # Fill missing numerics
        interactions['rating']          = interactions['rating'].fillna(0)
        interactions['watch_percentage'] = interactions['watch_percentage'].fillna(0)
        interactions['views']           = interactions['views'].fillna(0)

        # ------------------------------------------------------------------
        # FIX (Cause 3 & 4): Separate explicit and implicit signals.
        # Problem: rating × (watch_pct/100) → 0 for 75%+ of rows because
        # watch_pct is 0 for almost every implicit-only record.
        # Solution:
        #   explicit_signal  = rating / 10        (scale to 0–1, raw rating is 0–10)
        #   watch_signal     = watch_pct / 100     (0–1, use 0.20 if watch_pct = 0
        #                                           but a real rating exists)
        #   implicit_signal  = log1p(views)        (0–6.9 for 1–1000 views)
        #
        # Treat explicit and implicit rows differently:
        #   - Has explicit rating: combine rating + watch_pct + views
        #   - Implicit only:       rely purely on log1p(views)
        # ------------------------------------------------------------------

        has_explicit = interactions['rating'] > 0

        # For explicit rows: if watch_pct is 0 but rating exists, substitute 0.20
        # so the rating still contributes to the composite.
        effective_watch = interactions['watch_percentage'].copy()
        effective_watch[has_explicit & (effective_watch == 0)] = 20.0  # 20% default

        # Explicit signal: blend rating (0–10 → 0–1) and watch completion (0–1)
        explicit_signal = (
            (interactions['rating'] / 10.0) * 0.6 +
            (effective_watch / 100.0)        * 0.4
        ) * has_explicit.astype(int)  # 0 for implicit-only rows

        # Implicit signal: log-transformed view count (0–6.9 for 1–1000 views)
        implicit_signal = np.log1p(interactions['views'])

        # Combine: explicit rows get both signals; implicit-only rows get only implicit
        interactions['base_composite'] = (
            explicit_signal * 4.0 +   # weight explicit higher (0–4 range)
            implicit_signal * 1.0     # keep implicit as secondary (0–6.9)
        )

        # ------------------------------------------------------------------
        # Recency weighting (gentler decay: half-life ≈ 2 years)
        # ------------------------------------------------------------------
        interactions['created_at']      = pd.to_datetime(interactions['created_at'],      errors='coerce')
        interactions['last_interaction'] = pd.to_datetime(interactions['last_interaction'], errors='coerce')

        interaction_date = interactions[['created_at', 'last_interaction']].max(axis=1)
        days_ago = (pd.Timestamp.now() - interaction_date).dt.days.fillna(0).clip(lower=0)

        decay_rate = 0.001  # half-life ≈ 693 days (~2 years); gentler than before
        interactions['recency_weight'] = np.exp(-decay_rate * days_ago)

        interactions['composite_score'] = (
            interactions['base_composite'] * interactions['recency_weight']
        )

        # Scale to SVD range [1, 5] — now the distribution is spread across 0–10+
        interactions['scaled_score'] = self.scaler.fit_transform(
            interactions[['composite_score']]
        )

        return interactions
    
    def process_content_features(self, items):
        """
        Enhanced content features with:
        - Content type one-hot encoding
        - Popularity normalization
        - Duration features
        - Job target parsing
        """
        # Map difficulty to numeric
        diff_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        items['diff_weight'] = items['difficulty'].map(diff_map).fillna(1)
        
        # Add popularity features — FIX (Cause 5): clip at 95th percentile,
        # not max(), so one viral item (1,114 views) doesn't make every other
        # item's popularity score negligible.
        items['nb_views'] = pd.to_numeric(items['nb_views'], errors='coerce').fillna(0)
        p95_views = items['nb_views'].quantile(0.95)
        cap = p95_views if p95_views > 0 else 1
        items['popularity_score'] = (items['nb_views'].clip(upper=cap) / cap)
        
        # Add duration features (normalized)
        items['duration'] = pd.to_numeric(items['duration'], errors='coerce').fillna(0)
        max_duration = items['duration'].max() if items['duration'].max() > 0 else 1
        items['duration_normalized'] = items['duration'] / max_duration
        
        # Duration categories
        items['duration_category'] = pd.cut(
            items['duration'],
            bins=[0, 120, 300, 1000, 10000],
            labels=['short', 'medium', 'long', 'very_long']
        ).astype(str)
        
        # One-hot encode content type
        if 'type' in items.columns:
            type_dummies = pd.get_dummies(items['type'], prefix='type')
            items = pd.concat([items, type_dummies], axis=1)
            
            # Combine text features with type emphasis (repeat type 3x for TF-IDF)
            items['combined_text'] = (
                items['name'] + " " + 
                items['description'].fillna('') + " " + 
                items['theme'].fillna('') + " " + 
                items['software'].fillna('') + " " +
                items['type'].fillna('') + " " +
                items['type'].fillna('') + " " +
                items['type'].fillna('')  # Type repeated 3x for emphasis
            )
        else:
            items['combined_text'] = (
                items['name'] + " " + 
                items['description'].fillna('') + " " + 
                items['theme'].fillna('') + " " + 
                items['software'].fillna('')
            )
        
        # Parse job target fields (stored as string representation of lists)
        if 'job' in items.columns:
            items['target_jobs'] = items['job'].apply(self._parse_job_list)
        
        # Generate TF-IDF Matrix
        tfidf_matrix = self.tfidf.fit_transform(items['combined_text'])
        
        return items, tfidf_matrix
    
    def _parse_job_list(self, job_str):
        """Parse job list from string representation"""
        try:
            if pd.isna(job_str):
                return []
            # Try to parse as Python literal
            job_list = ast.literal_eval(job_str)
            if isinstance(job_list, list):
                return [str(j).lower().strip() for j in job_list]
            else:
                return [str(job_str).lower().strip()]
        except:
            return [str(job_str).lower().strip()] if job_str else []
    
    def extract_sessions(self, interactions, session_gap_minutes=60):
        """
        Extract user sessions based on time gaps.
        
        Args:
            interactions: DataFrame with user_id, item_id, created_at
            session_gap_minutes: Minutes of inactivity to define new session
            
        Returns:
            DataFrame with session information
        """
        # Ensure datetime
        interactions = interactions.copy()
        interactions['created_at'] = pd.to_datetime(interactions['created_at'], errors='coerce')
        
        # Sort by user and time
        interactions = interactions.sort_values(['user_id', 'created_at'])
        
        sessions = []
        session_gap = timedelta(minutes=session_gap_minutes)
        
        for user_id, group in interactions.groupby('user_id'):
            group = group.reset_index(drop=True)
            session_id = 0
            current_session = []
            
            for i, row in group.iterrows():
                if i == 0:
                    current_session.append(row['item_id'])
                else:
                    prev_time = group.loc[i-1, 'created_at']
                    curr_time = row['created_at']
                    
                    if pd.notna(prev_time) and pd.notna(curr_time):
                        time_gap = curr_time - prev_time
                        
                        if time_gap > session_gap:
                            # Save current session
                            if current_session:
                                sessions.append({
                                    'user_id': user_id,
                                    'session_id': f"{user_id}_{session_id}",
                                    'items': current_session.copy(),
                                    'session_length': len(current_session)
                                })
                            session_id += 1
                            current_session = [row['item_id']]
                        else:
                            current_session.append(row['item_id'])
                    else:
                        current_session.append(row['item_id'])
            
            # Save final session
            if current_session:
                sessions.append({
                    'user_id': user_id,
                    'session_id': f"{user_id}_{session_id}",
                    'items': current_session.copy(),
                    'session_length': len(current_session)
                })
        
        return pd.DataFrame(sessions)
    
    def create_user_features(self, users, interactions, items):
        """
        Create user-level features:
        - Average duration preference
        - Preferred content type
        - Interaction statistics
        """
        user_features = users.copy()
        
        # Clean job field
        user_features['job'] = user_features['job'].fillna('Unknown')
        user_features['job_clean'] = user_features['job'].str.lower().str.strip()
        
        # Calculate user interaction statistics
        user_stats = interactions.groupby('user_id').agg({
            'item_id': 'count',
            'scaled_score': ['mean', 'std'],
            'views': 'sum'
        }).reset_index()
        
        user_stats.columns = ['user_id', 'interaction_count', 'avg_rating', 'rating_std', 'total_views']
        user_features = user_features.merge(user_stats, on='user_id', how='left')
        
        # Calculate average duration preference
        user_item_interactions = interactions.merge(
            items[['item_id', 'duration', 'type']], 
            on='item_id', 
            how='left'
        )
        
        duration_prefs = user_item_interactions.groupby('user_id')['duration'].mean().reset_index()
        duration_prefs.columns = ['user_id', 'avg_duration_preference']
        user_features = user_features.merge(duration_prefs, on='user_id', how='left')
        
        # Calculate preferred content type
        if 'type' in user_item_interactions.columns:
            type_prefs = user_item_interactions.groupby(['user_id', 'type']).size().reset_index(name='count')
            type_prefs = type_prefs.sort_values('count', ascending=False).groupby('user_id').first().reset_index()
            type_prefs = type_prefs[['user_id', 'type']]
            type_prefs.columns = ['user_id', 'preferred_type']
            user_features = user_features.merge(type_prefs, on='user_id', how='left')
        
        return user_features
