import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------
def _normalize(items_list, signal_key, lo=1.0, hi=4.8):
    """
    Re-scale items in-place.
    - signal_key = 'hybrid'  → use item['_raw']
    - signal_key = 'nb_views'→ use item['_nb_views']  (raw integer, not capped score)
    - anything else          → use item['components'][signal_key]

    If all values are identical (span < 1e-9) we fall back to item['_raw']
    so the list is still ordered meaningfully instead of all showing 2.90.
    """
    if not items_list:
        return items_list

    def _val(item):
        if signal_key == 'hybrid':
            return item['_raw']
        if signal_key == 'nb_views':
            return item.get('_nb_views', 0)
        return item['components'].get(signal_key, 0)

    vals  = np.array([_val(item) for item in items_list], dtype=float)
    span  = vals.max() - vals.min()

    # If signal is flat, fall back to the raw hybrid score for differentiation
    if span < 1e-9:
        vals = np.array([item['_raw'] for item in items_list], dtype=float)
        span = vals.max() - vals.min()

    for item, v in zip(items_list, vals):
        if span > 1e-9:
            item['score'] = round(lo + (v - vals.min()) / span * (hi - lo), 2)
        else:
            item['score'] = round((lo + hi) / 2, 2)

    return items_list


class ImprovedRecommendationEngine:

    def __init__(self, users_df, items_df, interactions_df, hybrid_model, diversity_reranker):
        self.users        = users_df.set_index('user_id')
        self.items        = items_df.set_index('item_id')
        self.interactions = interactions_df
        self.hybrid       = hybrid_model
        self.diversity_reranker = diversity_reranker
        self._precompute_statistics()

    # ------------------------------------------------------------------
    def _precompute_statistics(self):
        if 'nb_views' in self.items.columns:
            self.item_popularity = self.items['nb_views'].to_dict()
            threshold = self.items['nb_views'].quantile(0.8)
            self.trending_items = set(self.items[self.items['nb_views'] >= threshold].index)
        else:
            self.item_popularity = {}
            self.trending_items  = set()

        item_ratings = self.interactions.groupby('item_id').agg(
            avg_rating=('scaled_score', 'mean'),
            rating_count=('user_id', 'count')
        ).reset_index()
        self.item_avg_ratings = item_ratings.set_index('item_id')['avg_rating'].to_dict()

        self.user_interactions = (
            self.interactions.groupby('user_id')['item_id'].apply(list).to_dict()
        )

        if 'job_clean' in self.users.columns:
            self.job_to_users = (
                self.users.reset_index()
                .groupby('job_clean')['user_id'].apply(set).to_dict()
            )
        else:
            self.job_to_users = {}

    # ------------------------------------------------------------------
    def _score_all_candidates(self, user_id, user_history):
        candidate_items = set(self.items.index) - set(user_history)
        scored = []
        for item_id in candidate_items:
            raw, components = self.hybrid.predict(user_id, item_id, user_history)
            scored.append({
                'item_id':    item_id,
                'score':      raw,
                '_raw':       raw,
                '_nb_views':  float(self.item_popularity.get(item_id, 0)),
                'components': components,
            })
        return scored

    # ------------------------------------------------------------------
    def _job_candidates(self, user_job, user_id, user_history, scored_by_id, top_n):
        """
        Build 'Based on Your Profession'.
        Three levels — stops as soon as we have enough candidates:
          1. Exact tag match  (item target_jobs contains user_job)
          2. Token overlap    (any shared word)
          3. Same-job CF      (items most watched by users with same job)
        No hybrid fallback — we show fewer but cleaner results.
        """
        history_set = set(user_history)
        seen        = set()
        results     = []

        def _add(item_id, match_type):
            if item_id in history_set or item_id in seen:
                return
            base = scored_by_id.get(item_id)
            if base is None:
                return
            results.append({**base, 'job_match': match_type})
            seen.add(item_id)

        # Pass 1 — exact
        for item_id, item_data in self.items.iterrows():
            tj = item_data.get('target_jobs', [])
            if isinstance(tj, list) and user_job in tj:
                _add(item_id, 'exact')

        # Pass 2 — token overlap
        if len(results) < top_n:
            user_tokens = set(user_job.lower().split())
            for item_id, item_data in self.items.iterrows():
                tj = item_data.get('target_jobs', [])
                if not isinstance(tj, list):
                    continue
                tj_tokens = set(' '.join(tj).lower().split())
                if user_tokens & tj_tokens:
                    _add(item_id, 'partial')

        # Pass 3 — same-job CF (items most watched by colleagues)
        if len(results) < top_n:
            same_job_users = self.job_to_users.get(user_job, set()) - {user_id}
            freq: dict = defaultdict(int)
            for uid in same_job_users:
                for iid in self.user_interactions.get(uid, []):
                    if iid not in history_set:
                        freq[iid] += 1
            for item_id, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True):
                _add(item_id, 'same_job_cf')
                if len(results) >= top_n * 2:
                    break

        return results

    # ------------------------------------------------------------------
    def generate_recommendations(self, user_id, top_n=10, diversify=True):
        user_exists  = user_id in self.users.index
        user_history = self.user_interactions.get(user_id, [])
        user_job     = None
        if user_exists:
            user_job = self.users.loc[user_id].get('job_clean', None)

        scored_items = self._score_all_candidates(user_id, user_history)
        if not scored_items:
            return {}

        scored_by_id = {item['item_id']: item for item in scored_items}
        recommendations = {}

        # ----------------------------------------------------------------
        # 1. TOP PICKS — hybrid score, diversity re-ranked
        # ----------------------------------------------------------------
        sorted_hybrid = sorted(scored_items, key=lambda x: x['_raw'], reverse=True)
        if diversify:
            cand_scores = [(item['item_id'], item['_raw']) for item in sorted_hybrid]
            diverse_ids = set(self.diversity_reranker.rerank(cand_scores, top_k=top_n * 2))
            hybrid_pool = [item for item in sorted_hybrid if item['item_id'] in diverse_ids]
        else:
            hybrid_pool = sorted_hybrid

        hybrid_list = [dict(item) for item in hybrid_pool[:top_n]]
        _normalize(hybrid_list, 'hybrid')
        recommendations['hybrid'] = hybrid_list

        # ----------------------------------------------------------------
        # 2. BASED ON YOUR PROFESSION
        #    Sort by CF score (most personalised), normalize by CF.
        #    Each match tier (exact → partial → same-job) is normalized
        #    separately so there is no cliff between tiers.
        # ----------------------------------------------------------------
        if user_job:
            job_pool = self._job_candidates(user_job, user_id, user_history, scored_by_id, top_n)

            # Split by tier and normalize each independently, then concatenate
            tiers = [
                [item for item in job_pool if item.get('job_match') == t]
                for t in ('exact', 'partial', 'same_job_cf')
            ]
            job_list = []
            for tier in tiers:
                if not tier:
                    continue
                tier_sorted = sorted(tier, key=lambda x: x['components'].get('cf', 0), reverse=True)
                # Map this tier into a sub-range proportional to tier size
                remaining = top_n - len(job_list)
                if remaining <= 0:
                    break
                chunk = tier_sorted[:remaining]
                # Normalize chunk into [hi - step, hi] based on position within full list
                hi_chunk = 4.8 - (len(job_list) / top_n) * 2.5
                lo_chunk = max(1.0, hi_chunk - 2.0)
                _normalize(chunk, 'cf', lo=lo_chunk, hi=hi_chunk)
                job_list.extend(chunk)

            recommendations['job_related'] = job_list
        else:
            recommendations['job_related'] = []

        # ----------------------------------------------------------------
        # 3. TRENDING NOW — sorted by raw nb_views (not capped score)
        # ----------------------------------------------------------------
        trending_pool = sorted(
            [item for item in scored_items if item['item_id'] in self.trending_items],
            key=lambda x: x['_nb_views'],
            reverse=True
        )
        trending_list = [dict(item) for item in trending_pool[:top_n]]
        _normalize(trending_list, 'nb_views')          # uses raw view counts → no flat scores
        recommendations['trending'] = trending_list

        # ----------------------------------------------------------------
        # 4. BECAUSE YOU WATCHED — sorted + normalized by CB signal
        # ----------------------------------------------------------------
        cb_pool = sorted(scored_items, key=lambda x: x['components'].get('cb', 0), reverse=True)
        cb_list = [dict(item) for item in cb_pool[:top_n]]
        _normalize(cb_list, 'cb')
        recommendations['content_based'] = cb_list

        # ----------------------------------------------------------------
        # 5. SIMILAR LEARNERS ALSO WATCHED — sorted + normalized by CF
        # ----------------------------------------------------------------
        cf_pool = sorted(scored_items, key=lambda x: x['components'].get('cf', 0), reverse=True)
        cf_list = [dict(item) for item in cf_pool[:top_n]]
        _normalize(cf_list, 'cf')
        recommendations['collaborative'] = cf_list

        # ----------------------------------------------------------------
        # 6. WHAT TO WATCH NEXT — real session transitions first
        #    Items with an actual transition (session > neutral 3.0) are
        #    sorted by session signal and normalized in [2.5, 4.8].
        #    If fewer than top_n, fill from CF-pool (normalized in [1.0, 2.4])
        #    so both groups are distinguishable.
        # ----------------------------------------------------------------
        if user_history:
            real_session = sorted(
                [item for item in scored_items if item['components'].get('session', 3.0) > 3.0],
                key=lambda x: x['components'].get('session', 0),
                reverse=True
            )[:top_n]

            # Fill remainder from CF pool (exclude already picked)
            real_ids = {item['item_id'] for item in real_session}
            cf_fill  = [item for item in cf_pool if item['item_id'] not in real_ids]
            fill_count = top_n - len(real_session)

            session_list = [dict(item) for item in real_session]
            fill_list    = [dict(item) for item in cf_fill[:fill_count]]

            _normalize(session_list, 'session', lo=2.5, hi=4.8)
            _normalize(fill_list,    'cf',       lo=1.0, hi=2.4)

            recommendations['session_based'] = session_list + fill_list
        else:
            recommendations['session_based'] = []

        # ----------------------------------------------------------------
        # 7. HIDDEN GEMS — below-median popularity, sorted AND normalized
        #    by _raw hybrid so scores are monotonically decreasing
        # ----------------------------------------------------------------
        if self.item_popularity:
            median_views = np.median(list(self.item_popularity.values()))
            gem_pool = sorted(
                [item for item in scored_items
                 if item['_nb_views'] < median_views],
                key=lambda x: x['_raw'],
                reverse=True
            )
            gem_list = [dict(item) for item in gem_pool[:top_n]]
            _normalize(gem_list, 'hybrid')
            recommendations['hidden_gems'] = gem_list
        else:
            recommendations['hidden_gems'] = []

        return recommendations

    # ------------------------------------------------------------------
    def get_recommendation_explanation(self, user_id, item_id, user_history):
        score, components = self.hybrid.predict(user_id, item_id, user_history)
        try:
            item_data  = self.items.loc[item_id]
            item_name  = item_data.get('name',  'Unknown')
            item_theme = item_data.get('theme', 'Unknown')
            item_type  = item_data.get('type',  'Unknown')
        except Exception:
            item_name = item_theme = item_type = 'Unknown'

        explanations = []
        if components.get('job', 0) > 0:
            explanations.append("Specifically designed for your profession")
        if components.get('cb', 0) > 3.0 and user_history:
            explanations.append(
                f"Similar to content you've watched (Theme: {item_theme}, Type: {item_type})"
            )
        if components.get('cf', 0) > 3.5:
            explanations.append("Learners with similar interests highly rated this")
        if components.get('session', 0) > 3.0 and user_history:
            explanations.append("Often watched right after courses in your history")
        if components.get('popularity', 0) > 3.5:
            explanations.append("Trending and popular across all learners")
        if not explanations:
            explanations.append("Recommended based on your overall learning profile")

        return {
            'item_id':     item_id,
            'item_name':   item_name,
            'score':       score,
            'explanation': ' • '.join(explanations),
            'components':  components,
        }

    # ------------------------------------------------------------------
    def get_category_title_and_description(self, category, has_job=False):
        mapping = {
            'hybrid': (
                "🎯 Top Picks For You",
                "Our smartest recommendations — combining your history, similar learners, and content trends"
            ),
            'job_related': (
                "💼 Based on Your Profession",
                "Courses matched to your job role — curated from items tagged for your field and from colleagues in the same role"
            ),
            'trending': (
                "📈 Trending Now",
                "The most-viewed courses across all learners right now"
            ),
            'content_based': (
                "🔍 Because You Watched",
                "Courses whose topics, themes, and software closely match what you've already completed"
            ),
            'collaborative': (
                "👥 Similar Learners Also Watched",
                "Courses finished and rated highly by learners with a profile like yours"
            ),
            'session_based': (
                "▶️ What to Watch Next",
                "Courses other learners typically watch right after the ones you've just completed"
            ),
            'hidden_gems': (
                "💎 Hidden Gems",
                "Highly-rated courses that fly under the radar — fewer views but strong engagement"
            ),
        }
        return mapping.get(category, (category.replace('_', ' ').title(), ""))

    # ------------------------------------------------------------------
    def generate_user_profile_summary(self, user_id):
        if user_id not in self.users.index:
            return {
                'exists': False, 'user_id': user_id,
                'job': 'New User (No Profile)',
                'interaction_count': 0, 'favorite_theme': 'None',
                'avg_rating': 0, 'preferred_type': 'Unknown',
            }
        user_info         = self.users.loc[user_id]
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        interaction_count = len(user_interactions)
        avg_rating        = user_interactions['scaled_score'].mean() if interaction_count > 0 else 0

        favorite_theme = preferred_type = 'Unknown'
        if interaction_count > 0:
            merged = user_interactions.merge(
                self.items.reset_index()[['item_id', 'theme', 'type']],
                on='item_id', how='left'
            )
            modes = merged['theme'].mode()
            favorite_theme = modes[0] if len(modes) else 'Various'
            modes = merged['type'].mode()
            preferred_type = modes[0] if len(modes) else 'Various'

        return {
            'exists': True, 'user_id': user_id,
            'job':  user_info.get('job', 'Not Specified'),
            'interaction_count': interaction_count,
            'favorite_theme': favorite_theme,
            'avg_rating': float(avg_rating),
            'preferred_type': preferred_type,
            'avg_duration_preference': user_info.get('avg_duration_preference', 0),
        }

    # ------------------------------------------------------------------
    def get_user_history_with_details(self, user_id, limit=10):
        user_interactions = self.interactions[self.interactions['user_id'] == user_id]
        if len(user_interactions) == 0:
            return pd.DataFrame()
        history = user_interactions.merge(
            self.items.reset_index()[['item_id', 'name', 'theme', 'type', 'difficulty']],
            on='item_id', how='left'
        )
        history = history.sort_values('composite_score', ascending=False)
        cols = ['name', 'theme', 'type', 'difficulty', 'rating', 'views', 'composite_score']
        return history[[c for c in cols if c in history.columns]].head(limit)
