import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

# ==========================================
# 1. DATA & MODEL INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource
def load_and_train_system():
    """Loads data, preprocesses, and trains the improved models once per session."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    # Load
    loader = DataLoader(data_dir)
    users, items, explicit, implicit = loader.load_data()
    
    # Preprocess with improvements
    preprocessor = ImprovedDataPreprocessor()
    interactions = preprocessor.create_composite_ratings(explicit, implicit)
    items_processed, tfidf_matrix = preprocessor.process_content_features(items)
    
    # Extract sessions
    sessions = preprocessor.extract_sessions(interactions, session_gap_minutes=60)
    
    # Create user features
    users_processed = preprocessor.create_user_features(users, interactions, items_processed)
    
    # Train Models
    cb_model = ContentBasedRecommender(tfidf_matrix, items_processed)
    cf_model = CollaborativeRecommender(n_factors=100)
    cf_model.train(interactions, test_size=0.2, validation_size=0.1)
    
    session_model = SessionBasedRecommender(sessions)
    
    hybrid_model = ImprovedHybridEnsemble(
        cb_model, cf_model, session_model, 
        items_processed, users_processed,
        w_cb=0.3, w_cf=0.5, w_session=0.1, w_popularity=0.1
    )
    
    diversity_reranker = DiversityReranker(cb_model, alpha=0.7)
    
    engine = ImprovedRecommendationEngine(
        users_processed, items_processed, interactions, 
        hybrid_model, diversity_reranker
    )
    
    return users_processed, items_processed, interactions, engine

# Execute the caching function
users, items, interactions, engine = load_and_train_system()


# ==========================================
# 2. MAIN UI SETUP
# ==========================================
st.set_page_config(page_title="MARS Recommender System - Enhanced", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 E-Learning Recommender System (MARS) - Enhanced</div>', 
            unsafe_allow_html=True)
st.markdown("---")


# ==========================================
# 3. USER SELECTION & PROFILE
# ==========================================
st.sidebar.header("🔧 Configuration")

valid_users = interactions['user_id'].unique().tolist()
options = ["✨ Simulate New User (Cold Start) ✨"] + valid_users

selected_option = st.sidebar.selectbox(
    "Select a User ID:",
    options,
    help="Choose an existing user or simulate a cold start scenario"
)

# Recommendation settings
st.sidebar.subheader("Recommendation Settings")
top_n = st.sidebar.slider("Number of recommendations per category:", 3, 20, 10)
enable_diversity = st.sidebar.checkbox("Enable diversity re-ranking", value=True, 
                                       help="Ensures recommendations are diverse")
show_explanations = st.sidebar.checkbox("Show detailed explanations", value=True)


# Routing: Cold Start vs. Existing User
if selected_option == "✨ Simulate New User (Cold Start) ✨":
    selected_user = "NEW_USER_999"
    is_cold_start = True
else:
    selected_user = selected_option
    is_cold_start = False

# Get user profile
user_profile = engine.generate_user_profile_summary(selected_user)


# ==========================================
# 4. USER PROFILE DISPLAY
# ==========================================
st.subheader("👤 User Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("User ID", selected_user)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    job_display = user_profile['job'] if user_profile['job'] != 'Unknown' else "Not Specified"
    st.metric("Job Title", job_display)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Interactions", user_profile['interaction_count'])
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Favorite Theme", user_profile['favorite_theme'])
    st.markdown('</div>', unsafe_allow_html=True)

if is_cold_start:
    st.warning("⚠️ **Cold Start Mode**: This user has no interaction history. "
               "Recommendations will be based on global popularity and collaborative item biases.")
else:
    st.success(f"✅ **Active Learner**: This user has {user_profile['interaction_count']} interactions. "
               f"Preferred content type: {user_profile['preferred_type']}")


# ==========================================
# 5. USER LEARNING HISTORY
# ==========================================
if not is_cold_start and user_profile['interaction_count'] > 0:
    st.markdown("### 📚 Learning History")
    
    history_df = engine.get_user_history_with_details(selected_user, limit=10)
    
    if not history_df.empty:
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No detailed history available")

st.markdown("---")


# ==========================================
# 6. GENERATE RECOMMENDATIONS
# ==========================================
st.markdown("### 🎯 Personalized Recommendations")

with st.spinner("Generating personalized recommendations..."):
    recommendations = engine.generate_recommendations(
        selected_user, 
        top_n=top_n, 
        diversify=enable_diversity
    )


# ==========================================
# 7. DISPLAY RECOMMENDATIONS BY CATEGORY
# ==========================================

# Determine which categories to show based on user context
has_job = user_profile['job'] not in ['Unknown', 'Not Specified', 'New User (No Profile)']
has_history = user_profile['interaction_count'] > 0

# Define category order and visibility
category_order = []

# Always show hybrid
category_order.append('hybrid')

# Job-related if user has job
if has_job and len(recommendations.get('job_related', [])) > 0:
    category_order.append('job_related')

# Trending
if len(recommendations.get('trending', [])) > 0:
    category_order.append('trending')

# Content-based and collaborative if user has history
if has_history:
    category_order.append('content_based')
    category_order.append('collaborative')
    
    # Session-based if available
    if len(recommendations.get('session_based', [])) > 0:
        category_order.append('session_based')

# Hidden gems
if len(recommendations.get('hidden_gems', [])) > 0:
    category_order.append('hidden_gems')


# Display each category
for category in category_order:
    recs = recommendations.get(category, [])
    
    if len(recs) == 0:
        continue
    
    # Get category title and description
    title, description = engine.get_category_title_and_description(category, has_job)
    
    st.markdown(f'<div class="category-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f"*{description}*")
    
    # Display recommendations
    for i, rec in enumerate(recs[:top_n], 1):
        item_id = rec['item_id']
        score = rec['score']
        
        # Get item details
        try:
            item_data = items[items['item_id'] == item_id].iloc[0]
            item_name = item_data['name']
            item_theme = item_data.get('theme', 'Unknown')
            item_type = item_data.get('type', 'Unknown')
            item_difficulty = item_data.get('difficulty', 'Unknown')
            item_duration = item_data.get('duration', 0)
        except:
            item_name = "Unknown Item"
            item_theme = "Unknown"
            item_type = "Unknown"
            item_difficulty = "Unknown"
            item_duration = 0
        
        # Format duration
        if item_duration > 0:
            duration_min = int(item_duration / 60)
            duration_display = f"{duration_min} min"
        else:
            duration_display = "N/A"
        
        # Create expandable recommendation card
        with st.expander(f"**{i}. {item_name}** ⭐ {score:.2f}/5.0"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Theme:** {item_theme}")
                st.markdown(f"**Type:** {item_type}")
            
            with col2:
                st.markdown(f"**Difficulty:** {item_difficulty}")
                st.markdown(f"**Duration:** {duration_display}")
            
            with col3:
                st.markdown(f"**Score:** {score:.2f}/5.0")
            
            # Show explanation if enabled
            if show_explanations:
                user_history = engine.user_interactions.get(selected_user, [])
                explanation = engine.get_recommendation_explanation(
                    selected_user, item_id, user_history
                )
                st.info(f"**Why this recommendation?**  \n{explanation['explanation']}")
                
                # Show component scores
                components = rec['components']
                if components:
                    st.markdown("**Score Breakdown:**")
                    comp_cols = st.columns(5)
                    
                    with comp_cols[0]:
                        st.metric("Content", f"{components.get('cb', 0):.2f}")
                    with comp_cols[1]:
                        st.metric("Collaborative", f"{components.get('cf', 0):.2f}")
                    with comp_cols[2]:
                        st.metric("Session", f"{components.get('session', 0):.2f}")
                    with comp_cols[3]:
                        st.metric("Popularity", f"{components.get('popularity', 0):.2f}")
                    with comp_cols[4]:
                        st.metric("Temporal", f"{components.get('temporal', 0):.2f}")


# ==========================================
# 8. VISUALIZATIONS
# ==========================================
st.markdown("---")
st.markdown("### 📊 Recommendation Insights")

tab1, tab2, tab3 = st.tabs(["Theme Distribution", "Score Distribution", "Component Analysis"])

with tab1:
    # Theme distribution across all recommendations
    all_recs = recommendations.get('hybrid', [])
    
    if all_recs:
        themes = []
        for rec in all_recs:
            try:
                item_data = items[items['item_id'] == rec['item_id']].iloc[0]
                themes.append(item_data.get('theme', 'Unknown'))
            except:
                pass
        
        if themes:
            theme_counts = pd.Series(themes).value_counts().reset_index()
            theme_counts.columns = ['Theme', 'Count']
            
            fig = px.bar(
                theme_counts,
                x='Theme',
                y='Count',
                title='Theme Distribution in Recommendations',
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Score distribution
    all_recs = recommendations.get('hybrid', [])
    
    if all_recs:
        scores = [rec['score'] for rec in all_recs]
        
        fig = px.histogram(
            x=scores,
            nbins=20,
            title='Score Distribution',
            labels={'x': 'Recommendation Score', 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {np.mean(scores):.2f}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Component contribution analysis
    all_recs = recommendations.get('hybrid', [])
    
    if all_recs and len(all_recs) > 0:
        # Calculate average component contributions
        avg_components = {
            'Content-Based': np.mean([r['components'].get('cb', 0) for r in all_recs]),
            'Collaborative': np.mean([r['components'].get('cf', 0) for r in all_recs]),
            'Session-Based': np.mean([r['components'].get('session', 0) for r in all_recs]),
            'Popularity': np.mean([r['components'].get('popularity', 0) for r in all_recs])
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(avg_components.keys()),
                y=list(avg_components.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )
        ])
        fig.update_layout(
            title='Average Component Contribution to Recommendations',
            xaxis_title='Component',
            yaxis_title='Average Score',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 9. FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Enhanced E-Learning Recommendation System (MARS)</strong></p>
    <p>Powered by Hybrid Ensemble (Content-Based + Collaborative + Session-Based) with Diversity Re-ranking</p>
    <p>Features: Temporal Decay • Popularity Boosting • Job Matching • Session Patterns • Diversity Enforcement</p>
</div>
""", unsafe_allow_html=True)
