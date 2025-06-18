#!/usr/bin/env python3
"""
BeatWizard Recommendation System
A comprehensive music recommendation engine using multiple approaches:
1. Collaborative Filtering (User-Item and Item-Item)
2. Content-Based Filtering (Audio Features)
3. Hybrid Approach
4. Actionable Insights Generation
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class BeatRecommendationSystem:
    """
    Advanced Music Recommendation System for BeatWizard
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_profiles = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.nmf_model = None
        self.svd_model = None
        self.knn_model = None
        self.clusters = None
        self.feature_importance = None
        
    def load_data(self, beats_df, interactions_df, audio_features_df=None):
        """
        Load and prepare data for recommendation system
        
        Args:
            beats_df: DataFrame with beat information (id, title, genre, artist, etc.)
            interactions_df: DataFrame with user interactions (user_id, beat_id, rating/play_count)
            audio_features_df: DataFrame with audio features (beat_id, tempo, energy, etc.)
        """
        self.beats_df = beats_df.copy()
        self.interactions_df = interactions_df.copy()
        self.audio_features_df = audio_features_df.copy() if audio_features_df is not None else None
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        # Prepare content features
        self._prepare_content_features()
        
        print(f"✅ Data loaded successfully!")
        print(f"   - {len(self.beats_df)} beats")
        print(f"   - {len(self.interactions_df)} interactions")
        print(f"   - {len(self.user_item_matrix)} users")
        print(f"   - Audio features: {'Yes' if self.audio_features_df is not None else 'No'}")
        
    def _create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        # Aggregate interactions (sum play counts, average ratings, etc.)
        user_item = self.interactions_df.groupby(['user_id', 'beat_id']).agg({
            'interaction_type': 'count',  # Number of interactions
        }).reset_index()
        
        # Create pivot table
        self.user_item_matrix = user_item.pivot(
            index='user_id', 
            columns='beat_id', 
            values='interaction_type'
        ).fillna(0)
        
        print(f"📊 User-Item Matrix: {self.user_item_matrix.shape}")
        
    def _prepare_content_features(self):
        """Prepare content-based features"""
        features = []
        
        # Genre encoding
        if 'genre' in self.beats_df.columns:
            genre_encoded = pd.get_dummies(self.beats_df['genre'], prefix='genre')
            features.append(genre_encoded)
        
        # Audio features
        if self.audio_features_df is not None:
            # Merge with beats
            beats_with_audio = self.beats_df.merge(
                self.audio_features_df, 
                left_on='id', 
                right_on='beat_id', 
                how='left'
            )
            
            # Select numeric audio features
            audio_cols = ['tempo', 'energy_mean', 'bass_energy', 'rhythm_density', 
                         'beat_consistency', 'spectral_centroid_mean', 'spectral_rolloff_mean']
            
            available_audio_cols = [col for col in audio_cols if col in beats_with_audio.columns]
            
            if available_audio_cols:
                audio_features = beats_with_audio[available_audio_cols].fillna(0)
                # Normalize audio features
                audio_features_scaled = pd.DataFrame(
                    self.scaler.fit_transform(audio_features),
                    columns=available_audio_cols,
                    index=beats_with_audio.index
                )
                features.append(audio_features_scaled)
        
        # Combine all features
        if features:
            self.item_features = pd.concat(features, axis=1)
            self.item_features.index = self.beats_df['id']
        else:
            # Fallback: use basic features
            self.item_features = pd.get_dummies(self.beats_df['genre'], prefix='genre')
            self.item_features.index = self.beats_df['id']
            
        print(f"🎵 Content Features: {self.item_features.shape}")
        
    def train_collaborative_filtering(self, n_components=50):
        """Train collaborative filtering models"""
        print("🤖 Training Collaborative Filtering Models...")
        
        # Non-negative Matrix Factorization
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=200)
        user_factors = self.nmf_model.fit_transform(self.user_item_matrix)
        item_factors = self.nmf_model.components_
        
        # Truncated SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_factors_svd = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Store user profiles
        self.user_profiles = pd.DataFrame(
            user_factors, 
            index=self.user_item_matrix.index,
            columns=[f'factor_{i}' for i in range(n_components)]
        )
        
        print(f"   ✅ NMF Model trained with {n_components} components")
        print(f"   ✅ SVD Model trained with {n_components} components")
        
    def train_content_based(self, n_neighbors=20):
        """Train content-based filtering model"""
        print("🎨 Training Content-Based Model...")
        
        # KNN for finding similar items
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(self.item_features)
        
        # Calculate item-item similarity matrix
        self.similarity_matrix = cosine_similarity(self.item_features)
        
        print(f"   ✅ KNN Model trained with {n_neighbors} neighbors")
        print(f"   ✅ Similarity matrix computed: {self.similarity_matrix.shape}")
        
    def cluster_analysis(self, n_clusters=10):
        """Perform clustering analysis for insights"""
        print("🔍 Performing Clustering Analysis...")
        
        # Cluster beats based on content features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.item_features)
        
        # Add cluster info to beats
        self.beats_df['cluster'] = self.clusters
        
        # Analyze clusters
        cluster_analysis = []
        for cluster_id in range(n_clusters):
            cluster_beats = self.beats_df[self.beats_df['cluster'] == cluster_id]
            
            analysis = {
                'cluster_id': cluster_id,
                'size': len(cluster_beats),
                'top_genres': cluster_beats['genre'].value_counts().head(3).to_dict(),
                'avg_interactions': self._get_avg_interactions_for_beats(cluster_beats['id'].tolist())
            }
            cluster_analysis.append(analysis)
            
        self.cluster_analysis = cluster_analysis
        print(f"   ✅ {n_clusters} clusters identified")
        
    def _get_avg_interactions_for_beats(self, beat_ids):
        """Get average interactions for a list of beats"""
        beat_interactions = self.interactions_df[
            self.interactions_df['beat_id'].isin(beat_ids)
        ]
        return beat_interactions.groupby('beat_id').size().mean() if len(beat_interactions) > 0 else 0
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_recommendations(n_recommendations)
            
        # Get user factors
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_factors = self.user_profiles.iloc[user_idx].values
        
        # Predict ratings for all items
        item_factors = self.nmf_model.components_
        predicted_ratings = np.dot(user_factors, item_factors)
        
        # Get items user hasn't interacted with
        user_interactions = self.user_item_matrix.loc[user_id]
        unrated_items = user_interactions[user_interactions == 0].index
        
        # Get predictions for unrated items
        item_indices = [self.user_item_matrix.columns.get_loc(item) for item in unrated_items]
        unrated_predictions = predicted_ratings[item_indices]
        
        # Sort and get top recommendations
        top_indices = np.argsort(unrated_predictions)[::-1][:n_recommendations]
        recommended_items = [unrated_items[i] for i in top_indices]
        scores = [unrated_predictions[i] for i in top_indices]
        
        return list(zip(recommended_items, scores))
        
    def get_content_based_recommendations(self, user_id=None, beat_id=None, n_recommendations=10):
        """Get recommendations using content-based filtering"""
        if beat_id:
            # Find similar beats to a specific beat
            if beat_id not in self.item_features.index:
                return []
                
            beat_idx = self.item_features.index.get_loc(beat_id)
            similarities = self.similarity_matrix[beat_idx]
            
            # Get top similar beats (excluding the beat itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            recommended_items = [self.item_features.index[i] for i in similar_indices]
            scores = [similarities[i] for i in similar_indices]
            
            return list(zip(recommended_items, scores))
            
        elif user_id:
            # Find beats similar to user's preferences
            if user_id not in self.user_item_matrix.index:
                return self._get_popular_recommendations(n_recommendations)
                
            # Get user's liked beats
            user_interactions = self.user_item_matrix.loc[user_id]
            liked_beats = user_interactions[user_interactions > 0].index.tolist()
            
            if not liked_beats:
                return self._get_popular_recommendations(n_recommendations)
                
            # Calculate average feature profile of liked beats
            liked_features = self.item_features.loc[liked_beats]
            user_profile = liked_features.mean()
            
            # Find beats similar to user profile
            similarities = cosine_similarity([user_profile], self.item_features)[0]
            
            # Exclude already interacted beats
            for beat in liked_beats:
                if beat in self.item_features.index:
                    beat_idx = self.item_features.index.get_loc(beat)
                    similarities[beat_idx] = -1
                    
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][:n_recommendations]
            recommended_items = [self.item_features.index[i] for i in top_indices]
            scores = [similarities[i] for i in top_indices]
            
            return list(zip(recommended_items, scores))
            
        return []
        
    def get_hybrid_recommendations(self, user_id, n_recommendations=10, cf_weight=0.6, cb_weight=0.4):
        """Get hybrid recommendations combining collaborative and content-based"""
        # Get collaborative filtering recommendations
        cf_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        cf_dict = {item: score for item, score in cf_recs}
        
        # Get content-based recommendations
        cb_recs = self.get_content_based_recommendations(user_id=user_id, n_recommendations=n_recommendations * 2)
        cb_dict = {item: score for item, score in cb_recs}
        
        # Combine scores
        all_items = set(cf_dict.keys()) | set(cb_dict.keys())
        hybrid_scores = {}
        
        for item in all_items:
            cf_score = cf_dict.get(item, 0)
            cb_score = cb_dict.get(item, 0)
            
            # Normalize scores to 0-1 range
            cf_score_norm = (cf_score - min(cf_dict.values())) / (max(cf_dict.values()) - min(cf_dict.values())) if cf_dict else 0
            cb_score_norm = (cb_score - min(cb_dict.values())) / (max(cb_dict.values()) - min(cb_dict.values())) if cb_dict else 0
            
            hybrid_score = cf_weight * cf_score_norm + cb_weight * cb_score_norm
            hybrid_scores[item] = hybrid_score
            
        # Sort and return top recommendations
        sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
        
    def _get_popular_recommendations(self, n_recommendations=10):
        """Get popular beats as fallback recommendations"""
        beat_popularity = self.interactions_df.groupby('beat_id').size().sort_values(ascending=False)
        popular_beats = beat_popularity.head(n_recommendations)
        return [(beat_id, count) for beat_id, count in popular_beats.items()]
        
    def generate_actionable_insights(self, user_id=None):
        """Generate actionable insights for users and content creators"""
        insights = {
            'user_insights': {},
            'content_insights': {},
            'market_insights': {},
            'recommendations': {}
        }
        
        # User-specific insights
        if user_id and user_id in self.user_item_matrix.index:
            user_insights = self._analyze_user_behavior(user_id)
            insights['user_insights'] = user_insights
            
        # Content insights
        content_insights = self._analyze_content_performance()
        insights['content_insights'] = content_insights
        
        # Market insights
        market_insights = self._analyze_market_trends()
        insights['market_insights'] = market_insights
        
        # Strategic recommendations
        strategic_recs = self._generate_strategic_recommendations()
        insights['recommendations'] = strategic_recs
        
        return insights
        
    def _analyze_user_behavior(self, user_id):
        """Analyze individual user behavior patterns"""
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return {'status': 'new_user', 'message': 'Not enough interaction data'}
            
        # Analyze listening patterns
        genre_preferences = user_interactions.merge(
            self.beats_df, left_on='beat_id', right_on='id'
        )['genre'].value_counts()
        
        # Temporal patterns
        if 'timestamp' in user_interactions.columns:
            user_interactions['hour'] = pd.to_datetime(user_interactions['timestamp']).dt.hour
            listening_hours = user_interactions['hour'].value_counts().sort_index()
        else:
            listening_hours = {}
            
        # Diversity score
        unique_genres = len(genre_preferences)
        total_interactions = len(user_interactions)
        diversity_score = unique_genres / max(total_interactions, 1)
        
        return {
            'total_interactions': total_interactions,
            'favorite_genres': genre_preferences.head(3).to_dict(),
            'diversity_score': diversity_score,
            'listening_pattern': listening_hours.to_dict() if hasattr(listening_hours, 'to_dict') else {},
            'user_type': self._classify_user_type(total_interactions, diversity_score)
        }
        
    def _classify_user_type(self, interactions, diversity):
        """Classify user type based on behavior"""
        if interactions < 5:
            return 'casual_listener'
        elif interactions > 50 and diversity > 0.3:
            return 'music_explorer'
        elif interactions > 50 and diversity < 0.2:
            return 'genre_specialist'
        elif interactions > 20:
            return 'regular_listener'
        else:
            return 'casual_listener'
            
    def _analyze_content_performance(self):
        """Analyze content performance patterns"""
        # Beat performance metrics
        beat_stats = self.interactions_df.groupby('beat_id').agg({
            'user_id': 'nunique',  # Unique listeners
            'interaction_type': 'count'  # Total interactions
        }).rename(columns={'user_id': 'unique_listeners', 'interaction_type': 'total_interactions'})
        
        # Merge with beat info
        beat_performance = beat_stats.merge(
            self.beats_df, left_index=True, right_on='id'
        )
        
        # Genre performance
        genre_performance = beat_performance.groupby('genre').agg({
            'unique_listeners': 'mean',
            'total_interactions': 'mean'
        }).sort_values('total_interactions', ascending=False)
        
        # Top performing beats
        top_beats = beat_performance.nlargest(10, 'total_interactions')[
            ['title', 'genre', 'unique_listeners', 'total_interactions']
        ].to_dict('records')
        
        return {
            'top_genres': genre_performance.head(5).to_dict(),
            'top_beats': top_beats,
            'total_beats': len(self.beats_df),
            'avg_interactions_per_beat': beat_stats['total_interactions'].mean()
        }
        
    def _analyze_market_trends(self):
        """Analyze market trends and opportunities"""
        # Genre distribution
        genre_dist = self.beats_df['genre'].value_counts()
        
        # Interaction trends by genre
        genre_interactions = self.interactions_df.merge(
            self.beats_df, left_on='beat_id', right_on='id'
        ).groupby('genre').size()
        
        # Calculate genre saturation (supply vs demand)
        genre_saturation = {}
        for genre in genre_dist.index:
            supply = genre_dist[genre]
            demand = genre_interactions.get(genre, 0)
            saturation = supply / max(demand, 1)  # Higher = oversaturated
            genre_saturation[genre] = saturation
            
        # Identify opportunities
        opportunities = []
        for genre, saturation in sorted(genre_saturation.items(), key=lambda x: x[1]):
            if saturation < 0.5:  # Undersaturated
                opportunities.append({
                    'genre': genre,
                    'opportunity_type': 'high_demand_low_supply',
                    'saturation_score': saturation
                })
                
        return {
            'genre_distribution': genre_dist.to_dict(),
            'genre_saturation': genre_saturation,
            'opportunities': opportunities[:5],
            'trending_genres': genre_interactions.nlargest(5).to_dict()
        }
        
    def _generate_strategic_recommendations(self):
        """Generate strategic recommendations for platform growth"""
        recommendations = []
        
        # Content creation recommendations
        if hasattr(self, 'cluster_analysis'):
            for cluster in self.cluster_analysis:
                if cluster['avg_interactions'] > 10:  # High-performing cluster
                    recommendations.append({
                        'type': 'content_creation',
                        'action': f"Create more beats in cluster {cluster['cluster_id']}",
                        'reason': f"This cluster has {cluster['avg_interactions']:.1f} avg interactions",
                        'priority': 'high' if cluster['avg_interactions'] > 20 else 'medium'
                    })
                    
        # User engagement recommendations
        inactive_users = self._find_inactive_users()
        if len(inactive_users) > 0:
            recommendations.append({
                'type': 'user_engagement',
                'action': f"Re-engage {len(inactive_users)} inactive users",
                'reason': "Users with no recent activity detected",
                'priority': 'medium'
            })
            
        # Platform optimization
        recommendations.append({
            'type': 'platform_optimization',
            'action': "Implement personalized playlists",
            'reason': "Increase user engagement through better recommendations",
            'priority': 'high'
        })
        
        return recommendations
        
    def _find_inactive_users(self, days_threshold=30):
        """Find users who haven't interacted recently"""
        if 'timestamp' not in self.interactions_df.columns:
            return []
            
        recent_date = datetime.now() - timedelta(days=days_threshold)
        recent_interactions = self.interactions_df[
            pd.to_datetime(self.interactions_df['timestamp']) > recent_date
        ]
        
        all_users = set(self.interactions_df['user_id'].unique())
        active_users = set(recent_interactions['user_id'].unique())
        inactive_users = all_users - active_users
        
        return list(inactive_users)
        
    def visualize_insights(self, save_plots=True):
        """Create visualizations for insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Genre distribution
        genre_counts = self.beats_df['genre'].value_counts().head(10)
        axes[0, 0].bar(range(len(genre_counts)), genre_counts.values)
        axes[0, 0].set_xticks(range(len(genre_counts)))
        axes[0, 0].set_xticklabels(genre_counts.index, rotation=45)
        axes[0, 0].set_title('Beat Distribution by Genre')
        axes[0, 0].set_ylabel('Number of Beats')
        
        # 2. User interaction distribution
        user_interactions = self.interactions_df.groupby('user_id').size()
        axes[0, 1].hist(user_interactions.values, bins=20, alpha=0.7)
        axes[0, 1].set_title('User Interaction Distribution')
        axes[0, 1].set_xlabel('Number of Interactions')
        axes[0, 1].set_ylabel('Number of Users')
        
        # 3. Beat popularity
        beat_popularity = self.interactions_df.groupby('beat_id').size().sort_values(ascending=False)
        axes[1, 0].plot(range(len(beat_popularity)), beat_popularity.values)
        axes[1, 0].set_title('Beat Popularity Distribution')
        axes[1, 0].set_xlabel('Beat Rank')
        axes[1, 0].set_ylabel('Number of Interactions')
        axes[1, 0].set_yscale('log')
        
        # 4. Cluster visualization (if available)
        if hasattr(self, 'clusters'):
            cluster_sizes = pd.Series(self.clusters).value_counts().sort_index()
            axes[1, 1].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in cluster_sizes.index], autopct='%1.1f%%')
            axes[1, 1].set_title('Beat Clusters Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Clustering not performed', ha='center', va='center')
            axes[1, 1].set_title('Beat Clusters')
            
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('recommendation_insights.png', dpi=300, bbox_inches='tight')
            print("📊 Visualization saved as 'recommendation_insights.png'")
            
        plt.show()
        
    def export_recommendations(self, user_id, filename=None):
        """Export recommendations to JSON file"""
        recommendations = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'collaborative_filtering': self.get_collaborative_recommendations(user_id),
            'content_based': self.get_content_based_recommendations(user_id=user_id),
            'hybrid': self.get_hybrid_recommendations(user_id),
            'insights': self.generate_actionable_insights(user_id)
        }
        
        if filename is None:
            filename = f'recommendations_{user_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
            
        print(f"💾 Recommendations exported to {filename}")
        return filename

def create_sample_data():
    """Create sample data for testing the recommendation system"""
    np.random.seed(42)
    
    # Sample beats data
    genres = ['Hip-Hop', 'Electronic', 'Pop', 'R&B', 'Rock', 'Jazz', 'Classical']
    beats_data = []
    
    for i in range(100):
        beats_data.append({
            'id': f'beat_{i}',
            'title': f'Beat {i}',
            'genre': np.random.choice(genres),
            'artist': f'Artist_{i % 20}',
            'duration': np.random.randint(120, 300)
        })
    
    beats_df = pd.DataFrame(beats_data)
    
    # Sample interactions data
    interactions_data = []
    users = [f'user_{i}' for i in range(50)]
    
    for _ in range(1000):
        interactions_data.append({
            'user_id': np.random.choice(users),
            'beat_id': np.random.choice(beats_df['id']),
            'interaction_type': np.random.choice(['play', 'like', 'share']),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
    
    interactions_df = pd.DataFrame(interactions_data)
    
    # Sample audio features data
    audio_features_data = []
    for beat_id in beats_df['id']:
        audio_features_data.append({
            'beat_id': beat_id,
            'tempo': np.random.randint(80, 180),
            'energy_mean': np.random.uniform(0.3, 0.9),
            'bass_energy': np.random.uniform(0.1, 0.8),
            'rhythm_density': np.random.uniform(0.2, 0.7),
            'beat_consistency': np.random.uniform(0.5, 1.0),
            'spectral_centroid_mean': np.random.uniform(1000, 5000),
            'spectral_rolloff_mean': np.random.uniform(2000, 8000)
        })
    
    audio_features_df = pd.DataFrame(audio_features_data)
    
    return beats_df, interactions_df, audio_features_df

def main():
    """Main function to demonstrate the recommendation system"""
    print("🎵 BeatWizard Recommendation System")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    beats_df, interactions_df, audio_features_df = create_sample_data()
    
    # Initialize recommendation system
    rec_system = BeatRecommendationSystem()
    
    # Load data
    rec_system.load_data(beats_df, interactions_df, audio_features_df)
    
    # Train models
    rec_system.train_collaborative_filtering()
    rec_system.train_content_based()
    rec_system.cluster_analysis()
    
    # Generate recommendations for a sample user
    sample_user = 'user_0'
    print(f"\n🎯 Generating recommendations for {sample_user}...")
    
    # Get different types of recommendations
    cf_recs = rec_system.get_collaborative_recommendations(sample_user, 5)
    cb_recs = rec_system.get_content_based_recommendations(user_id=sample_user, n_recommendations=5)
    hybrid_recs = rec_system.get_hybrid_recommendations(sample_user, 5)
    
    print(f"\n📋 Collaborative Filtering Recommendations:")
    for i, (beat_id, score) in enumerate(cf_recs, 1):
        beat_info = beats_df[beats_df['id'] == beat_id].iloc[0]
        print(f"   {i}. {beat_info['title']} ({beat_info['genre']}) - Score: {score:.3f}")
    
    print(f"\n🎨 Content-Based Recommendations:")
    for i, (beat_id, score) in enumerate(cb_recs, 1):
        beat_info = beats_df[beats_df['id'] == beat_id].iloc[0]
        print(f"   {i}. {beat_info['title']} ({beat_info['genre']}) - Score: {score:.3f}")
    
    print(f"\n🔄 Hybrid Recommendations:")
    for i, (beat_id, score) in enumerate(hybrid_recs, 1):
        beat_info = beats_df[beats_df['id'] == beat_id].iloc[0]
        print(f"   {i}. {beat_info['title']} ({beat_info['genre']}) - Score: {score:.3f}")
    
    # Generate actionable insights
    print(f"\n💡 Generating actionable insights...")
    insights = rec_system.generate_actionable_insights(sample_user)
    
    print(f"\n👤 User Insights:")
    user_insights = insights['user_insights']
    print(f"   - User Type: {user_insights.get('user_type', 'Unknown')}")
    print(f"   - Total Interactions: {user_insights.get('total_interactions', 0)}")
    print(f"   - Diversity Score: {user_insights.get('diversity_score', 0):.3f}")
    print(f"   - Favorite Genres: {user_insights.get('favorite_genres', {})}")
    
    print(f"\n📈 Market Insights:")
    market_insights = insights['market_insights']
    print(f"   - Total Beats: {market_insights.get('total_beats', 0)}")
    print(f"   - Top Opportunities: {len(market_insights.get('opportunities', []))}")
    
    print(f"\n🎯 Strategic Recommendations:")
    for rec in insights['recommendations'][:3]:
        print(f"   - {rec['action']} ({rec['priority']} priority)")
        print(f"     Reason: {rec['reason']}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    rec_system.visualize_insights()
    
    # Export recommendations
    print(f"\n💾 Exporting recommendations...")
    filename = rec_system.export_recommendations(sample_user)
    
    print(f"\n✅ Recommendation system demo completed!")
    print(f"   Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 