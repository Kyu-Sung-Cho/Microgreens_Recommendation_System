import pandas as pd
import numpy as np
import sqlite3
import datetime
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class KoppertCressRecommender:
    """
    Provides personalized microgreen recommendations to chefs based on their preferences,
    purchase history, seasonality, and more. Implements advanced recommendation algorithms
    using SVD and TF-IDF techniques.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the recommendation system and connect to database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.conn = sqlite3.connect(db_path)
        self.load_data()
        self.setup_advanced_models()
        
    def load_data(self) -> None:
        """Load necessary data tables from the database"""
        self.microgreens = pd.read_sql("SELECT * FROM microgreens", self.conn)
        self.chefs = pd.read_sql("SELECT * FROM chefs", self.conn)
        self.purchase_history = pd.read_sql("SELECT * FROM purchase_history", self.conn)
        self.dish_pairings = pd.read_sql("SELECT * FROM dish_pairings", self.conn)
        self.seasonal_recommendations = pd.read_sql("SELECT * FROM seasonal_recommendations", self.conn)
        self.chef_preferences = pd.read_sql("SELECT * FROM chef_preferences", self.conn)
        
    def setup_advanced_models(self) -> None:
        """Configure TF-IDF and SVD models"""
        self.setup_tfidf_model()
        self.setup_svd_model()
        
    def setup_tfidf_model(self) -> None:
        """Set up TF-IDF model for flavor profile analysis"""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        flavor_profiles = self.microgreens['flavor_profile'].fillna('')
        self.flavor_vectors = self.vectorizer.fit_transform(flavor_profiles)
    
    def setup_svd_model(self) -> None:
        """Set up SVD model for collaborative filtering"""
        # Create chef-microgreen interaction matrix
        chef_ids = self.chefs['id'].unique()
        microgreen_ids = self.microgreens['id'].unique()
        
        # Create mappings between IDs and indices
        self.chef_idx_map = {chef_id: i for i, chef_id in enumerate(chef_ids)}
        self.microgreen_idx_map = {mg_id: i for i, mg_id in enumerate(microgreen_ids)}
        self.idx_microgreen_map = {i: mg_id for mg_id, i in self.microgreen_idx_map.items()}
        
        # Create interaction matrix (feedback scores/purchase counts)
        interaction_matrix = np.zeros((len(chef_ids), len(microgreen_ids)))
        
        for _, purchase in self.purchase_history.iterrows():
            chef_idx = self.chef_idx_map.get(purchase['chef_id'])
            mg_idx = self.microgreen_idx_map.get(purchase['microgreen_id'])
            
            if chef_idx is not None and mg_idx is not None:
                if pd.isna(purchase['feedback_score']):
                    interaction_matrix[chef_idx, mg_idx] += 1
                else:
                    interaction_matrix[chef_idx, mg_idx] = purchase['feedback_score']
        
        # Apply SVD model
        n_components = min(20, min(interaction_matrix.shape)-1)
        if n_components > 0:
            self.svd_model = TruncatedSVD(n_components=n_components)
            self.svd_features = self.svd_model.fit_transform(interaction_matrix)
        else:
            self.svd_model = None
            self.svd_features = None
        
    def get_current_season(self) -> str:
        """Return the current season based on the date"""
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def get_chef_cuisine_preferences(self, chef_id: int) -> Dict[str, float]:
        """
        Get a chef's cuisine type preferences and their weights
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping cuisine types to weight values
        """
        chef_data = self.chefs[self.chefs['id'] == chef_id]
        if chef_data.empty:
            return {}
            
        cuisine_type = chef_data['cuisine_type'].iloc[0]
        
        cuisine_prefs = self.chef_preferences[
            (self.chef_preferences['chef_id'] == chef_id) & 
            (self.chef_preferences['preference_type'] == 'cuisine')
        ]
        
        preferences = {cuisine_type: 1.0}
        
        for _, row in cuisine_prefs.iterrows():
            preferences[row['preference_value']] = row['preference_weight']
            
        return preferences
    
    def get_chef_flavor_preferences(self, chef_id: int) -> Dict[str, float]:
        """
        Get a chef's flavor preferences and their weights
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping flavor preferences to weight values
        """
        chef_data = self.chefs[self.chefs['id'] == chef_id]
        if chef_data.empty:
            return {}
            
        base_flavors = chef_data['preferred_flavors'].iloc[0].split(',')
        
        flavor_prefs = self.chef_preferences[
            (self.chef_preferences['chef_id'] == chef_id) & 
            (self.chef_preferences['preference_type'] == 'flavor')
        ]
        
        preferences = {flavor.strip(): 1.0 for flavor in base_flavors}
        
        for _, row in flavor_prefs.iterrows():
            preferences[row['preference_value']] = row['preference_weight']
            
        return preferences
    
    def calculate_purchase_history_score(self, chef_id: int) -> Dict[int, float]:
        """
        Calculate scores based on purchase history (RFM-like analysis)
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        chef_purchases = self.purchase_history[self.purchase_history['chef_id'] == chef_id]
        
        if chef_purchases.empty:
            return {}
        
        # Calculate Recency score
        current_date = datetime.datetime.now().date()
        chef_purchases['days_since_purchase'] = (
            current_date - pd.to_datetime(chef_purchases['purchase_date']).dt.date
        ).dt.days
        
        max_days = chef_purchases['days_since_purchase'].max()
        if max_days == 0:
            chef_purchases['recency_score'] = 1.0
        else:
            chef_purchases['recency_score'] = 1 - (chef_purchases['days_since_purchase'] / max_days)
        
        # Calculate Frequency score
        purchase_counts = chef_purchases.groupby('microgreen_id').size()
        max_count = purchase_counts.max() if not purchase_counts.empty else 1
        frequency_scores = purchase_counts / max_count
        
        # Calculate Feedback score
        feedback_avg = chef_purchases.groupby('microgreen_id')['feedback_score'].mean()
        max_feedback = feedback_avg.max() if not feedback_avg.empty else 5.0
        if max_feedback == 0:
            feedback_scores = feedback_avg.apply(lambda x: 0.5)
        else:
            feedback_scores = feedback_avg / max_feedback
            
        # Combine scores: 30% recency, 30% frequency, 40% feedback
        final_scores = {}
        
        for microgreen_id in chef_purchases['microgreen_id'].unique():
            microgreen_purchases = chef_purchases[chef_purchases['microgreen_id'] == microgreen_id]
            recency_score = microgreen_purchases['recency_score'].mean()
            
            freq_score = frequency_scores.get(microgreen_id, 0)
            feed_score = feedback_scores.get(microgreen_id, 0.5)
            
            final_scores[microgreen_id] = (
                0.3 * recency_score + 
                0.3 * freq_score + 
                0.4 * feed_score
            )
        
        return final_scores
    
    def calculate_cuisine_match_score(self, chef_id: int) -> Dict[int, float]:
        """
        Calculate cuisine type matching scores
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        cuisine_preferences = self.get_chef_cuisine_preferences(chef_id)
        
        if not cuisine_preferences:
            return {}
        
        scores = {}
        
        for cuisine_type, weight in cuisine_preferences.items():
            relevant_pairings = self.dish_pairings[
                self.dish_pairings['cuisine_type'].str.lower() == cuisine_type.lower()
            ]
            
            for _, pairing in relevant_pairings.iterrows():
                microgreen_id = pairing['microgreen_id']
                pairing_score = pairing['pairing_score'] / 5.0
                
                weighted_score = pairing_score * weight
                
                if microgreen_id in scores:
                    scores[microgreen_id] = max(scores[microgreen_id], weighted_score)
                else:
                    scores[microgreen_id] = weighted_score
        
        return scores
    
    def calculate_flavor_match_score(self, chef_id: int) -> Dict[int, float]:
        """
        Calculate flavor profile matching scores using basic string matching
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        flavor_preferences = self.get_chef_flavor_preferences(chef_id)
        
        if not flavor_preferences:
            return {}
        
        scores = {}
        
        for _, microgreen in self.microgreens.iterrows():
            microgreen_id = microgreen['id']
            flavor_profile = microgreen['flavor_profile'].lower()
            
            match_score = 0
            matches_found = 0
            
            for flavor, weight in flavor_preferences.items():
                if flavor.lower() in flavor_profile:
                    match_score += weight
                    matches_found += 1
            
            if matches_found > 0:
                scores[microgreen_id] = match_score / matches_found
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def calculate_flavor_match_score_tfidf(self, chef_id: int) -> Dict[int, float]:
        """
        Calculate flavor profile matching scores using TF-IDF
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        chef_data = self.chefs[self.chefs['id'] == chef_id]
        if chef_data.empty:
            return {}
            
        preferred_flavors = chef_data['preferred_flavors'].iloc[0]
        if not preferred_flavors:
            return {}
        
        chef_vector = self.vectorizer.transform([preferred_flavors])
        
        similarity_scores = cosine_similarity(chef_vector, self.flavor_vectors).flatten()
        
        return {mg_id: similarity_scores[i] for i, mg_id in enumerate(self.microgreens['id'])}
    
    def calculate_seasonal_score(self) -> Dict[int, float]:
        """
        Calculate seasonal relevance scores
        
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        current_season = self.get_current_season()
        
        season_recs = self.seasonal_recommendations[
            self.seasonal_recommendations['season'].str.lower() == current_season.lower()
        ]
        
        if season_recs.empty:
            return {}
        
        max_score = season_recs['recommendation_score'].max()
        if max_score == 0:
            return {row['microgreen_id']: 0.5 for _, row in season_recs.iterrows()}
        
        return {
            row['microgreen_id']: row['recommendation_score'] / max_score 
            for _, row in season_recs.iterrows()
        }
    
    def calculate_stock_availability_score(self) -> Dict[int, float]:
        """
        Calculate stock availability scores
        
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        if self.microgreens.empty:
            return {}
            
        max_stock = self.microgreens['stock_quantity'].max()
        
        if max_stock == 0:
            return {row['id']: 0 for _, row in self.microgreens.iterrows()}
            
        return {
            row['id']: min(1.0, row['stock_quantity'] / (max_stock * 0.5))
            for _, row in self.microgreens.iterrows()
        }
    
    def calculate_collaborative_filtering_score(self, chef_id: int) -> Dict[int, float]:
        """
        Calculate collaborative filtering scores using SVD
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary mapping microgreen IDs to scores
        """
        if self.svd_model is None or chef_id not in self.chef_idx_map:
            return {}
            
        chef_idx = self.chef_idx_map[chef_id]
        chef_vector = self.svd_features[chef_idx]
        
        predicted_ratings = np.dot(chef_vector, self.svd_model.components_)
        
        # Normalize scores
        min_rating = predicted_ratings.min()
        max_rating = predicted_ratings.max()
        if max_rating > min_rating:
            normalized_ratings = (predicted_ratings - min_rating) / (max_rating - min_rating)
        else:
            normalized_ratings = np.zeros_like(predicted_ratings)
            
        return {self.idx_microgreen_map[i]: score for i, score in enumerate(normalized_ratings)}
    
    def get_recommendations_for_chef(
        self, 
        chef_id: int, 
        num_recommendations: int = 5,
        include_recently_purchased: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate microgreen recommendations for a chef (traditional approach)
        
        Args:
            chef_id: Chef ID
            num_recommendations: Number of recommendations to return
            include_recently_purchased: Whether to include recently purchased items
            
        Returns:
            List of recommended microgreen information
        """
        purchase_history_scores = self.calculate_purchase_history_score(chef_id)
        cuisine_match_scores = self.calculate_cuisine_match_score(chef_id)
        flavor_match_scores = self.calculate_flavor_match_score(chef_id)
        seasonal_scores = self.calculate_seasonal_score()
        stock_scores = self.calculate_stock_availability_score()
        
        all_microgreen_ids = set(self.microgreens['id'])
        final_scores = {}
        
        for microgreen_id in all_microgreen_ids:
            score = (
                0.25 * purchase_history_scores.get(microgreen_id, 0.0) +
                0.25 * cuisine_match_scores.get(microgreen_id, 0.0) +
                0.25 * flavor_match_scores.get(microgreen_id, 0.0) +
                0.15 * seasonal_scores.get(microgreen_id, 0.0) +
                0.10 * stock_scores.get(microgreen_id, 0.0)
            )
            
            final_scores[microgreen_id] = score
        
        # Exclude recently purchased items if specified
        if not include_recently_purchased:
            recent_cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=30)).date()
            recent_purchases = set(
                self.purchase_history[
                    (self.purchase_history['chef_id'] == chef_id) & 
                    (pd.to_datetime(self.purchase_history['purchase_date']).dt.date >= recent_cutoff_date)
                ]['microgreen_id']
            )
            
            for microgreen_id in recent_purchases:
                if microgreen_id in final_scores:
                    del final_scores[microgreen_id]
        
        # Return all scores (for testing)
        if num_recommendations is None:
            return [
                {
                    'id': int(microgreen_id),
                    'score': score
                }
                for microgreen_id, score in final_scores.items()
            ]
        
        # Sort by score and select top recommendations
        sorted_recommendations = sorted(
            ((microgreen_id, score) for microgreen_id, score in final_scores.items()),
            key=lambda x: x[1],
            reverse=True
        )[:num_recommendations]
        
        # Return with detailed information
        recommendations = []
        for microgreen_id, score in sorted_recommendations:
            microgreen_data = self.microgreens[self.microgreens['id'] == microgreen_id].iloc[0]
            
            # Get relevant pairing information
            chef_cuisine = self.chefs[self.chefs['id'] == chef_id]['cuisine_type'].iloc[0]
            relevant_pairings = self.dish_pairings[
                (self.dish_pairings['microgreen_id'] == microgreen_id) & 
                (self.dish_pairings['cuisine_type'] == chef_cuisine)
            ]
            
            pairing_notes = []
            if not relevant_pairings.empty:
                for _, pairing in relevant_pairings.iterrows():
                    pairing_notes.append(f"{pairing['dish_type']}: {pairing['pairing_note']}")
            
            recommendations.append({
                'id': int(microgreen_id),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'score': round(score * 100) / 100,
                'flavor_profile': microgreen_data['flavor_profile'],
                'nutritional_value': microgreen_data['nutritional_value'],
                'stock_quantity': int(microgreen_data['stock_quantity']),
                'price': float(microgreen_data['price']),
                'season': microgreen_data['season'],
                'pairing_suggestions': pairing_notes,
                'image_url': microgreen_data['image_url']
            })
        
        # Save recommendations to database
        self.save_recommendations(chef_id, recommendations)
        
        return recommendations
    
    def get_enhanced_recommendations(
        self, 
        chef_id: int, 
        num_recommendations: int = 5,
        include_recently_purchased: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate enhanced microgreen recommendations using TF-IDF and SVD
        
        Args:
            chef_id: Chef ID
            num_recommendations: Number of recommendations to return
            include_recently_purchased: Whether to include recently purchased items
            
        Returns:
            List of recommended microgreen information
        """
        # Get traditional scores
        base_recommendations = self.get_recommendations_for_chef(
            chef_id, 
            num_recommendations=None,
            include_recently_purchased=include_recently_purchased
        )
        base_score_dict = {rec['id']: rec['score'] for rec in base_recommendations}
        
        # Get TF-IDF based flavor matching scores
        tfidf_scores = self.calculate_flavor_match_score_tfidf(chef_id)
        
        # Get SVD based collaborative filtering scores
        cf_scores = self.calculate_collaborative_filtering_score(chef_id)
        
        # All microgreen IDs
        all_microgreen_ids = set(self.microgreens['id'])
        
        # Calculate final scores (weighted average)
        final_scores = {}
        for mg_id in all_microgreen_ids:
            # Exclude recently purchased items if specified
            if not include_recently_purchased:
                recent_cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=30)).date()
                recent_purchases = set(
                    self.purchase_history[
                        (self.purchase_history['chef_id'] == chef_id) & 
                        (pd.to_datetime(self.purchase_history['purchase_date']).dt.date >= recent_cutoff_date)
                    ]['microgreen_id']
                )
                if mg_id in recent_purchases:
                    continue
            
            # Apply weights to combine scores
            score = (
                0.5 * base_score_dict.get(mg_id, 0) +
                0.3 * tfidf_scores.get(mg_id, 0) +
                0.2 * cf_scores.get(mg_id, 0)
            )
            final_scores[mg_id] = score
        
        # Sort by score and select top recommendations
        top_ids = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_recommendations]
        
        # Return with detailed information
        recommendations = []
        for microgreen_id, score in top_ids:
            microgreen_data = self.microgreens[self.microgreens['id'] == microgreen_id].iloc[0]
            
            # Get relevant pairing information
            chef_cuisine = self.chefs[self.chefs['id'] == chef_id]['cuisine_type'].iloc[0]
            relevant_pairings = self.dish_pairings[
                (self.dish_pairings['microgreen_id'] == microgreen_id) & 
                (self.dish_pairings['cuisine_type'] == chef_cuisine)
            ]
            
            pairing_notes = []
            if not relevant_pairings.empty:
                for _, pairing in relevant_pairings.iterrows():
                    pairing_notes.append(f"{pairing['dish_type']}: {pairing['pairing_note']}")
            
            recommendations.append({
                'id': int(microgreen_id),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'score': round(score * 100) / 100,
                'flavor_profile': microgreen_data['flavor_profile'],
                'nutritional_value': microgreen_data['nutritional_value'],
                'stock_quantity': int(microgreen_data['stock_quantity']),
                'price': float(microgreen_data['price']),
                'season': microgreen_data['season'],
                'pairing_suggestions': pairing_notes,
                'image_url': microgreen_data['image_url']
            })
        
        # Save recommendations to database
        self.save_recommendations(chef_id, recommendations)
        
        return recommendations
    
    def save_recommendations(self, chef_id: int, recommendations: List[Dict[str, Any]]) -> None:
        """
        Save generated recommendations to the database
        
        Args:
            chef_id: Chef ID
            recommendations: List of recommendation results
        """
        today = datetime.datetime.now().date()
        cursor = self.conn.cursor()
        
        for rec in recommendations:
            cursor.execute(
                """
                INSERT INTO recommendations
                (chef_id, microgreen_id, recommendation_date, recommendation_score, viewed, purchased)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chef_id, rec['id'], today, rec['score'], False, False)
            )
        
        self.conn.commit()
    
    def get_similar_microgreens(self, microgreen_id: int, num_similar: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar microgreens using TF-IDF flavor profile matching
        
        Args:
            microgreen_id: Reference microgreen ID
            num_similar: Number of similar products to return
            
        Returns:
            List of similar microgreen information
        """
        if microgreen_id not in set(self.microgreens['id']):
            return []
            
        # Find the reference microgreen index
        reference_idx = self.microgreens[self.microgreens['id'] == microgreen_id].index[0]
        
        # Get reference microgreen vector
        reference_vector = self.flavor_vectors[reference_idx]
        
        # Calculate cosine similarity with all microgreens
        similarity_scores = cosine_similarity(reference_vector, self.flavor_vectors).flatten()
        
        # Create index-score pairs (excluding the reference product)
        index_scores = [(i, similarity_scores[i]) for i in range(len(similarity_scores)) if i != reference_idx]
        
        # Sort by similarity and select top matches
        top_matches = sorted(index_scores, key=lambda x: x[1], reverse=True)[:num_similar]
        
        # Return with detailed information
        results = []
        for idx, similarity in top_matches:
            microgreen_data = self.microgreens.iloc[idx]
            results.append({
                'id': int(microgreen_data['id']),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'similarity_score': round(similarity * 100) / 100,
                'flavor_profile': microgreen_data['flavor_profile'],
                'season': microgreen_data['season'],
                'price': float(microgreen_data['price']),
                'stock_quantity': int(microgreen_data['stock_quantity']),
                'image_url': microgreen_data['image_url']
            })
            
        return results
    
    def get_trending_microgreens(self, days: int = 30, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find recently popular microgreens
        
        Args:
            days: Time period to consider (in days)
            limit: Number of trending products to return
            
        Returns:
            List of trending microgreen information
        """
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).date()
        
        # Get recent purchase data
        recent_purchases = self.purchase_history[
            pd.to_datetime(self.purchase_history['purchase_date']).dt.date >= cutoff_date
        ]
        
        if recent_purchases.empty:
            return []
        
        # Count purchases by microgreen
        purchase_counts = recent_purchases.groupby('microgreen_id').size()
        
        # Calculate average feedback score
        feedback_scores = recent_purchases.groupby('microgreen_id')['feedback_score'].mean()
        
        # Calculate trend score (70% purchase volume, 30% feedback)
        trending_scores = {}
        max_count = purchase_counts.max()
        
        for microgreen_id in purchase_counts.index:
            count_score = purchase_counts[microgreen_id] / max_count
            feedback_score = feedback_scores.get(microgreen_id, 0) / 5.0
            
            trending_scores[microgreen_id] = 0.7 * count_score + 0.3 * feedback_score
        
        # Select top trending products
        trending = sorted(
            trending_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        # Return with detailed information
        results = []
        for microgreen_id, score in trending:
            microgreen_data = self.microgreens[self.microgreens['id'] == microgreen_id].iloc[0]
            purchase_volume = purchase_counts[microgreen_id]
            
            results.append({
                'id': int(microgreen_id),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'trend_score': round(score * 100) / 100,
                'purchase_volume': int(purchase_volume),
                'average_feedback': round(feedback_scores.get(microgreen_id, 0), 1),
                'price': float(microgreen_data['price']),
                'stock_quantity': int(microgreen_data['stock_quantity']),
                'image_url': microgreen_data['image_url']
            })
            
        return results
    
    def get_seasonal_highlights(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend microgreens suitable for the current season
        
        Args:
            limit: Number of recommendations to return
            
        Returns:
            List of seasonal microgreen information
        """
        current_season = self.get_current_season()
        
        # Get current season recommendations
        season_recs = self.seasonal_recommendations[
            self.seasonal_recommendations['season'].str.lower() == current_season.lower()
        ]
        
        if season_recs.empty:
            return []
        
        # Sort by recommendation score
        top_seasonal = season_recs.sort_values(
            by='recommendation_score', 
            ascending=False
        ).head(limit)
        
        # Return with detailed information
        results = []
        for _, rec in top_seasonal.iterrows():
            microgreen_id = rec['microgreen_id']
            microgreen_data = self.microgreens[self.microgreens['id'] == microgreen_id].iloc[0]
            
            results.append({
                'id': int(microgreen_id),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'season': current_season,
                'recommendation_score': int(rec['recommendation_score']),
                'recommendation_note': rec['recommendation_note'],
                'flavor_profile': microgreen_data['flavor_profile'],
                'price': float(microgreen_data['price']),
                'stock_quantity': int(microgreen_data['stock_quantity']),
                'image_url': microgreen_data['image_url']
            })
            
        return results
    
    def get_chef_analytics(self, chef_id: int) -> Dict[str, Any]:
        """
        Analyze a chef's microgreen preferences and purchase patterns
        
        Args:
            chef_id: Chef ID
            
        Returns:
            Dictionary containing analysis data
        """
        chef_purchases = self.purchase_history[self.purchase_history['chef_id'] == chef_id]
        
        if chef_purchases.empty:
            return {
                'total_purchases': 0,
                'average_feedback': 0,
                'favorite_microgreens': [],
                'preferred_flavors': [],
                'purchase_trend': 'No data'
            }
        
        # Total purchases
        total_purchases = chef_purchases['quantity'].sum()
        
        # Average feedback score
        average_feedback = chef_purchases['feedback_score'].mean()
        
        # Favorite microgreens (based on purchase frequency and feedback)
        microgreen_stats = chef_purchases.groupby('microgreen_id').agg({
            'id': 'count',  # Purchase count
            'quantity': 'sum',  # Total quantity
            'feedback_score': 'mean'  # Average feedback
        }).reset_index()
        
        microgreen_stats['score'] = (
            0.4 * microgreen_stats['id'] / microgreen_stats['id'].max() +
            0.3 * microgreen_stats['quantity'] / microgreen_stats['quantity'].max() +
            0.3 * microgreen_stats['feedback_score'] / 5.0
        )
        
        top_microgreens = microgreen_stats.sort_values(by='score', ascending=False).head(3)
        
        # Detailed favorite microgreens information
        favorite_microgreens = []
        for _, row in top_microgreens.iterrows():
            microgreen_data = self.microgreens[self.microgreens['id'] == row['microgreen_id']].iloc[0]
            favorite_microgreens.append({
                'id': int(row['microgreen_id']),
                'name': microgreen_data['name'],
                'korean_name': microgreen_data['korean_name'],
                'purchase_count': int(row['id']),
                'total_quantity': int(row['quantity']),
                'average_feedback': round(row['feedback_score'], 1)
            })
        
        # Extract preferred flavor profile
        preferred_flavors = {}
        for fav in favorite_microgreens:
            microgreen_data = self.microgreens[self.microgreens['id'] == fav['id']].iloc[0]
            flavor_profile = microgreen_data['flavor_profile'].lower()
            
            # Simple tokenization to extract flavor keywords
            flavor_tokens = flavor_profile.split()
            for token in flavor_tokens:
                if token not in preferred_flavors:
                    preferred_flavors[token] = 0
                preferred_flavors[token] += 1
        
        # Top flavor preferences
        top_flavors = sorted(
            preferred_flavors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Purchase trend analysis (increasing, decreasing, stable)
        chef_purchases['purchase_date'] = pd.to_datetime(chef_purchases['purchase_date'])
        chef_purchases = chef_purchases.sort_values(by='purchase_date')
        
        if len(chef_purchases) >= 3:
            # Compare first and second half
            midpoint = len(chef_purchases) // 2
            first_half = chef_purchases.iloc[:midpoint]
            second_half = chef_purchases.iloc[midpoint:]
            
            first_half_volume = first_half['quantity'].sum() / len(first_half)
            second_half_volume = second_half['quantity'].sum() / len(second_half)
            
            percent_change = ((second_half_volume - first_half_volume) / first_half_volume) * 100
            
            if percent_change > 15:
                trend = 'Increasing'
            elif percent_change < -15:
                trend = 'Decreasing'
            else:
                trend = 'Stable'
        else:
            trend = 'Not enough data'
        
        return {
            'total_purchases': int(total_purchases),
            'average_feedback': round(average_feedback, 1),
            'favorite_microgreens': favorite_microgreens,
            'preferred_flavors': [flavor for flavor, _ in top_flavors],
            'purchase_trend': trend
        }
    
    def run_recommendation_comparison(self, chef_id: int, num_recommendations: int = 5) -> Dict[str, Any]:
        """
        Compare traditional and enhanced (TF-IDF + SVD) recommendation results
        
        Args:
            chef_id: Chef ID
            num_recommendations: Number of recommendations
            
        Returns:
            Comparison results of both recommendation approaches
        """
        traditional = self.get_recommendations_for_chef(chef_id, num_recommendations)
        enhanced = self.get_enhanced_recommendations(chef_id, num_recommendations)
        
        # Analyze differences
        trad_ids = set(item['id'] for item in traditional)
        enhanced_ids = set(item['id'] for item in enhanced)
        
        common_ids = trad_ids.intersection(enhanced_ids)
        different_ids = trad_ids.symmetric_difference(enhanced_ids)
        
        return {
            'traditional_recommendations': traditional,
            'enhanced_recommendations': enhanced,
            'comparison': {
                'common_items': len(common_ids),
                'different_items': len(different_ids),
                'similarity_percentage': round((len(common_ids) / num_recommendations) * 100)
            }
        }
    
    def close(self):
        """Close the database connection"""
        self.conn.close()