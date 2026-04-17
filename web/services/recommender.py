"""
services/recommender.py
Recommendation Service - Recommends drugs based on condition
Learns everything from the training data - no hard-coded values
"""

import pandas as pd
import numpy as np
from pathlib import Path


class RecommendationService:
    """Service for drug recommendations - learns from data"""
    
    def __init__(self, data_path):
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        self.data_path = project_root / data_path
        
        print(f"[DEBUG] Recommender data path: {self.data_path}")
        print(f"[DEBUG] Data exists: {self.data_path.exists()}")
        
        self.drug_recommendations = {}
        self.drug_info = {}
        self.condition_stats = {}
        self._load_data()
    
    def _load_data(self):
        """Load and process drug data - learns everything from the dataset"""
        try:
            if self.data_path.exists():
                df = pd.read_csv(self.data_path)
                print(f"✓ Loaded {len(df):,} reviews from {self.data_path}")
                self._build_recommendations(df)
                self._calculate_condition_stats(df)
                print(f"✓ Built recommendations for {len(self.drug_recommendations)} conditions")
            else:
                print(f"⚠ Data not found at {self.data_path}")
                print("   Please ensure cleaned_train_data.csv exists in data/processed/")
                self.drug_recommendations = {}
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            self.drug_recommendations = {}
    
    def _build_recommendations(self, df):
        """
        Build drug recommendations from actual data
        No hard-coded drugs - everything learned from reviews
        """
        conditions = df['condition'].unique()
        
        for condition in conditions:
            condition_df = df[df['condition'] == condition]
            
            drug_stats = condition_df.groupby('drug_name').agg({
                'rating': ['mean', 'count', 'std'],
                'useful_count': 'sum'
            }).round(2)
            
            drug_stats.columns = ['avg_rating', 'review_count', 'rating_std', 'total_useful']
            drug_stats = drug_stats.reset_index()
            drug_stats = drug_stats[drug_stats['review_count'] >= 3]
            
            if len(drug_stats) == 0:
                print(f"   ⚠ No drugs with sufficient reviews for {condition}")
                continue
            
            drug_stats['effectiveness_score'] = (
                drug_stats['avg_rating'] * 0.4 +
                np.log1p(drug_stats['review_count']) * 0.3 +
                np.log1p(drug_stats['total_useful']) * 0.2 +
                (1 / (1 + drug_stats['rating_std'].fillna(0))) * 0.1
            )
            
            drug_stats = drug_stats.sort_values('effectiveness_score', ascending=False)
            
            self.drug_recommendations[condition] = []
            
            for _, drug in drug_stats.head(10).iterrows():
                drug_data = {
                    'name': drug['drug_name'],
                    'rating': float(drug['avg_rating']),
                    'reviews': int(drug['review_count']),
                    'total_useful': int(drug['total_useful']),
                    'effectiveness': self._calculate_effectiveness(
                        drug['avg_rating'], 
                        drug['review_count']
                    ),
                    'score': float(drug['effectiveness_score'])
                }
                self.drug_recommendations[condition].append(drug_data)
                
                self.drug_info[drug['drug_name']] = {
                    'avg_rating': float(drug['avg_rating']),
                    'review_count': int(drug['review_count']),
                    'conditions': condition
                }
            
            print(f"   ✓ {condition}: {len(self.drug_recommendations[condition])} drugs recommended")
    
    def _calculate_condition_stats(self, df):
        """Calculate overall statistics for each condition"""
        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]
            
            self.condition_stats[condition] = {
                'total_reviews': len(condition_df),
                'avg_rating': float(condition_df['rating'].mean()),
                'unique_drugs': condition_df['drug_name'].nunique(),
                'high_effectiveness_pct': float((condition_df['rating'] >= 8).mean() * 100)
            }
    
    def _calculate_effectiveness(self, rating, review_count):
        """Calculate effectiveness category based on rating and review count"""
        if rating >= 8.5 and review_count >= 10:
            return 'Excellent'
        elif rating >= 7.5:
            return 'Very Good'
        elif rating >= 6.5:
            return 'Good'
        elif rating >= 5.5:
            return 'Moderate'
        else:
            return 'Limited'
    
    def recommend(self, condition, limit=5):
        """Get drug recommendations for a condition"""
        recommendations = self.drug_recommendations.get(condition, [])
        
        if not recommendations:
            print(f"⚠ No recommendations available for {condition}")
            return []
        
        return recommendations[:limit]
    
    def get_all_conditions(self):
        """Get list of all conditions with recommendations"""
        return list(self.drug_recommendations.keys())
    
    def get_condition_stats(self, condition):
        """Get statistics for a specific condition"""
        return self.condition_stats.get(condition, {})
    
    def get_drug_info(self, drug_name):
        """Get information about a specific drug"""
        return self.drug_info.get(drug_name, {})
    
    def search_drugs(self, query, limit=10):
        """Search for drugs by name"""
        query = query.lower()
        results = []
        
        for drug_name, info in self.drug_info.items():
            if query in drug_name.lower():
                results.append({
                    'name': drug_name,
                    **info
                })
        
        return results[:limit]