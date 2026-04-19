"""
User History and Feedback Service
With Collaborative Learning
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path


class HistoryService:
    """Service for managing user history, feedback, and learning"""
    
    def __init__(self):
        db_path = Path(__file__).parent.parent.parent / 'data' / 'medai.db'
        db_path.parent.mkdir(exist_ok=True)
        self.db_path = str(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create all necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Consultations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consultations (
                    consultation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    predicted_condition TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    recommendations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consultation_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    drug_name TEXT NOT NULL,
                    worked INTEGER,
                    effectiveness_rating INTEGER,
                    side_effects TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # Checkups table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkups (
                    checkup_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    consultation_id INTEGER,
                    scheduled_date DATE,
                    status TEXT DEFAULT 'pending',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
                )
            ''')
            
            conn.commit()
    
    # =========================================================================
    # BASIC CRUD OPERATIONS
    # =========================================================================
    
    def save_consultation(self, user_id, symptoms, condition, confidence, recommendations):
        """Save a consultation for authenticated user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO consultations 
                (user_id, symptoms, predicted_condition, confidence, recommendations)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, symptoms, condition, confidence, json.dumps(recommendations)))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_history(self, user_id, limit=20):
        """Get user's consultation history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT consultation_id, symptoms, predicted_condition, confidence, 
                       recommendations, created_at
                FROM consultations
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    'consultation_id': row[0],
                    'symptoms': row[1],
                    'condition': row[2],
                    'confidence': row[3],
                    'recommendations': json.loads(row[4]) if row[4] else [],
                    'created_at': row[5]
                })
            return history
    
    def get_previous_conditions(self, user_id):
        """Get user's previously diagnosed conditions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT predicted_condition, COUNT(*) as count, MAX(created_at)
                FROM consultations
                WHERE user_id = ?
                GROUP BY predicted_condition
                ORDER BY MAX(created_at) DESC
            ''', (user_id,))
            
            return [{'condition': row[0], 'count': row[1]} for row in cursor.fetchall()]
    
    def save_feedback(self, consultation_id, user_id, drug_name, worked, 
                      effectiveness=None, side_effects=None, notes=None):
        """Save user feedback on a medication"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback 
                (consultation_id, user_id, drug_name, worked, effectiveness_rating, side_effects, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (consultation_id, user_id, drug_name, worked, effectiveness, side_effects, notes))
            conn.commit()
    
    def get_drug_feedback_stats(self, condition, drug_name):
        """Get aggregated feedback for a drug"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN f.worked = 1 THEN 1 ELSE 0 END) as worked_count,
                    AVG(f.effectiveness_rating) as avg_effectiveness
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE c.predicted_condition = ? AND f.drug_name = ?
            ''', (condition, drug_name))
            
            row = cursor.fetchone()
            if row and row[0] > 0:
                return {
                    'total': row[0],
                    'worked_count': row[1] or 0,
                    'success_rate': round((row[1] / row[0] * 100) if row[1] else 0, 1),
                    'avg_effectiveness': round(row[2], 1) if row[2] else None
                }
            return None
    
    def schedule_checkup(self, user_id, consultation_id=None, days_from_now=30):
        """Schedule a follow-up check-up"""
        scheduled_date = (datetime.now() + timedelta(days=days_from_now)).date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO checkups (user_id, consultation_id, scheduled_date)
                VALUES (?, ?, ?)
            ''', (user_id, consultation_id, scheduled_date))
            conn.commit()
            return cursor.lastrowid
    
    def get_upcoming_checkups(self, user_id):
        """Get user's upcoming check-ups"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT checkup_id, scheduled_date, status, notes, 
                       c.symptoms, c.predicted_condition
                FROM checkups ch
                LEFT JOIN consultations c ON ch.consultation_id = c.consultation_id
                WHERE ch.user_id = ? AND ch.status = 'pending'
                ORDER BY ch.scheduled_date
            ''', (user_id,))
            
            rows = cursor.fetchall()
            return [{
                'checkup_id': row[0],
                'scheduled_date': row[1],
                'status': row[2],
                'notes': row[3],
                'symptoms': row[4],
                'condition': row[5]
            } for row in rows]
    
    def get_user_stats(self, user_id):
        """Get user statistics for personalized dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM consultations WHERE user_id = ?', (user_id,))
            total_consultations = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT predicted_condition) FROM consultations WHERE user_id = ?', (user_id,))
            conditions_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE user_id = ?', (user_id,))
            feedback_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM checkups WHERE user_id = ? AND status = "pending"', (user_id,))
            pending_checkups = cursor.fetchone()[0]
            
            return {
                'total_consultations': total_consultations,
                'conditions_treated': conditions_count,
                'feedback_given': feedback_count,
                'pending_checkups': pending_checkups
            }
    
    # =========================================================================
    # LEARNING & PERSONALIZATION METHODS
    # =========================================================================
    
    def get_effective_drugs_for_user(self, user_id, condition):
        """Get drugs that worked for this specific user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.drug_name, f.effectiveness_rating, f.worked, c.created_at
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.user_id = ? AND c.predicted_condition = ? AND f.worked = 1
                ORDER BY c.created_at DESC
            ''', (user_id, condition))
            
            rows = cursor.fetchall()
            return [{'drug_name': row[0], 'effectiveness': row[1], 'created_at': row[3]} for row in rows]
    
    def get_ineffective_drugs_for_user(self, user_id, condition):
        """Get drugs that didn't work for this user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.drug_name, f.side_effects, c.created_at
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.user_id = ? AND c.predicted_condition = ? AND f.worked = 0
                ORDER BY c.created_at DESC
            ''', (user_id, condition))
            
            rows = cursor.fetchall()
            return [{'drug_name': row[0], 'side_effects': row[1], 'created_at': row[2]} for row in rows]
    
    def get_similar_users(self, user_id, condition, limit=10):
        """Find users with similar condition history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM consultations 
                WHERE user_id = ? AND predicted_condition = ?
            ''', (user_id, condition))
            user_condition_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT DISTINCT u2.user_id, COUNT(*) as match_count
                FROM consultations c1
                JOIN consultations c2 ON c1.predicted_condition = c2.predicted_condition
                JOIN users u2 ON c2.user_id = u2.user_id
                WHERE c1.user_id = ? 
                AND c2.user_id != ?
                AND c2.predicted_condition = ?
                GROUP BY u2.user_id
                HAVING match_count >= ?
                ORDER BY match_count DESC
                LIMIT ?
            ''', (user_id, user_id, condition, min(2, user_condition_count), limit))
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_collaborative_recommendations(self, user_id, condition, limit=5):
        """Get drug recommendations based on similar users' success"""
        similar_users = self.get_similar_users(user_id, condition)
        
        if not similar_users:
            return []
        
        placeholders = ','.join(['?'] * len(similar_users))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT f.drug_name, 
                       COUNT(*) as success_count,
                       AVG(f.effectiveness_rating) as avg_rating
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.user_id IN ({placeholders})
                AND c.predicted_condition = ?
                AND f.worked = 1
                GROUP BY f.drug_name
                HAVING success_count >= 2
                ORDER BY avg_rating DESC, success_count DESC
                LIMIT ?
            ''', (*similar_users, condition, limit))
            
            rows = cursor.fetchall()
            return [{'drug_name': row[0], 'success_count': row[1], 'avg_rating': row[2]} for row in rows]
    
    def get_personalized_recommendations(self, user_id, condition, base_recommendations, limit=5):
        """
        Generate personalized recommendations by:
        1. Prioritizing drugs that worked for this user before
        2. Excluding drugs that caused side effects
        3. Boosting drugs that worked for similar users
        """
        
        effective_drugs = {d['drug_name']: d for d in self.get_effective_drugs_for_user(user_id, condition)}
        ineffective_drugs = {d['drug_name'] for d in self.get_ineffective_drugs_for_user(user_id, condition)}
        collaborative_drugs = {d['drug_name']: d for d in self.get_collaborative_recommendations(user_id, condition)}
        
        personalized = []
        
        for drug in base_recommendations:
            drug_name = drug.get('name', drug.get('drug_name', ''))
            if not drug_name:
                continue
                
            score = drug.get('composite_score', drug.get('avg_rating', drug.get('rating', 5)))
            if isinstance(score, (int, float)):
                score = float(score)
            else:
                score = 5.0
            
            # Boost score if it worked for this user before
            if drug_name in effective_drugs:
                score *= 1.5
                drug['previously_worked'] = True
                drug['previous_effectiveness'] = effective_drugs[drug_name].get('effectiveness')
            
            # Penalize if it didn't work or caused side effects
            if drug_name in ineffective_drugs:
                score *= 0.3
                drug['previously_ineffective'] = True
            
            # Boost if it worked for similar users
            if drug_name in collaborative_drugs:
                score *= 1.3
                drug['similar_users_success'] = collaborative_drugs[drug_name]['success_count']
            
            drug['personalized_score'] = round(score, 2)
            personalized.append(drug)
        
        personalized.sort(key=lambda x: x.get('personalized_score', 0), reverse=True)
        return personalized[:limit]
    
    def learn_from_feedback(self):
        """Update global drug effectiveness based on all user feedback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.predicted_condition, f.drug_name,
                       COUNT(*) as total_feedback,
                       SUM(CASE WHEN f.worked = 1 THEN 1 ELSE 0 END) as worked_count,
                       AVG(f.effectiveness_rating) as avg_rating,
                       GROUP_CONCAT(DISTINCT f.side_effects) as common_side_effects
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.worked IS NOT NULL
                GROUP BY c.predicted_condition, f.drug_name
                HAVING total_feedback >= 3
            ''')
            
            learned_stats = {}
            for row in cursor.fetchall():
                condition = row[0]
                drug = row[1]
                
                if condition not in learned_stats:
                    learned_stats[condition] = {}
                
                learned_stats[condition][drug] = {
                    'total_feedback': row[2],
                    'success_rate': round((row[3] / row[2]) * 100, 1) if row[2] else 0,
                    'avg_rating': round(row[4], 1) if row[4] else None,
                    'common_side_effects': row[5].split(',') if row[5] else []
                }
            
            return learned_stats