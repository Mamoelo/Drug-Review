"""
User History and Feedback Service
"""

import uuid
import json
from datetime import datetime, timedelta
from web.models import Database


class HistoryService:
    """Service for managing user history, feedback, and check-ups"""
    
    def __init__(self):
        self.db = Database()
    
    def get_or_create_user(self, user_id=None):
        """Get existing user or create new one"""
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO users (user_id) VALUES (?)',
                (user_id,)
            )
            cursor.execute(
                'UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?',
                (user_id,)
            )
            conn.commit()
        
        return user_id
    
    def save_consultation(self, user_id, symptoms, condition, confidence, recommendations):
        """Save a consultation to history"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO consultations 
                (user_id, symptoms, predicted_condition, confidence, recommendations)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id, symptoms, condition, confidence,
                json.dumps(recommendations)
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_user_history(self, user_id, limit=10):
        """Get user's consultation history"""
        with self.db.get_connection() as conn:
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
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT predicted_condition
                FROM consultations
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            return [row[0] for row in cursor.fetchall()]
    
    def save_feedback(self, consultation_id, drug_name, worked, effectiveness=None, 
                      side_effects=None, notes=None):
        """Save user feedback on a medication"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback 
                (consultation_id, drug_name, worked, effectiveness_rating, side_effects, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(consultation_id, drug_name) DO UPDATE SET
                    worked = excluded.worked,
                    effectiveness_rating = excluded.effectiveness_rating,
                    side_effects = excluded.side_effects,
                    notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
            ''', (consultation_id, drug_name, worked, effectiveness, side_effects, notes))
            conn.commit()
    
    def get_drug_feedback_stats(self, condition, drug_name):
        """Get aggregated feedback for a drug"""
        with self.db.get_connection() as conn:
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
                    'success_rate': (row[1] / row[0] * 100) if row[1] else 0,
                    'avg_effectiveness': round(row[2], 1) if row[2] else None
                }
            return None
    
    def schedule_checkup(self, user_id, consultation_id=None, days_from_now=30):
        """Schedule a follow-up check-up"""
        scheduled_date = (datetime.now() + timedelta(days=days_from_now)).date()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO checkups (user_id, consultation_id, scheduled_date)
                VALUES (?, ?, ?)
            ''', (user_id, consultation_id, scheduled_date))
            conn.commit()
            return cursor.lastrowid
    
    def get_upcoming_checkups(self, user_id):
        """Get user's upcoming check-ups"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT checkup_id, scheduled_date, status, notes, 
                       c.symptoms, c.predicted_condition
                FROM checkups ch
                LEFT JOIN consultations c ON ch.consultation_id = c.consultation_id
                WHERE ch.user_id = ? AND ch.status = 'pending'
                ORDER BY ch.scheduled_date
            ''', (user_id))
            
            rows = cursor.fetchall()
            return [{
                'checkup_id': row[0],
                'scheduled_date': row[1],
                'status': row[2],
                'notes': row[3],
                'symptoms': row[4],
                'condition': row[5]
            } for row in rows]