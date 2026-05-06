"""
User History and Feedback Service
With Collaborative Learning and Doctor Diagnosis Tracking
"""

import csv
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
        self.feedback_csv_path = Path(__file__).parent.parent.parent / 'data' / 'feedback_log.csv'
        self._init_tables()
        self._migrate()
        self._init_csv()

    def _init_tables(self):
        """Create all necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consultations (
                    consultation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    predicted_condition TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    recommendations TEXT,
                    user_diagnosis TEXT,
                    diagnosis_agreement TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')

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

    def _migrate(self):
        """Safe migration: add new columns to existing databases."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(consultations)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            for col, definition in [
                ('user_diagnosis',      'TEXT'),
                ('diagnosis_agreement', 'TEXT'),
            ]:
                if col not in existing_cols:
                    cursor.execute(f'ALTER TABLE consultations ADD COLUMN {col} {definition}')
                    print(f"[DB migration] Added column: consultations.{col}")
            conn.commit()

    def _init_csv(self):
        """Create feedback_log.csv with headers if it does not exist yet."""
        if not self.feedback_csv_path.exists():
            with open(self.feedback_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'user_id', 'consultation_id',
                    'condition', 'drug_name', 'worked',
                    'effectiveness_rating', 'side_effects', 'notes',
                ])
                writer.writeheader()

    # =========================================================================
    # BASIC CRUD OPERATIONS
    # =========================================================================

    def save_consultation(self, user_id, symptoms, condition, confidence,
                          recommendations, user_diagnosis=None,
                          diagnosis_agreement=None):
        """Save a consultation for an authenticated user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO consultations
                (user_id, symptoms, predicted_condition, confidence,
                 recommendations, user_diagnosis, diagnosis_agreement)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, symptoms, condition, confidence,
                json.dumps(recommendations),
                user_diagnosis,
                diagnosis_agreement
            ))
            conn.commit()
            return cursor.lastrowid

    def get_user_history(self, user_id, limit=20):
        """Get user's consultation history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT consultation_id, symptoms, predicted_condition, confidence,
                       recommendations, created_at, user_diagnosis, diagnosis_agreement
                FROM consultations
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))

            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append({
                    'consultation_id':    row[0],
                    'symptoms':           row[1],
                    'condition':          row[2],
                    'confidence':         row[3],
                    'recommendations':    json.loads(row[4]) if row[4] else [],
                    'created_at':         row[5],
                    'user_diagnosis':     row[6],
                    'diagnosis_agreement': row[7],
                })
            return history

    def get_previous_conditions(self, user_id):
        """Get user's previously predicted conditions."""
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
        """Save user feedback on a medication and mirror to feedback_log.csv."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback
                (consultation_id, user_id, drug_name, worked,
                 effectiveness_rating, side_effects, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (consultation_id, user_id, drug_name, worked,
                  effectiveness, side_effects, notes))
            conn.commit()

        condition = 'Unknown'
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT predicted_condition FROM consultations WHERE consultation_id = ?',
                    (consultation_id,)
                )
                row = cursor.fetchone()
                if row:
                    condition = row[0]
        except Exception:
            pass

        csv_row = {
            'timestamp':            datetime.now().isoformat(),
            'user_id':              user_id,
            'consultation_id':      consultation_id,
            'condition':            condition,
            'drug_name':            drug_name,
            'worked':               1 if worked else 0,
            'effectiveness_rating': effectiveness if effectiveness is not None else '',
            'side_effects':         side_effects or '',
            'notes':                notes or '',
        }
        with open(self.feedback_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
            writer.writerow(csv_row)

    def get_drug_feedback_stats(self, condition, drug_name):
        """Get aggregated feedback for a drug."""
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

    def complete_checkup(self, checkup_id, notes='', feeling=None,
                         severity=None, medication_adherence=None):
        """
        Mark a checkup as completed, store check-in data, then automatically
        schedule the next checkup based on how the patient is feeling.

        No follow-up is scheduled when:
            - feeling is 'better' or 'much_better' AND severity <= 4
              (patient is recovering well, no need to come back soon)

        Otherwise scheduling logic:
            medication stopped          → 3 days  (urgent)
            much_worse / new_symptoms  → 7 days
            worse                      → 10 days
            same                       → 14 days
            better                     → 21 days
            much_better                → 30 days

        Returns (new_checkup_id, days) if a next checkup was scheduled,
        or None if no follow-up is needed.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE checkups
                SET status = 'completed', notes = ?
                WHERE checkup_id = ?
            ''', (notes, checkup_id))

            cursor.execute(
                'SELECT user_id, consultation_id FROM checkups WHERE checkup_id = ?',
                (checkup_id,)
            )
            row = cursor.fetchone()
            conn.commit()

        if not row:
            return None

        user_id, consultation_id = row

        severity_int = int(severity) if severity is not None else None

        # Patient is doing well — no follow-up needed
        if feeling in ('better', 'much_better') and severity_int is not None and severity_int <= 4:
            return None

        # Medication stopped — urgent regardless of feeling
        if medication_adherence and 'stopped' in str(medication_adherence).lower():
            days = 3
        else:
            days_map = {
                'much_better':  30,
                'better':       21,
                'same':         14,
                'worse':        10,
                'much_worse':    7,
                'new_symptoms':  7,
            }
            days = days_map.get(feeling, 30)

        new_checkup_id = self.schedule_checkup(
            user_id=user_id,
            consultation_id=consultation_id,
            days_from_now=days,
        )
        return new_checkup_id, days

    def schedule_checkup(self, user_id, consultation_id=None, days_from_now=30):
        """Schedule a follow-up check-up."""
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
        """Get user's upcoming checkups."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT checkup_id, scheduled_date, status, notes, consultation_id
                FROM checkups
                WHERE user_id = ? AND status = 'pending'
                ORDER BY scheduled_date ASC
                LIMIT 5
            ''', (user_id,))

            return [{
                'checkup_id':      row[0],
                'scheduled_date':  row[1],
                'status':          row[2],
                'notes':           row[3],
                'consultation_id': row[4],
            } for row in cursor.fetchall()]

    def get_user_stats(self, user_id):
        """Get summary statistics for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                'SELECT COUNT(*) FROM consultations WHERE user_id = ?', (user_id,)
            )
            total_consultations = cursor.fetchone()[0]

            cursor.execute(
                'SELECT COUNT(DISTINCT predicted_condition) FROM consultations WHERE user_id = ?',
                (user_id,)
            )
            conditions_count = cursor.fetchone()[0]

            cursor.execute(
                'SELECT COUNT(*) FROM feedback WHERE user_id = ?', (user_id,)
            )
            feedback_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM checkups WHERE user_id = ? AND status = 'pending'",
                (user_id,)
            )
            pending_checkups = cursor.fetchone()[0]

            return {
                'total_consultations': total_consultations,
                'conditions_treated':  conditions_count,
                'feedback_given':      feedback_count,
                'pending_checkups':    pending_checkups,
            }

    # =========================================================================
    # LEARNING & PERSONALIZATION
    # =========================================================================

    def get_effective_drugs_for_user(self, user_id, condition):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.drug_name, f.effectiveness_rating, f.worked, c.created_at
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.user_id = ? AND c.predicted_condition = ? AND f.worked = 1
                ORDER BY c.created_at DESC
            ''', (user_id, condition))
            return [{'drug_name': r[0], 'effectiveness': r[1], 'created_at': r[3]}
                    for r in cursor.fetchall()]

    def get_ineffective_drugs_for_user(self, user_id, condition):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.drug_name, f.side_effects, c.created_at
                FROM feedback f
                JOIN consultations c ON f.consultation_id = c.consultation_id
                WHERE f.user_id = ? AND c.predicted_condition = ? AND f.worked = 0
                ORDER BY c.created_at DESC
            ''', (user_id, condition))
            return [{'drug_name': r[0], 'side_effects': r[1], 'created_at': r[2]}
                    for r in cursor.fetchall()]

    def get_similar_users(self, user_id, condition, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) FROM consultations WHERE user_id = ? AND predicted_condition = ?',
                (user_id, condition)
            )
            user_condition_count = cursor.fetchone()[0]

            cursor.execute('''
                SELECT DISTINCT c2.user_id, COUNT(*) as match_count
                FROM consultations c1
                JOIN consultations c2 ON c1.predicted_condition = c2.predicted_condition
                WHERE c1.user_id = ?
                  AND c2.user_id != ?
                  AND c2.predicted_condition = ?
                GROUP BY c2.user_id
                HAVING match_count >= ?
                ORDER BY match_count DESC
                LIMIT ?
            ''', (user_id, user_id, condition, min(2, user_condition_count), limit))
            return [row[0] for row in cursor.fetchall()]

    def get_collaborative_recommendations(self, user_id, condition, limit=5):
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
            return [{'drug_name': r[0], 'success_count': r[1], 'avg_rating': r[2]}
                    for r in cursor.fetchall()]

    def get_personalized_recommendations(self, user_id, condition,
                                          base_recommendations, limit=5):
        effective   = {d['drug_name']: d for d in self.get_effective_drugs_for_user(user_id, condition)}
        ineffective = {d['drug_name'] for d in self.get_ineffective_drugs_for_user(user_id, condition)}
        collab      = {d['drug_name']: d for d in self.get_collaborative_recommendations(user_id, condition)}

        personalized = []
        for drug in base_recommendations:
            drug_name = drug.get('name', drug.get('drug_name', ''))
            if not drug_name:
                continue

            score = drug.get('composite_score', drug.get('avg_rating', drug.get('rating', 5)))
            score = float(score) if isinstance(score, (int, float)) else 5.0

            if drug_name in effective:
                score *= 1.5
                drug['previously_worked'] = True
                drug['previous_effectiveness'] = effective[drug_name].get('effectiveness')

            if drug_name in ineffective:
                score *= 0.3
                drug['previously_ineffective'] = True

            if drug_name in collab:
                score *= 1.3
                drug['similar_users_success'] = collab[drug_name]['success_count']

            drug['personalized_score'] = round(score, 2)
            personalized.append(drug)

        personalized.sort(key=lambda x: x.get('personalized_score', 0), reverse=True)
        return personalized[:limit]

    def learn_from_feedback(self):
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
                condition, drug = row[0], row[1]
                if condition not in learned_stats:
                    learned_stats[condition] = {}
                learned_stats[condition][drug] = {
                    'total_feedback':       row[2],
                    'success_rate':         round((row[3] / row[2]) * 100, 1) if row[2] else 0,
                    'avg_rating':           round(row[4], 1) if row[4] else None,
                    'common_side_effects':  row[5].split(',') if row[5] else []
                }
            return learned_stats