"""
Database Models for User History and Feedback
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


class Database:
    """Simple SQLite database handler"""
    
    def __init__(self):
        db_path = Path(__file__).parent.parent / 'data' / 'user_history.db'
        db_path.parent.mkdir(exist_ok=True)
        self.db_path = str(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table (simple session-based, no login required)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Consultations (predictions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consultations (
                    consultation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symptoms TEXT NOT NULL,
                    predicted_condition TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    recommendations TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            
            # Feedback on recommendations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consultation_id INTEGER NOT NULL,
                    drug_name TEXT NOT NULL,
                    worked INTEGER,  -- 1 = worked, 0 = didn't work, NULL = no feedback yet
                    effectiveness_rating INTEGER,  -- 1-10
                    side_effects TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
                )
            ''')
            
            # Check-ups (follow-up appointments)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS checkups (
                    checkup_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    consultation_id INTEGER,
                    scheduled_date DATE,
                    status TEXT DEFAULT 'pending',  -- pending, completed, missed
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
                )
            ''')
            
            conn.commit()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)