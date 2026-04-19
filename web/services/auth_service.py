"""
Authentication Service - User management
"""

import sqlite3
import hashlib
import uuid
from datetime import datetime
from pathlib import Path


class AuthService:
    """Service for user authentication and management"""
    
    def __init__(self):
        db_path = Path(__file__).parent.parent.parent / 'data' / 'medai.db'
        db_path.parent.mkdir(exist_ok=True)
        self.db_path = str(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create users table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    full_name TEXT,
                    password_hash TEXT,
                    age INTEGER,
                    gender TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
            # Add sessions table for "remember me"
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            ''')
            conn.commit()
    
    def _hash_password(self, password):
        """Hash password with salt"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, email, password, full_name=None, age=None, gender=None):
        """Register a new user"""
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (user_id, email, full_name, password_hash, age, gender)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, email, full_name, password_hash, age, gender))
                conn.commit()
                return {'success': True, 'user_id': user_id}
        except sqlite3.IntegrityError:
            return {'success': False, 'error': 'Email already registered'}
    
    def login(self, email, password):
        """Authenticate user"""
        password_hash = self._hash_password(password)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, email, full_name FROM users
                WHERE email = ? AND password_hash = ?
            ''', (email, password_hash))
            
            row = cursor.fetchone()
            if row:
                # Update last login
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?', (row[0],))
                conn.commit()
                
                # Create session
                session_token = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO sessions (session_token, user_id, expires_at)
                    VALUES (?, ?, datetime('now', '+30 days'))
                ''', (session_token, row[0]))
                conn.commit()
                
                return {
                    'success': True,
                    'user_id': row[0],
                    'email': row[1],
                    'full_name': row[2],
                    'session_token': session_token
                }
            
            return {'success': False, 'error': 'Invalid email or password'}
    
    def logout(self, session_token):
        """Logout user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
            conn.commit()
    
    def validate_session(self, session_token):
        """Validate session token and return user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT u.user_id, u.email, u.full_name
                FROM sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'email': row[1],
                    'full_name': row[2]
                }
            return None
    
    def get_user_profile(self, user_id):
        """Get user profile"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, email, full_name, age, gender, created_at, last_login
                FROM users WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'email': row[1],
                    'full_name': row[2],
                    'age': row[3],
                    'gender': row[4],
                    'created_at': row[5],
                    'last_login': row[6]
                }
            return None