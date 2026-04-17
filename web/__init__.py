"""
Flask Application Factory
"""

from flask import Flask
import os
from pathlib import Path


def create_app():
    """Create and configure Flask application"""
    
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MODEL_PATH'] = Path('models/tuned_pipeline.pkl')
    app.config['ENCODER_PATH'] = Path('models/label_encoder.pkl')
    app.config['DATA_PATH'] = Path('data/processed/cleaned_train_data.csv')
    
    from web.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app