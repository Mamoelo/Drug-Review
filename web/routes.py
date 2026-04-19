"""
Application Routes - With Authentication, History, and Learning
"""

from flask import Blueprint, render_template, request, jsonify, current_app, session, redirect, url_for, flash
from web.services.predictor import PredictionService
from web.services.recommender import RecommendationService
from web.services.history_service import HistoryService
from web.services.auth_service import AuthService
from functools import wraps

main_bp = Blueprint('main', __name__)


def login_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to access this page.', 'error')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    return decorated_function


def get_predictor():
    if not hasattr(current_app, '_predictor'):
        current_app._predictor = PredictionService(
            current_app.config['MODEL_PATH'],
            current_app.config['ENCODER_PATH']
        )
    return current_app._predictor


def get_recommender():
    if not hasattr(current_app, '_recommender'):
        current_app._recommender = RecommendationService(
            current_app.config['DATA_PATH']
        )
    return current_app._recommender


def get_history_service():
    if not hasattr(current_app, '_history'):
        current_app._history = HistoryService()
    return current_app._history


def get_auth_service():
    if not hasattr(current_app, '_auth'):
        current_app._auth = AuthService()
    return current_app._auth


@main_bp.route('/')
def index():
    recommender = get_recommender()
    conditions = recommender.get_all_conditions()
    stats = {'conditions': len(conditions)}
    
    user = None
    if 'user_id' in session:
        auth = get_auth_service()
        user = auth.get_user_profile(session['user_id'])
    
    return render_template('index.html', stats=stats, conditions=conditions, user=user)


@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        auth = get_auth_service()
        result = auth.login(email, password)
        
        if result['success']:
            session['user_id'] = result['user_id']
            session['user_email'] = result['email']
            session['user_name'] = result.get('full_name', email.split('@')[0])
            flash('Welcome back!', 'success')
            return redirect(url_for('main.dashboard'))
        
        flash(result['error'], 'error')
        return render_template('login.html')
    
    return render_template('login.html')


@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender')
        
        auth = get_auth_service()
        result = auth.register(email, password, full_name, age, gender)
        
        if result['success']:
            flash('Account created successfully! Please sign in.', 'success')
            return redirect(url_for('main.login'))
        
        flash(result['error'], 'error')
        return render_template('register.html')
    
    return render_template('register.html')


@main_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'success')
    return redirect(url_for('main.index'))


@main_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    predictor = get_predictor()
    user_id = session['user_id']
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        
        if not symptoms:
            return render_template('predict.html', error='Please describe your symptoms')
        
        result = predictor.predict(symptoms)
        
        if result['success']:
            recommender = get_recommender()
            history_service = get_history_service()
            condition = result['condition']
            
            recommendations = []
            if condition in recommender.get_all_conditions():
                base_recommendations = recommender.recommend(condition)
                
                # Get personalized recommendations based on user history
                personalized = history_service.get_personalized_recommendations(
                    user_id, condition, base_recommendations
                )
                
                # Enhance with global feedback data
                for drug in personalized:
                    feedback_stats = history_service.get_drug_feedback_stats(condition, drug.get('name', drug.get('drug_name', '')))
                    if feedback_stats:
                        drug['global_feedback'] = feedback_stats
                    recommendations.append(drug)
            
            # Save consultation
            consultation_id = history_service.save_consultation(
                user_id, symptoms, condition, result['confidence'], recommendations
            )
            
            # Get user's previous conditions
            previous_conditions = history_service.get_previous_conditions(user_id)
            
            # Auto-schedule checkup
            checkup_scheduled = None
            if condition in ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']:
                history_service.schedule_checkup(user_id, consultation_id, days_from_now=30)
                checkup_scheduled = '30 days'
            
            session['prediction'] = {
                'consultation_id': consultation_id,
                'symptoms': symptoms,
                'condition': condition,
                'confidence': result['confidence'],
                'confidence_level': result.get('confidence_level', 'N/A'),
                'probabilities': result.get('probabilities', {}),
                'sentiment': result.get('sentiment', {}),
                'recommendations': recommendations,
                'message': result.get('message', ''),
                'is_ood': result.get('is_ood', False),
                'previous_conditions': previous_conditions,
                'checkup_scheduled': checkup_scheduled
            }
            
            return render_template('result.html', **session['prediction'])
        else:
            return render_template('predict.html', error=result.get('error', 'Prediction failed'))
    
    return render_template('predict.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    history_service = get_history_service()
    auth_service = get_auth_service()
    user_id = session['user_id']
    
    user = auth_service.get_user_profile(user_id)
    consultations = history_service.get_user_history(user_id, limit=10)
    checkups = history_service.get_upcoming_checkups(user_id)
    previous_conditions = history_service.get_previous_conditions(user_id)
    stats = history_service.get_user_stats(user_id)
    
    return render_template('dashboard.html',
                         user=user,
                         consultations=consultations,
                         checkups=checkups,
                         previous_conditions=previous_conditions,
                         stats=stats)


@main_bp.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    history_service = get_history_service()
    user_id = session['user_id']
    
    consultation_id = request.form.get('consultation_id')
    drug_name = request.form.get('drug_name')
    worked = request.form.get('worked') == 'true'
    effectiveness = request.form.get('effectiveness', type=int)
    side_effects = request.form.get('side_effects', '')
    notes = request.form.get('notes', '')
    
    if consultation_id and drug_name:
        history_service.save_feedback(
            int(consultation_id), user_id, drug_name, worked, 
            effectiveness, side_effects, notes
        )
        return jsonify({'success': True, 'message': 'Thank you for your feedback!'})
    
    return jsonify({'success': False, 'error': 'Missing required fields'}), 400


@main_bp.route('/history')
@login_required
def history():
    history_service = get_history_service()
    user_id = session['user_id']
    
    consultations = history_service.get_user_history(user_id)
    return render_template('history.html', consultations=consultations)


@main_bp.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    predictor = get_predictor()
    result = predictor.predict(data['text'])
    
    if result['success']:
        recommender = get_recommender()
        condition = result['condition']
        if condition in recommender.get_all_conditions():
            result['recommendations'] = recommender.recommend(condition)
        else:
            result['recommendations'] = []
    
    return jsonify(result)


@main_bp.route('/api/learn', methods=['POST'])
def trigger_learning():
    """Trigger model learning from all feedback"""
    history_service = get_history_service()
    learned_stats = history_service.learn_from_feedback()
    
    # Save learned stats to file
    import json
    from pathlib import Path
    
    learned_path = Path('models/learned_effectiveness.json')
    learned_path.parent.mkdir(exist_ok=True)
    with open(learned_path, 'w') as f:
        json.dump(learned_stats, f, indent=2)
    
    return jsonify({
        'success': True,
        'conditions_updated': len(learned_stats),
        'stats': learned_stats
    })


@main_bp.route('/api/schedule-checkup', methods=['POST'])
@login_required
def api_schedule_checkup():
    """API endpoint to schedule a checkup"""
    data = request.get_json()
    user_id = session['user_id']
    
    condition = data.get('condition')
    days = int(data.get('days', 30))
    notes = data.get('notes', '')
    
    history_service = get_history_service()
    checkup_id = history_service.schedule_checkup(user_id, None, days)
    
    return jsonify({'success': True, 'checkup_id': checkup_id})


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404