"""
Application Routes - With User History and Feedback
"""

from flask import Blueprint, render_template, request, jsonify, current_app, session
from web.services.predictor import PredictionService
from web.services.recommender import RecommendationService
from web.services.history_service import HistoryService
import uuid

main_bp = Blueprint('main', __name__)


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


def get_user_id():
    """Get or create user ID from session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


@main_bp.route('/')
def index():
    recommender = get_recommender()
    conditions = recommender.get_all_conditions()
    
    stats = {'conditions': len(conditions)}
    return render_template('index.html', stats=stats, conditions=conditions)


@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    predictor = get_predictor()
    history_service = get_history_service()
    user_id = get_user_id()
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        
        if not symptoms:
            return render_template('predict.html', error='Please describe your symptoms')
        
        result = predictor.predict(symptoms)
        
        if result['success']:
            recommender = get_recommender()
            condition = result['condition']
            
            recommendations = []
            if condition in recommender.get_all_conditions():
                base_recommendations = recommender.recommend(condition)
                
                # Enhance recommendations with user feedback data
                for drug in base_recommendations:
                    feedback_stats = history_service.get_drug_feedback_stats(condition, drug['name'])
                    if feedback_stats:
                        drug['feedback'] = feedback_stats
                    recommendations.append(drug)
            
            # Save consultation to history
            consultation_id = history_service.save_consultation(
                user_id, symptoms, condition, result['confidence'], recommendations
            )
            
            # Get user's previous conditions for context
            previous_conditions = history_service.get_previous_conditions(user_id)
            
            # Auto-schedule checkup for certain conditions
            checkup_scheduled = None
            if condition in ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']:
                checkup_id = history_service.schedule_checkup(user_id, consultation_id, days_from_now=30)
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


@main_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on medication effectiveness"""
    history_service = get_history_service()
    
    consultation_id = request.form.get('consultation_id')
    drug_name = request.form.get('drug_name')
    worked = request.form.get('worked') == 'true'
    effectiveness = request.form.get('effectiveness', type=int)
    side_effects = request.form.get('side_effects', '')
    notes = request.form.get('notes', '')
    
    if consultation_id and drug_name:
        history_service.save_feedback(
            int(consultation_id), drug_name, worked, effectiveness, side_effects, notes
        )
        return jsonify({'success': True, 'message': 'Thank you for your feedback!'})
    
    return jsonify({'success': False, 'error': 'Missing required fields'}), 400


@main_bp.route('/history')
def history():
    """View user's consultation history"""
    history_service = get_history_service()
    user_id = get_user_id()
    
    consultations = history_service.get_user_history(user_id)
    checkups = history_service.get_upcoming_checkups(user_id)
    
    return render_template('history.html', 
                         consultations=consultations, 
                         checkups=checkups)


@main_bp.route('/dashboard')
def dashboard():
    """User dashboard with history and checkups"""
    history_service = get_history_service()
    user_id = get_user_id()
    
    consultations = history_service.get_user_history(user_id, limit=5)
    checkups = history_service.get_upcoming_checkups(user_id)
    previous_conditions = history_service.get_previous_conditions(user_id)
    
    return render_template('dashboard.html',
                         consultations=consultations,
                         checkups=checkups,
                         previous_conditions=previous_conditions)


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


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404