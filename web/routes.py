"""
Application Routes - No hard-coded values
"""

from flask import Blueprint, render_template, request, jsonify, current_app, session
from web.services.predictor import PredictionService
from web.services.recommender import RecommendationService

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


@main_bp.route('/')
def index():
    recommender = get_recommender()
    conditions = recommender.get_all_conditions()
    
    stats = {
        'conditions': len(conditions)
    }
    
    return render_template('index.html', stats=stats, conditions=conditions)


@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    predictor = get_predictor()
    
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
                recommendations = recommender.recommend(condition)
            
            session['prediction'] = {
                'symptoms': symptoms,
                'condition': condition,
                'confidence': result['confidence'],
                'confidence_level': result.get('confidence_level', 'N/A'),
                'probabilities': result.get('probabilities', {}),
                'sentiment': result.get('sentiment', {}),
                'recommendations': recommendations,
                'message': result.get('message', ''),
                'is_ood': result.get('is_ood', False)
            }
            
            return render_template('result.html', **session['prediction'])
        else:
            return render_template('predict.html', error=result.get('error', 'Prediction failed'))
    
    return render_template('predict.html')


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