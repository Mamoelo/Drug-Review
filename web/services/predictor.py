"""
Prediction Service - Loads model and makes predictions
With Out-of-Distribution Detection
"""

import joblib
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# Only these are constants - they define the business objective
SUPPORTED_CONDITIONS = {'Depression', 'High Blood Pressure', 'Diabetes, Type 2'}


class PredictionService:
    """Service for making condition predictions with learned OOD detection"""
    
    def __init__(self, model_path, encoder_path):
        project_root = Path(__file__).parent.parent.parent
        
        self.model_path = project_root / model_path
        self.encoder_path = project_root / encoder_path
        
        print(f"[DEBUG] Project root: {project_root}")
        print(f"[DEBUG] Model path: {self.model_path}")
        print(f"[DEBUG] Model exists: {self.model_path.exists()}")
        
        self.model = None
        self.encoder = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load learned statistics
        self.ood_stats = self._load_ood_statistics(project_root)
        self.in_distribution_vocab = set(self.ood_stats.get('in_distribution_vocab', []))
        self.all_medical_terms = set(self.ood_stats.get('all_medical_terms', []))
        
        # Get confidence thresholds from learned data
        self.confidence_thresholds = self.ood_stats.get('confidence_thresholds', {
            'very_low': 0.15, 'low': 0.25, 'medium': 0.40, 'high': 0.55
        })
        
        self._load_model()
    
    def _load_ood_statistics(self, project_root=None):
        """Load learned OOD statistics from training"""
        try:
            if project_root is None:
                project_root = Path(__file__).parent.parent.parent
            ood_path = project_root / 'models' / 'ood_statistics.json'
            
            if ood_path.exists():
                with open(ood_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"⚠ OOD statistics not found, using defaults")
        except Exception as e:
            print(f"⚠ Could not load OOD statistics: {e}")
        return {}
    
    def _load_model(self):
        """Load the trained model and encoder"""
        try:
            if not self.model_path.exists():
                print(f"⚠ Model not found at {self.model_path}")
                return

            import __main__
            from web.services.custom_transformers import (
                TextFeatureExtractor,
                SentimentFeatureExtractor,
                LearnedVocabularyExtractor
            )
            __main__.TextFeatureExtractor = TextFeatureExtractor
            __main__.SentimentFeatureExtractor = SentimentFeatureExtractor
            __main__.LearnedVocabularyExtractor = LearnedVocabularyExtractor

            self.model = joblib.load(str(self.model_path))
            self.encoder = joblib.load(str(self.encoder_path))
            print(f"✓ Model loaded, classes: {self.encoder.classes_}")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def predict(self, text):
        """Predict patient condition from drug review text."""

        # 1. Handle conversational inputs
        conversational = self._handle_conversational(text)
        if conversational:
            return conversational

        # 2. Model must be available
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded.',
                'condition': None,
                'confidence': 0.0
            }

        text_lower = text.lower()
        words = set(text_lower.split())
        
        # 3. Check if input is completely empty or nonsensical
        if len(words) < 2:
            return {
                'success': True,
                'condition': 'Invalid Input',
                'confidence': 0.0,
                'confidence_level': 'N/A',
                'is_ood': True,
                'message': 'Please provide more details about your symptoms.',
                'sentiment': self._analyze_sentiment(text)
            }

        try:
            # 4. Get model probabilities
            probabilities = self.model.predict_proba([text])[0]
            prediction_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction_idx])
            condition = self.encoder.inverse_transform([prediction_idx])[0]

            all_probabilities = {
                cond: round(float(prob), 4)
                for cond, prob in zip(self.encoder.classes_, probabilities)
            }

            # 5. Check if predicted condition is supported
            if condition not in SUPPORTED_CONDITIONS:
                return {
                    'success': True,
                    'condition': 'Out of Scope',
                    'confidence': confidence,
                    'confidence_level': 'N/A',
                    'is_ood': True,
                    'message': f"This appears to relate to {condition}, which is not in my trained scope. I specialize in Depression, High Blood Pressure, and Type 2 Diabetes.",
                    'probabilities': all_probabilities,
                    'sentiment': self._analyze_sentiment(text)
                }

            # 6. Check confidence - MUCH MORE LENIENT
            very_low_threshold = self.confidence_thresholds.get('very_low', 0.15)
            
            if confidence < very_low_threshold:
                # Check if there's a clear second choice
                sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
                if len(sorted_probs) > 1:
                    second_condition = self.encoder.inverse_transform([sorted_probs[1][0]])[0]
                    second_confidence = sorted_probs[1][1]
                    
                    if second_confidence > confidence * 0.8:
                        return {
                            'success': True,
                            'condition': 'Uncertain',
                            'confidence': confidence,
                            'confidence_level': 'Very Low',
                            'is_ood': True,
                            'message': f"Could be {condition} or {second_condition}. Please provide more details about your symptoms.",
                            'probabilities': all_probabilities,
                            'sentiment': self._analyze_sentiment(text)
                        }
                
                return {
                    'success': True,
                    'condition': 'Uncertain',
                    'confidence': confidence,
                    'confidence_level': 'Very Low',
                    'is_ood': True,
                    'message': f"I'm not very confident, but this might relate to {condition}. Could you describe your symptoms in more detail?",
                    'probabilities': all_probabilities,
                    'sentiment': self._analyze_sentiment(text)
                }

            # 7. Valid prediction - ACCEPT IT
            return {
                'success': True,
                'condition': condition,
                'confidence': confidence,
                'confidence_level': self._get_confidence_level(confidence),
                'is_ood': False,
                'probabilities': all_probabilities,
                'sentiment': self._analyze_sentiment(text)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'condition': None,
                'confidence': 0.0
            }
    
    def _handle_conversational(self, text):
        """Handle conversational inputs"""
        text_lower = text.lower().strip()
        
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        thanks = ['thanks', 'thank you', 'thx', 'thank you so much', 'appreciate it']
        
        if text_lower in greetings:
            return {
                'success': True,
                'condition': 'Conversational',
                'confidence': 1.0,
                'confidence_level': 'High',
                'is_conversational': True,
                'probabilities': {},
                'sentiment': self._analyze_sentiment(text),
                'message': "Hello! I'm here to help with medication questions. Please describe any symptoms you're experiencing."
            }
        
        if text_lower in thanks:
            return {
                'success': True,
                'condition': 'Conversational',
                'confidence': 1.0,
                'confidence_level': 'High',
                'is_conversational': True,
                'probabilities': {},
                'sentiment': self._analyze_sentiment(text),
                'message': "You're welcome! Is there anything else I can help you with?"
            }
        
        return None
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        return {
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'label': 'Positive' if sentiment['compound'] >= 0.05 else (
                'Negative' if sentiment['compound'] <= -0.05 else 'Neutral'
            )
        }
    
    def _get_confidence_level(self, confidence):
        """Get confidence level using learned thresholds"""
        thresholds = self.confidence_thresholds
        if confidence >= thresholds.get('high', 0.55):
            return 'High'
        elif confidence >= thresholds.get('medium', 0.40):
            return 'Moderate'
        elif confidence >= thresholds.get('low', 0.25):
            return 'Low'
        else:
            return 'Very Low'