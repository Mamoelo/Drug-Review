"""
Prediction Service - Loads model and makes predictions
With Out-of-Distribution Detection
"""

import joblib
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_CONDITIONS = {'Depression', 'High Blood Pressure', 'Diabetes, Type 2'}


# ============================================================================
# PREDICTION SERVICE
# ============================================================================

class PredictionService:
    """Service for making condition predictions with learned OOD detection"""
    
    def __init__(self, model_path, encoder_path):
        # Get the absolute path to the project root
        project_root = Path(__file__).parent.parent.parent
        
        # Build absolute paths
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
        self.confidence_thresholds = self.ood_stats.get('confidence_thresholds', {
            'low': 0.3, 'medium': 0.5, 'high': 0.7
        })
        
        self._load_model()
    
    def _load_ood_statistics(self, project_root=None):
        """Load learned OOD statistics from training"""
        try:
            if project_root is None:
                project_root = Path(__file__).parent.parent.parent
            ood_path = project_root / 'models' / 'ood_statistics.json'
            print(f"[DEBUG] OOD path: {ood_path}")
            print(f"[DEBUG] OOD exists: {ood_path.exists()}")
            
            if ood_path.exists():
                with open(ood_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"⚠ OOD statistics not found at {ood_path}")
        except Exception as e:
            print(f"⚠ Could not load OOD statistics: {e}")
        return {}
    
    def _load_model(self):
        """Load the trained model and encoder"""
        try:
            print(f"[DEBUG] Checking model path: {self.model_path}")
            print(f"[DEBUG] Model exists: {self.model_path.exists()}")

            if not self.model_path.exists():
                print(f"⚠ Model not found at {self.model_path}")
                return

            print("[DEBUG] Loading model...")

            # Register custom classes under __main__ so pickle can resolve them
            import __main__
            from web.services.custom_transformers import (
                TextFeatureExtractor,
                SentimentFeatureExtractor,
                LearnedVocabularyExtractor
            )
            __main__.TextFeatureExtractor = TextFeatureExtractor
            __main__.SentimentFeatureExtractor = SentimentFeatureExtractor
            __main__.LearnedVocabularyExtractor = LearnedVocabularyExtractor

            # Use joblib — it handles numpy arrays inside pickles better
            self.model = joblib.load(str(self.model_path))
            print("✓ Model loaded successfully!")

            self.encoder = joblib.load(str(self.encoder_path))
            print(f"✓ Encoder loaded, classes: {self.encoder.classes_}")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def predict(self, text):
        """Predict patient condition from drug review text.
        Only classifies: Depression, High Blood Pressure, Diabetes Type 2.
        Rejects anything outside these three conditions.
        """

        # 1. Handle greetings/conversational inputs first
        conversational = self._handle_conversational(text)
        if conversational:
            return conversational

        # 2. Model must be available
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Please try again later.',
                'condition': None,
                'confidence': 0.0
            }

        # 3. Check vocabulary overlap with trained medical terms
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Get learned medical terms from OOD stats
        all_medical_terms = set(self.ood_stats.get('all_medical_terms', []))
        
        # Calculate overlap with trained medical vocabulary
        overlap = len(words.intersection(all_medical_terms))
        
        # If NO medical terms are recognized, reject immediately
        if overlap == 0:
            return {
                'success': True,
                'condition': 'Out of Scope',
                'confidence': 0.0,
                'confidence_level': 'N/A',
                'is_ood': True,
                'message': (
                    "This does not appear to relate to Depression, High Blood Pressure, "
                    "or Type 2 Diabetes — the only conditions this system is trained to classify."
                ),
                'sentiment': self._analyze_sentiment(text)
            }

        try:
            # 4. Get model probabilities across the 3 conditions
            probabilities = self.model.predict_proba([text])[0]
            prediction_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction_idx])
            condition = self.encoder.inverse_transform([prediction_idx])[0]

            all_probabilities = {
                cond: round(float(prob), 4)
                for cond, prob in zip(self.encoder.classes_, probabilities)
            }

            # 5. Hard check — predicted condition must be one of our 3
            if condition not in SUPPORTED_CONDITIONS:
                return {
                    'success': True,
                    'condition': 'Out of Scope',
                    'confidence': confidence,
                    'confidence_level': 'N/A',
                    'is_ood': True,
                    'message': (
                        "This review does not appear to relate to Depression, "
                        "High Blood Pressure, or Type 2 Diabetes — the only conditions "
                        "this system is trained to classify."
                    ),
                    'probabilities': all_probabilities,
                    'sentiment': self._analyze_sentiment(text)
                }

            # 6. Low confidence — in scope but unclear
            low_threshold = 0.25
            if confidence < low_threshold:
                return {
                    'success': True,
                    'condition': 'Uncertain',
                    'confidence': confidence,
                    'confidence_level': 'Very Low',
                    'is_ood': True,
                    'message': (
                        f"Closest match is '{condition}' but confidence is only "
                        f"{confidence*100:.0f}%. Please describe your symptoms in "
                        f"more detail for a reliable classification."
                    ),
                    'probabilities': all_probabilities,
                    'sentiment': self._analyze_sentiment(text)
                }

            # 7. Valid, confident, in-scope prediction
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
        """Handle conversational/greeting inputs"""
        text_lower = text.lower().strip()
        
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        thanks = ['thanks', 'thank you', 'thx', 'thank you so much', 'appreciate it']
        farewells = ['bye', 'goodbye', 'see you', 'take care', 'farewell']
        
        if text_lower in greetings:
            return {
                'success': True,
                'condition': 'Conversational',
                'confidence': 1.0,
                'confidence_level': 'High',
                'is_conversational': True,
                'probabilities': {},
                'sentiment': self._analyze_sentiment(text),
                'message': "Hello! I'm here to help with medication questions. Please describe any symptoms you're experiencing related to Depression, High Blood Pressure, or Type 2 Diabetes."
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
        
        if text_lower in farewells:
            return {
                'success': True,
                'condition': 'Conversational',
                'confidence': 1.0,
                'confidence_level': 'High',
                'is_conversational': True,
                'probabilities': {},
                'sentiment': self._analyze_sentiment(text),
                'message': "Take care! Remember to consult your doctor before making any medication changes."
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
        """Get confidence level based on learned thresholds"""
        if confidence >= 0.65:
            return 'High'
        elif confidence >= 0.45:
            return 'Moderate'
        elif confidence >= 0.25:
            return 'Low'
        else:
            return 'Very Low'