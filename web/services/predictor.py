"""
Prediction Service - Loads model and makes predictions
With Out-of-Distribution Detection and Doctor Diagnosis Integration
"""

import joblib
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

SUPPORTED_CONDITIONS = {'Depression', 'High Blood Pressure', 'Diabetes, Type 2'}

# Mapping of what users might select → internal condition names
DIAGNOSIS_MAP = {
    'Depression': 'Depression',
    'High Blood Pressure': 'High Blood Pressure',
    'Diabetes, Type 2': 'Diabetes, Type 2',
    'none': None,
    '': None,
    None: None,
}


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

        self.ood_stats = self._load_ood_statistics(project_root)
        self.in_distribution_vocab = set(self.ood_stats.get('in_distribution_vocab', []))
        self.all_medical_terms = set(self.ood_stats.get('all_medical_terms', []))

        self.confidence_thresholds = self.ood_stats.get('confidence_thresholds', {
            'very_low': 0.15, 'low': 0.25, 'medium': 0.40, 'high': 0.55
        })

        self._load_model()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, text, user_diagnosis=None):
        """
        Predict patient condition from drug review text.

        Parameters
        ----------
        text : str
            The drug review / symptom description.
        user_diagnosis : str or None
            Optional condition the user's doctor has already diagnosed.
            One of: 'Depression', 'High Blood Pressure', 'Diabetes, Type 2', or None.

        Returns
        -------
        dict with keys: success, condition, confidence, confidence_level,
                        is_ood, probabilities, sentiment, [diagnosis_used,
                        diagnosis_agreement, diagnosis_note]
        """
        # 1. Normalise the diagnosis value
        user_diagnosis = DIAGNOSIS_MAP.get(user_diagnosis, user_diagnosis)
        if user_diagnosis not in SUPPORTED_CONDITIONS:
            user_diagnosis = None

        # 2. Conversational short-circuit
        conversational = self._handle_conversational(text)
        if conversational:
            return conversational

        # 3. Model must be loaded
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded.',
                'condition': None,
                'confidence': 0.0
            }

        # 4. Reject trivially short input
        if len(set(text.lower().split())) < 2:
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
            # 5. ML probabilities
            probabilities = self.model.predict_proba([text])[0]
            prediction_idx = int(np.argmax(probabilities))
            ml_confidence = float(probabilities[prediction_idx])
            ml_condition = self.encoder.inverse_transform([prediction_idx])[0]

            all_probabilities = {
                cond: round(float(prob), 4)
                for cond, prob in zip(self.encoder.classes_, probabilities)
            }

            # 6. If user supplied a diagnosis, integrate it
            if user_diagnosis:
                return self._predict_with_diagnosis(
                    text, user_diagnosis, ml_condition, ml_confidence,
                    probabilities, all_probabilities
                )

            # 7. Pure-ML path (no diagnosis supplied)
            return self._pure_ml_result(
                text, ml_condition, ml_confidence, probabilities, all_probabilities
            )

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'condition': None,
                'confidence': 0.0
            }

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSIS-AWARE DECISION ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_with_diagnosis(self, text, user_diagnosis, ml_condition,
                                 ml_confidence, probabilities, all_probabilities):
        """
        Integrate a doctor's diagnosis with the ML prediction.

        Rules
        ─────
        A  ML agrees with diagnosis AND confidence is high  → confirm strongly
        B  ML agrees with diagnosis AND confidence is low   → accept diagnosis
           (doctor's knowledge overrides weak ML signal)
        C  ML disagrees AND ML confidence is high           → surface conflict,
           show both, let user decide
        D  ML disagrees AND ML confidence is low            → trust the diagnosis
        """
        sentiment = self._analyze_sentiment(text)
        agrees = (ml_condition == user_diagnosis)

        # Probability the ML assigns to the doctor-diagnosed condition
        diagnosis_prob = float(all_probabilities.get(user_diagnosis, 0.0))

        very_low = self.confidence_thresholds.get('very_low', 0.15)
        high = self.confidence_thresholds.get('high', 0.55)

        # ── Case A: both agree, ML is confident ───────────────────────────
        if agrees and ml_confidence >= very_low:
            boosted_confidence = min(1.0, ml_confidence * 1.15 + 0.05)
            return {
                'success': True,
                'condition': ml_condition,
                'confidence': round(boosted_confidence, 4),
                'confidence_level': self._get_confidence_level(boosted_confidence),
                'is_ood': False,
                'probabilities': all_probabilities,
                'sentiment': sentiment,
                'diagnosis_used': user_diagnosis,
                'diagnosis_agreement': 'confirmed',
                'diagnosis_note': (
                    f"Your doctor's diagnosis of {user_diagnosis} aligns with the AI analysis. "
                    f"Confidence boosted to {round(boosted_confidence * 100)}%."
                )
            }

        # ── Case B: agree but ML confidence very low → trust diagnosis ────
        if agrees and ml_confidence < very_low:
            accepted_confidence = max(diagnosis_prob, very_low + 0.05, 0.35)
            return {
                'success': True,
                'condition': user_diagnosis,
                'confidence': round(accepted_confidence, 4),
                'confidence_level': self._get_confidence_level(accepted_confidence),
                'is_ood': False,
                'probabilities': all_probabilities,
                'sentiment': sentiment,
                'diagnosis_used': user_diagnosis,
                'diagnosis_agreement': 'accepted',
                'diagnosis_note': (
                    f"The AI signal for {user_diagnosis} was weak, but your doctor's "
                    f"diagnosis has been accepted. Results are based on your confirmed condition."
                )
            }

        # ── Case C: disagree AND ML is highly confident → flag conflict ───
        if not agrees and ml_confidence >= high:
            return {
                'success': True,
                'condition': ml_condition,
                'confidence': round(ml_confidence, 4),
                'confidence_level': self._get_confidence_level(ml_confidence),
                'is_ood': False,
                'probabilities': all_probabilities,
                'sentiment': sentiment,
                'diagnosis_used': user_diagnosis,
                'diagnosis_agreement': 'conflict',
                'diagnosis_note': (
                    f"Your doctor diagnosed {user_diagnosis}, but the AI is strongly "
                    f"predicting {ml_condition} ({round(ml_confidence * 100)}% confidence). "
                    f"Please consult your healthcare provider — the review may describe "
                    f"symptoms related to {ml_condition}."
                ),
                # Surface the doctor's condition too so the UI can show both
                'doctor_condition': user_diagnosis,
                'doctor_condition_prob': round(diagnosis_prob, 4),
            }

        # ── Case D: disagree AND ML is weak → trust the doctor ────────────
        accepted_confidence = max(diagnosis_prob + 0.1, 0.40)
        return {
            'success': True,
            'condition': user_diagnosis,
            'confidence': round(accepted_confidence, 4),
            'confidence_level': self._get_confidence_level(accepted_confidence),
            'is_ood': False,
            'probabilities': all_probabilities,
            'sentiment': sentiment,
            'diagnosis_used': user_diagnosis,
            'diagnosis_agreement': 'overridden',
            'diagnosis_note': (
                f"The AI leaned toward {ml_condition}, but given your doctor's diagnosis "
                f"of {user_diagnosis}, results are tailored to that condition."
            )
        }

    def _pure_ml_result(self, text, ml_condition, ml_confidence,
                         probabilities, all_probabilities):
        """Standard ML-only prediction path (no user diagnosis)."""
        sentiment = self._analyze_sentiment(text)

        if ml_condition not in SUPPORTED_CONDITIONS:
            return {
                'success': True,
                'condition': 'Out of Scope',
                'confidence': ml_confidence,
                'confidence_level': 'N/A',
                'is_ood': True,
                'message': (
                    f"This appears to relate to {ml_condition}, which is outside my trained scope. "
                    f"I specialise in Depression, High Blood Pressure, and Type 2 Diabetes."
                ),
                'probabilities': all_probabilities,
                'sentiment': sentiment
            }

        very_low = self.confidence_thresholds.get('very_low', 0.15)

        if ml_confidence < very_low:
            sorted_probs = sorted(
                enumerate(probabilities), key=lambda x: x[1], reverse=True
            )
            second_condition = self.encoder.inverse_transform([sorted_probs[1][0]])[0]
            second_conf = sorted_probs[1][1]
            if len(sorted_probs) > 1 and second_conf > ml_confidence * 0.8:
                message = (
                    f"Could be {ml_condition} or {second_condition}. "
                    f"Please describe your symptoms in more detail."
                )
            else:
                message = (
                    f"I'm not very confident, but this might relate to {ml_condition}. "
                    f"Could you provide more detail?"
                )
            return {
                'success': True,
                'condition': 'Uncertain',
                'confidence': ml_confidence,
                'confidence_level': 'Very Low',
                'is_ood': True,
                'message': message,
                'probabilities': all_probabilities,
                'sentiment': sentiment
            }

        return {
            'success': True,
            'condition': ml_condition,
            'confidence': ml_confidence,
            'confidence_level': self._get_confidence_level(ml_confidence),
            'is_ood': False,
            'probabilities': all_probabilities,
            'sentiment': sentiment
        }

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_ood_statistics(self, project_root=None):
        try:
            if project_root is None:
                project_root = Path(__file__).parent.parent.parent
            ood_path = project_root / 'models' / 'ood_statistics.json'
            if ood_path.exists():
                with open(ood_path, 'r') as f:
                    return json.load(f)
            else:
                print("⚠ OOD statistics not found, using defaults")
        except Exception as e:
            print(f"⚠ Could not load OOD statistics: {e}")
        return {}

    def _load_model(self):
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

    def _handle_conversational(self, text):
        text_lower = text.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        thanks = ['thanks', 'thank you', 'thx', 'thank you so much', 'appreciate it']

        if text_lower in greetings:
            return {
                'success': True, 'condition': 'Conversational', 'confidence': 1.0,
                'confidence_level': 'High', 'is_conversational': True,
                'probabilities': {}, 'sentiment': self._analyze_sentiment(text),
                'message': "Hello! I'm here to help with medication questions. Please describe your symptoms."
            }
        if text_lower in thanks:
            return {
                'success': True, 'condition': 'Conversational', 'confidence': 1.0,
                'confidence_level': 'High', 'is_conversational': True,
                'probabilities': {}, 'sentiment': self._analyze_sentiment(text),
                'message': "You're welcome! Is there anything else I can help you with?"
            }
        return None

    def _analyze_sentiment(self, text):
        s = self.sentiment_analyzer.polarity_scores(text)
        return {
            'compound': s['compound'], 'positive': s['pos'],
            'negative': s['neg'], 'neutral': s['neu'],
            'label': 'Positive' if s['compound'] >= 0.05 else (
                'Negative' if s['compound'] <= -0.05 else 'Neutral'
            )
        }

    def _get_confidence_level(self, confidence):
        t = self.confidence_thresholds
        if confidence >= t.get('high', 0.55):   return 'High'
        if confidence >= t.get('medium', 0.40): return 'Moderate'
        if confidence >= t.get('low', 0.25):    return 'Low'
        return 'Very Low'