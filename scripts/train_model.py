"""
Task 3 & 4: Model Determination and Training
File: scripts/train_model.py
Purpose: Train an intelligent model that learns everything from data
         No hard-coded values - all patterns learned from training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import pickle
import sys
from datetime import datetime
from collections import defaultdict, Counter

# Add project root to path so we can import from web.services
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import custom transformers from the stable module
from web.services.custom_transformers import (
    TextFeatureExtractor,
    SentimentFeatureExtractor,
    LearnedVocabularyExtractor
)

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Text Processing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data silently
nltk_data_required = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for data in nltk_data_required:
    try:
        if data == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif data == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif data == 'omw-1.4':
            nltk.data.find('corpora/omw-1.4')
        else:
            nltk.data.find(f'corpora/{data}')
    except LookupError:
        nltk.download(data, quiet=True)

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. OUT-OF-DISTRIBUTION DETECTOR
# ============================================================================

class OutOfDistributionDetector:
    """Detects if input is within learned distribution"""
    
    def __init__(self):
        self.vectorizer = None
        self.distribution_stats = {}
        self.confidence_threshold = 0.3
        self.learned_vocabulary = set()
        self.learned_bigrams = set()
        self.avg_word_count = 0
        self.min_word_count = 2
        
    def fit(self, X_train, y_train):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words='english', ngram_range=(1, 2), min_df=3
        )
        X_vec = self.vectorizer.fit_transform(X_train)
        
        all_text = ' '.join(X_train).lower()
        words = all_text.split()
        word_freq = Counter(words)
        
        self.learned_vocabulary = set([
            word for word, count in word_freq.items() 
            if count >= 5 and len(word) > 2 and word.isalpha()
        ])
        
        bigram_counter = Counter()
        for i in range(len(words) - 1):
            bigram_counter[f"{words[i]}_{words[i+1]}"] += 1
        self.learned_bigrams = set([bg for bg, _ in bigram_counter.most_common(500)])
        
        word_counts = [len(str(text).split()) for text in X_train]
        self.avg_word_count = np.mean(word_counts)
        self.min_word_count = max(2, int(np.percentile(word_counts, 5)))
        
        tfidf_sums = np.array(X_vec.sum(axis=1)).flatten()
        nonzero_counts = np.array((X_vec > 0).sum(axis=1)).flatten()
        
        self.distribution_stats = {
            'mean_tfidf_sum': float(np.mean(tfidf_sums)),
            'std_tfidf_sum': float(np.std(tfidf_sums)) if len(tfidf_sums) > 1 else 1.0,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'training_samples': len(X_train)
        }
        return self
    
    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold
    
    def _calculate_vocabulary_overlap(self, text):
        text_lower = text.lower()
        words = set(text_lower.split())
        unigram_overlap = len(words.intersection(self.learned_vocabulary))
        unigram_ratio = unigram_overlap / max(len(words), 1)
        
        text_words = text_lower.split()
        text_bigrams = set()
        for i in range(len(text_words) - 1):
            text_bigrams.add(f"{text_words[i]}_{text_words[i+1]}")
        
        bigram_overlap = len(text_bigrams.intersection(self.learned_bigrams))
        bigram_ratio = bigram_overlap / max(len(text_bigrams), 1)
        
        return unigram_overlap, unigram_ratio, bigram_overlap, bigram_ratio
    
    def is_in_distribution(self, text):
        if self.vectorizer is None:
            return True, 1.0, "Detector not fitted"
        
        word_count = len(text.split())
        if word_count < self.min_word_count:
            return False, 0.0, f"Please provide more details (at least {self.min_word_count} words)."
        
        unigram_overlap, unigram_ratio, bigram_overlap, bigram_ratio = self._calculate_vocabulary_overlap(text)
        
        if unigram_overlap >= 2 or bigram_overlap >= 1 or unigram_ratio >= 0.1:
            confidence = min(0.95, 0.4 + unigram_ratio + bigram_ratio * 2)
            return True, confidence, "Input matches learned medical vocabulary."
        
        text_vec = self.vectorizer.transform([text])
        tfidf_sum = float(text_vec.sum())
        nonzero_count = int((text_vec > 0).sum())
        
        mean_sum = self.distribution_stats['mean_tfidf_sum']
        std_sum = max(self.distribution_stats['std_tfidf_sum'], 1)
        sum_zscore = abs(tfidf_sum - mean_sum) / std_sum
        
        if tfidf_sum < mean_sum * 0.15:
            return False, 0.1, "I don't recognize enough medical context in your input."
        
        vocab_overlap = nonzero_count / max(word_count, 1)
        confidence = min(0.8, vocab_overlap * (1 / (1 + sum_zscore)) + unigram_ratio)
        
        if confidence < 0.2:
            return False, confidence, "I'm not confident this is within my learned domain."
        
        return True, max(confidence, 0.25), "Input matches learned patterns."


# ============================================================================
# 2. INTELLIGENT PREDICTOR
# ============================================================================

class IntelligentDrugAdvisor:
    """Main predictor that learns everything from data"""
    
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.ood_detector = OutOfDistributionDetector()
        self.drug_recommendations = {}
        self.condition_info = {}
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.conversational_indicators = set()
        
    def _learn_conversational_patterns(self, X_train):
        starters = []
        for text in X_train:
            text = str(text).lower().strip()
            if text:
                words = text.split()
                if len(words) >= 1:
                    starters.append(words[0])
        
        starter_freq = Counter(starters)
        self.conversational_indicators = set([word for word, _ in starter_freq.most_common(20)])
    
    def fit(self, df_train, df_test=None):
        print("\n" + "=" * 80)
        print("TRAINING INTELLIGENT DRUG ADVISOR")
        print("=" * 80)
        
        X_train = df_train['review'].fillna('').values
        y_train = self.label_encoder.fit_transform(df_train['condition'])
        
        print(f"\n[1] Training data: {len(X_train):,} reviews")
        print(f"    Conditions: {self.label_encoder.classes_.tolist()}")
        
        print("\n[2] Learning conversational patterns...")
        self._learn_conversational_patterns(X_train)
        
        print("\n[3] Learning data distribution patterns...")
        self.ood_detector.fit(X_train, y_train)
        print(f"    - Learned vocabulary: {len(self.ood_detector.learned_vocabulary):,} terms")
        
        print("\n[4] Building feature extraction pipeline...")
        
        tfidf = TfidfVectorizer(
            max_features=2000, stop_words='english', ngram_range=(1, 2),
            min_df=3, max_df=0.85, sublinear_tf=True
        )
        
        feature_union = FeatureUnion([
            ('tfidf', tfidf),
            ('text_stats', TextFeatureExtractor()),
            ('sentiment', SentimentFeatureExtractor()),
            ('learned_vocab', LearnedVocabularyExtractor(max_features_per_class=50))
        ])
        
        print("\n[5] Building ensemble classifiers...")
        
        base_models = [
            ('lr', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
        ]
        
        ensemble = VotingClassifier(estimators=base_models, voting='soft', weights=[1, 2, 3, 1])
        
        self.pipeline = Pipeline([
            ('features', feature_union),
            ('scaler', MaxAbsScaler()),
            ('classifier', ensemble)
        ])
        
        print("\n[6] Training ensemble model...")
        self.pipeline.fit(X_train, y_train)
        print("    ✓ Training complete")
        
        if df_test is not None:
            X_test = df_test['review'].fillna('').values
            y_pred_proba = self.pipeline.predict_proba(X_test)
            threshold = np.percentile(y_pred_proba.max(axis=1), 15)
            self.ood_detector.set_confidence_threshold(threshold)
            print(f"    - Confidence threshold: {threshold:.3f}")
        
        print("\n[7] Learning drug recommendations...")
        self._learn_drug_recommendations(df_train)
        
        print("\n[8] Extracting condition patterns...")
        self._learn_condition_patterns(df_train)
        
        return self
    
    def _learn_drug_recommendations(self, df):
        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]
            drug_stats = condition_df.groupby('drug_name').agg({
                'rating': ['mean', 'count', 'std'],
                'useful_count': 'sum'
            }).round(2)
            
            drug_stats.columns = ['avg_rating', 'review_count', 'rating_std', 'total_useful']
            drug_stats = drug_stats.reset_index()
            drug_stats = drug_stats[drug_stats['review_count'] >= 3]
            
            if len(drug_stats) == 0:
                continue
            
            drug_stats['composite_score'] = (
                drug_stats['avg_rating'] * 0.5 +
                np.log1p(drug_stats['review_count']) * 2 +
                np.log1p(drug_stats['total_useful']) * 0.3 -
                drug_stats['rating_std'].fillna(0) * 0.2
            )
            
            top_drugs = drug_stats.nlargest(10, 'composite_score')
            self.drug_recommendations[condition] = top_drugs[[
                'drug_name', 'avg_rating', 'review_count', 'composite_score'
            ]].to_dict('records')
        
        print(f"    - Learned recommendations for {len(self.drug_recommendations)} conditions")
    
    def _learn_condition_patterns(self, df):
        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]
            high_rated = condition_df[condition_df['rating'] >= 7]['review'].tolist()
            
            if not high_rated:
                continue
            
            all_words = ' '.join(high_rated).lower().split()
            word_freq = Counter([w for w in all_words if len(w) > 3 and w.isalpha()])
            
            self.condition_info[condition] = {
                'sample_count': len(condition_df),
                'avg_rating': float(condition_df['rating'].mean()),
                'common_terms': [word for word, _ in word_freq.most_common(15)],
                'top_drugs': [d['drug_name'] for d in self.drug_recommendations.get(condition, [])[:5]]
            }
        
        print(f"    - Extracted patterns for {len(self.condition_info)} conditions")
    
    def _is_conversational(self, text):
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        if not words:
            return False, None
        
        first_word = words[0]
        if first_word in self.conversational_indicators:
            if first_word in ['hi', 'hello', 'hey']:
                return True, 'greeting'
        
        if 'thank' in text_lower or 'thanks' in text_lower:
            return True, 'gratitude'
        
        if any(word in text_lower for word in ['bye', 'goodbye', 'farewell']):
            return True, 'farewell'
        
        return False, None
    
    def predict(self, text):
        is_conv, conv_type = self._is_conversational(text)
        if is_conv:
            return self._create_conversational_response(conv_type)
        
        is_valid, confidence, reason = self.ood_detector.is_in_distribution(text)
        
        if not is_valid:
            return {
                'success': False,
                'condition': 'Out of Distribution',
                'confidence': float(confidence) if confidence else 0.0,
                'is_ood': True,
                'ood_reason': reason,
                'message': reason,
                'suggestion': "Try describing symptoms related to Depression, High Blood Pressure, or Type 2 Diabetes."
            }
        
        try:
            probabilities = self.pipeline.predict_proba([text])[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            condition = self.label_encoder.inverse_transform([prediction_idx])[0]
            
            threshold = self.ood_detector.confidence_threshold or 0.3
            if confidence < threshold:
                return {
                    'success': True,
                    'condition': condition,
                    'confidence': float(confidence),
                    'is_low_confidence': True,
                    'probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                    'message': f"I'm not entirely confident. Could you provide more details?",
                    'recommendations': self.drug_recommendations.get(condition, [])[:3]
                }
            
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            return {
                'success': True,
                'condition': condition,
                'confidence': float(confidence),
                'confidence_level': self._get_confidence_level(confidence),
                'probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                'sentiment': {
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                    'label': 'Positive' if sentiment['compound'] >= 0.05 else ('Negative' if sentiment['compound'] <= -0.05 else 'Neutral')
                },
                'recommendations': self.drug_recommendations.get(condition, [])[:5]
            }
            
        except Exception as e:
            return {
                'success': False,
                'condition': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _create_conversational_response(self, conv_type):
        responses = {
            'greeting': "Hello! I'm here to help with medication questions. Please describe any symptoms you're experiencing.",
            'gratitude': "You're welcome! Is there anything else I can help you with?",
            'farewell': "Take care! Remember to consult your doctor before making any medication changes."
        }
        return {
            'success': True,
            'condition': 'Conversational',
            'confidence': 1.0,
            'is_conversational': True,
            'message': responses.get(conv_type, responses['greeting'])
        }
    
    def _get_confidence_level(self, confidence):
        if confidence >= 0.7:
            return 'High'
        elif confidence >= 0.5:
            return 'Moderate'
        elif confidence >= 0.3:
            return 'Low'
        else:
            return 'Very Low'


# ============================================================================
# 3. MODEL EVALUATION
# ============================================================================

def evaluate_model(predictor, df_test):
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    X_test = df_test['review'].fillna('').values
    y_test = predictor.label_encoder.transform(df_test['condition'])
    
    y_pred = predictor.pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    
    print("\nPer-Class Metrics:")
    print(classification_report(y_test, y_pred, target_names=predictor.label_encoder.classes_, digits=4))
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}


# ============================================================================
# 4. OOD STATISTICS COMPUTATION
# ============================================================================

# ============================================================================
# 4. OOD STATISTICS COMPUTATION
# ============================================================================

def compute_ood_statistics(predictor, df_train):
    print("\n" + "=" * 80)
    print("COMPUTING OUT-OF-DISTRIBUTION STATISTICS")
    print("=" * 80)
    
    from scipy.spatial.distance import cosine
    
    print("\n[1] Building in-distribution vocabulary...")
    
    all_reviews = ' '.join(df_train['review'].fillna('').astype(str).tolist()).lower()
    words = all_reviews.split()
    filtered_words = [w for w in words if w.isalpha() and len(w) > 2]
    word_freq = Counter(filtered_words)
    
    in_distribution_vocab = set([word for word, count in word_freq.items() if count >= 5])
    print(f"    - In-distribution vocabulary: {len(in_distribution_vocab):,} terms")
    
    # ========================================================================
    # NEW: Extract condition-specific terms from training data
    # ========================================================================
    print("\n[2] Extracting condition-specific medical terms...")
    condition_terms = {}
    
    for condition in predictor.label_encoder.classes_:
        print(f"    Processing: {condition}")
        
        # Get all reviews for this condition
        condition_df = df_train[df_train['condition'] == condition]
        condition_reviews = ' '.join(condition_df['review'].fillna('').astype(str).tolist()).lower()
        condition_words = condition_reviews.split()
        
        # Filter words: alphabetic, length > 2
        condition_filtered = [w for w in condition_words if w.isalpha() and len(w) > 2]
        condition_freq = Counter(condition_filtered)
        
        # Get stop words to exclude
        from nltk.corpus import stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        # Filter out stop words and get top distinctive terms
        distinctive_terms = []
        for word, count in condition_freq.most_common(100):
            if word not in stop_words and count >= 10:
                distinctive_terms.append(word)
        
        # Keep top 50 terms
        top_terms = distinctive_terms[:50]
        condition_terms[condition] = top_terms
        
        print(f"        - Learned {len(top_terms)} distinctive terms")
        print(f"        - Examples: {', '.join(top_terms[:10])}")
    
    # Combine all condition terms as medical indicators
    all_medical_terms = set()
    for terms in condition_terms.values():
        all_medical_terms.update(terms)
    
    print(f"\n    ✓ Total medical indicators learned: {len(all_medical_terms):,}")
    # ========================================================================
    
    print("\n[3] Computing TF-IDF statistics...")
    
    tfidf_ood = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2), min_df=2)
    X_train_text = df_train['review'].fillna('').astype(str).values
    y_train_text = df_train['condition'].values
    X_tfidf = tfidf_ood.fit_transform(X_train_text)
    
    condition_stats = {}
    for condition in predictor.label_encoder.classes_:
        print(f"    Processing: {condition}")
        mask = y_train_text == condition
        condition_vectors = X_tfidf[mask]
        
        if condition_vectors.shape[0] > 0:
            centroid = condition_vectors.mean(axis=0)
            
            if hasattr(centroid, 'toarray'):
                centroid_dense = centroid.toarray().flatten()
            elif hasattr(centroid, 'A'):
                centroid_dense = centroid.A.flatten()
            else:
                centroid_dense = np.array(centroid).flatten()
            
            distances = []
            sample_limit = min(condition_vectors.shape[0], 1000)
            
            for i in range(sample_limit):
                vec = condition_vectors[i]
                if vec.nnz > 0:
                    if hasattr(vec, 'toarray'):
                        vec_dense = vec.toarray().flatten()
                    elif hasattr(vec, 'A'):
                        vec_dense = vec.A.flatten()
                    else:
                        vec_dense = np.array(vec).flatten()
                    
                    dist = cosine(vec_dense, centroid_dense)
                    distances.append(dist)
            
            if distances:
                condition_stats[condition] = {
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'max_distance': float(np.max(distances)),
                    'sample_count': int(condition_vectors.shape[0])
                }
            else:
                condition_stats[condition] = {
                    'mean_distance': 0.5,
                    'std_distance': 0.1,
                    'max_distance': 1.0,
                    'sample_count': int(condition_vectors.shape[0])
                }
        else:
            condition_stats[condition] = {
                'mean_distance': 0.5,
                'std_distance': 0.1,
                'max_distance': 1.0,
                'sample_count': 0
            }
    
    print("\n[4] Computing confidence thresholds...")
    
    y_pred_proba = predictor.pipeline.predict_proba(X_train_text)
    max_probas = y_pred_proba.max(axis=1)
    
    confidence_thresholds = {
        'very_low': float(np.percentile(max_probas, 5)),
        'low': float(np.percentile(max_probas, 15)),
        'medium': float(np.percentile(max_probas, 35)),
        'high': float(np.percentile(max_probas, 65))
    }
    
    print(f"    - Very Low threshold: {confidence_thresholds['very_low']:.3f}")
    print(f"    - Low threshold: {confidence_thresholds['low']:.3f}")
    print(f"    - Medium threshold: {confidence_thresholds['medium']:.3f}")
    print(f"    - High threshold: {confidence_thresholds['high']:.3f}")
    
    print("\n[5] Computing text statistics...")
    
    review_lengths = [len(str(r).split()) for r in X_train_text]
    text_stats = {
        'avg_review_length': float(np.mean(review_lengths)),
        'std_review_length': float(np.std(review_lengths)),
        'min_review_length': int(np.min(review_lengths)),
        'max_review_length': int(np.max(review_lengths)),
        'percentile_5': int(np.percentile(review_lengths, 5)),
        'percentile_10': int(np.percentile(review_lengths, 10))
    }
    
    print(f"    - Average review length: {text_stats['avg_review_length']:.1f} words")
    print(f"    - Minimum recommended: {text_stats['percentile_10']} words")
    
    # ========================================================================
    # Save all statistics
    # ========================================================================
    ood_stats = {
        'in_distribution_vocab': list(in_distribution_vocab),
        'vocab_size': len(in_distribution_vocab),
        'condition_stats': condition_stats,
        'confidence_thresholds': confidence_thresholds,
        'text_stats': text_stats,
        'tfidf_feature_names': tfidf_ood.get_feature_names_out().tolist(),
        'training_samples': len(df_train),
        'conditions': predictor.label_encoder.classes_.tolist(),
        'created_date': datetime.now().isoformat(),
        # NEW: Learned medical terms
        'condition_terms': condition_terms,
        'all_medical_terms': list(all_medical_terms)
    }
    
    ood_path = Path('models/ood_statistics.json')
    with open(ood_path, 'w') as f:
        json.dump(ood_stats, f, indent=2)
    
    print(f"\n    ✓ OOD statistics saved to: {ood_path}")
    print(f"    ✓ File includes:")
    print(f"        - In-distribution vocabulary: {len(in_distribution_vocab):,} terms")
    print(f"        - Medical indicators: {len(all_medical_terms):,} terms")
    print(f"        - Condition statistics for {len(condition_stats)} conditions")
    print(f"        - Confidence thresholds")
    print(f"        - Text statistics")
    
    return ood_stats


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("INTELLIGENT DRUG ADVISOR - MODEL TRAINING")
    print("=" * 80)
    
    train_path = Path('data/processed/cleaned_train_data.csv')
    test_path = Path('data/processed/cleaned_test_data.csv')
    
    if not train_path.exists():
        print("\nError: Could not find training data!")
        return
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"\n[1] Loaded data:")
    print(f"    Training: {df_train.shape[0]:,} reviews")
    print(f"    Test: {df_test.shape[0]:,} reviews")
    print(f"    Conditions: {df_train['condition'].unique().tolist()}")
    
    predictor = IntelligentDrugAdvisor()
    predictor.fit(df_train, df_test)
    
    metrics = evaluate_model(predictor, df_test)
    
    # Compute OOD statistics
    compute_ood_statistics(predictor, df_train)
    
    # Test predictions
    print("\n" + "=" * 80)
    print("TESTING PREDICTOR")
    print("=" * 80)
    
    test_inputs = [
        "I have been feeling very sad and hopeless for weeks, no energy",
        "My blood pressure is consistently high around 150/95",
        "I was diagnosed with type 2 diabetes and need medication",
        "Hello there!",
        "Thank you so much!",
        "sad",
    ]
    
    for text in test_inputs:
        result = predictor.predict(text)
        print(f"\nInput: \"{text[:60]}\"")
        print(f"  → {result.get('condition', 'N/A')} ({result.get('confidence', 0):.1%})")
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save using pickle with protocol 4 (Flask-compatible)
    with open(models_dir / 'tuned_pipeline.pkl', 'wb') as f:
        pickle.dump(predictor.pipeline, f, protocol=4)
    print("✓ Pipeline saved with pickle (Flask-compatible)")
    
    joblib.dump(predictor.label_encoder, models_dir / 'label_encoder.pkl')
    print("✓ Label encoder saved")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'training_samples': len(df_train),
        'test_samples': len(df_test),
        'conditions': predictor.label_encoder.classes_.tolist(),
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()}
    }
    
    with open(models_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Metadata saved")
    
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Accuracy: {metrics['accuracy']:.2%}")
    print("=" * 80)


if __name__ == '__main__':
    main()