"""
Compute OOD Statistics Using Tuned Model
File: scripts/compute_ood_stats.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from collections import defaultdict

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


# ============================================================================
# DEFINE CUSTOM CLASSES (Must match the ones used during training)
# ============================================================================

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract comprehensive text features from reviews"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text = str(text).lower()
            words = text.split()
            
            char_count = len(text)
            word_count = len(words)
            unique_words = len(set(words))
            
            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            sentence_count = max(text.count('.') + text.count('!') + text.count('?'), 1)
            avg_sentence_length = word_count / sentence_count
            
            exclamation_count = text.count('!')
            question_count = text.count('?')
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)
            ttr = unique_words / max(word_count, 1)
            
            features.append([
                char_count, word_count, unique_words, avg_word_length,
                sentence_count, avg_sentence_length, exclamation_count,
                question_count, uppercase_ratio, ttr,
                np.log1p(char_count), np.log1p(word_count), np.sqrt(word_count)
            ])
        
        return np.array(features)


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment features using VADER"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            scores = self.analyzer.polarity_scores(str(text))
            compound_normalized = (scores['compound'] + 1) / 2
            
            features.append([
                compound_normalized, scores['pos'], scores['neg'], scores['neu'],
                abs(scores['compound']),
                1 if scores['compound'] >= 0.05 else 0,
                1 if scores['compound'] <= -0.05 else 0,
                scores['pos'] + scores['neu']
            ])
        
        return np.array(features)


class LearnedVocabularyExtractor(BaseEstimator, TransformerMixin):
    """Extract features based on vocabulary learned from training data"""
    
    def __init__(self, max_features_per_class=50):
        self.max_features_per_class = max_features_per_class
        self.condition_vocabularies = {}
        self.condition_bigrams = {}
        self.all_learned_terms = set()
        self.conditions_ = []
        
    def fit(self, X, y):
        self.conditions_ = sorted(np.unique(y))
        
        texts_by_condition = defaultdict(list)
        for text, label in zip(X, y):
            texts_by_condition[label].append(str(text).lower())
        
        for condition in self.conditions_:
            texts = texts_by_condition[condition]
            if not texts:
                continue
            
            vectorizer = TfidfVectorizer(
                max_features=self.max_features_per_class,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=2
            )
            
            condition_texts = texts
            other_texts = []
            for other_cond in self.conditions_:
                if other_cond != condition:
                    other_texts.extend(texts_by_condition[other_cond][:len(texts)])
            
            all_texts = condition_texts + other_texts
            labels = [1] * len(condition_texts) + [0] * len(other_texts)
            
            if len(all_texts) > 0 and sum(labels) > 0:
                vectorizer.fit(all_texts)
                terms = vectorizer.get_feature_names_out()
                self.condition_vocabularies[condition] = set(terms)
                self.all_learned_terms.update(terms)
            
            all_words = ' '.join(texts).split()
            bigram_counter = Counter()
            for i in range(len(all_words) - 1):
                bigram = f"{all_words[i]}_{all_words[i+1]}"
                bigram_counter[bigram] += 1
            
            top_bigrams = [bg for bg, _ in bigram_counter.most_common(30)]
            self.condition_bigrams[condition] = set(top_bigrams)
        
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_lower = str(text).lower()
            words = text_lower.split()
            
            text_bigrams = set()
            for i in range(len(words) - 1):
                text_bigrams.add(f"{words[i]}_{words[i+1]}")
            
            row_features = []
            for condition in self.conditions_:
                vocab = self.condition_vocabularies.get(condition, set())
                bigrams = self.condition_bigrams.get(condition, set())
                
                unigram_matches = sum(1 for word in words if word in vocab)
                unigram_ratio = unigram_matches / max(len(words), 1)
                bigram_matches = len(text_bigrams.intersection(bigrams))
                bigram_ratio = bigram_matches / max(len(text_bigrams), 1)
                
                row_features.extend([unigram_matches, unigram_ratio, bigram_matches, bigram_ratio])
            
            features.append(row_features)
        
        return np.array(features)


# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("COMPUTING OOD STATISTICS FOR TUNED MODEL")
print("=" * 80)

# Load data
train_path = Path('data/processed/cleaned_train_data.csv')
df_train = pd.read_csv(train_path)
print(f"\n[1] Loaded {len(df_train):,} training reviews")

# Load tuned model and encoder
print("[2] Loading tuned model...")
model = joblib.load('models/tuned_pipeline.pkl')
encoder = joblib.load('models/label_encoder.pkl')
print("    ✓ Model loaded")

X_train = df_train['review'].fillna('').astype(str).values
y_train = df_train['condition'].values

# Build vocabulary
print("\n[3] Building in-distribution vocabulary...")
all_reviews = ' '.join(X_train).lower()
words = all_reviews.split()
filtered_words = [w for w in words if w.isalpha() and len(w) > 2]
word_freq = Counter(filtered_words)
in_distribution_vocab = set([word for word, count in word_freq.items() if count >= 5])
print(f"    - {len(in_distribution_vocab):,} terms")

# TF-IDF statistics
print("\n[4] Computing TF-IDF statistics...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf.fit_transform(X_train)

condition_stats = {}
for condition in encoder.classes_:
    print(f"    Processing: {condition}")
    mask = y_train == condition
    condition_vectors = X_tfidf[mask]
    
    if condition_vectors.shape[0] > 0:
        centroid = condition_vectors.mean(axis=0)
        
        # Convert centroid to dense array
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
                'mean_distance': 0.5, 'std_distance': 0.1, 'max_distance': 1.0,
                'sample_count': int(condition_vectors.shape[0])
            }
    else:
        condition_stats[condition] = {
            'mean_distance': 0.5, 'std_distance': 0.1, 'max_distance': 1.0,
            'sample_count': 0
        }

# Confidence thresholds from tuned model
print("\n[5] Computing confidence thresholds...")
y_pred_proba = model.predict_proba(X_train)
max_probas = y_pred_proba.max(axis=1)

confidence_thresholds = {
    'very_low': float(np.percentile(max_probas, 5)),
    'low': float(np.percentile(max_probas, 15)),
    'medium': float(np.percentile(max_probas, 35)),
    'high': float(np.percentile(max_probas, 65))
}
print(f"    - Very Low: {confidence_thresholds['very_low']:.3f}")
print(f"    - Low: {confidence_thresholds['low']:.3f}")
print(f"    - Medium: {confidence_thresholds['medium']:.3f}")
print(f"    - High: {confidence_thresholds['high']:.3f}")

# Text statistics
print("\n[6] Computing text statistics...")
review_lengths = [len(str(r).split()) for r in X_train]
text_stats = {
    'avg_review_length': float(np.mean(review_lengths)),
    'std_review_length': float(np.std(review_lengths)),
    'min_review_length': int(np.min(review_lengths)),
    'percentile_5': int(np.percentile(review_lengths, 5)),
    'percentile_10': int(np.percentile(review_lengths, 10))
}
print(f"    - Avg length: {text_stats['avg_review_length']:.1f} words")
print(f"    - Min recommended: {text_stats['percentile_10']} words")

# Save
ood_stats = {
    'in_distribution_vocab': list(in_distribution_vocab),
    'vocab_size': len(in_distribution_vocab),
    'condition_stats': condition_stats,
    'confidence_thresholds': confidence_thresholds,
    'text_stats': text_stats,
    'tfidf_feature_names': tfidf.get_feature_names_out().tolist(),
    'training_samples': len(df_train),
    'conditions': encoder.classes_.tolist(),
    'created_date': datetime.now().isoformat(),
    'model_used': 'tuned_pipeline.pkl'
}

with open('models/ood_statistics.json', 'w') as f:
    json.dump(ood_stats, f, indent=2)

print(f"\n[7] ✓ OOD statistics saved to: models/ood_statistics.json")
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)