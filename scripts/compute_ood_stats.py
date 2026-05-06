"""
Compute OOD Statistics Using Tuned Model
"""

import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

# ── Use the same stable module path as tune_model.py ─────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from web.services.custom_transformers import (
    TextFeatureExtractor,
    SentimentFeatureExtractor,
    LearnedVocabularyExtractor,
)

# Register under __main__ so pickle can resolve them when loading
import __main__
__main__.TextFeatureExtractor    = TextFeatureExtractor
__main__.SentimentFeatureExtractor = SentimentFeatureExtractor
__main__.LearnedVocabularyExtractor = LearnedVocabularyExtractor

import joblib

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
with open('models/tuned_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)
encoder = joblib.load('models/label_encoder.pkl')
print(f"    ✓ Model loaded — classes: {encoder.classes_.tolist()}")

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
tfidf = TfidfVectorizer(
    max_features=500, stop_words='english', ngram_range=(1, 2), min_df=2
)
X_tfidf = tfidf.fit_transform(X_train)

condition_stats = {}
for condition in encoder.classes_:
    print(f"    Processing: {condition}")
    mask = y_train == condition
    condition_vectors = X_tfidf[mask]

    if condition_vectors.shape[0] > 0:
        centroid = np.asarray(condition_vectors.mean(axis=0)).flatten()

        distances = []
        sample_limit = min(condition_vectors.shape[0], 1000)
        for i in range(sample_limit):
            vec = np.asarray(condition_vectors[i].todense()).flatten()
            if vec.sum() > 0:
                distances.append(cosine(vec, centroid))

        if distances:
            condition_stats[condition] = {
                'mean_distance':  float(np.mean(distances)),
                'std_distance':   float(np.std(distances)),
                'max_distance':   float(np.max(distances)),
                'sample_count':   int(condition_vectors.shape[0])
            }
        else:
            condition_stats[condition] = {
                'mean_distance': 0.5, 'std_distance': 0.1,
                'max_distance': 1.0, 'sample_count': int(condition_vectors.shape[0])
            }
    else:
        condition_stats[condition] = {
            'mean_distance': 0.5, 'std_distance': 0.1,
            'max_distance': 1.0, 'sample_count': 0
        }

# Confidence thresholds from tuned model
print("\n[5] Computing confidence thresholds from tuned model...")
y_pred_proba = model.predict_proba(X_train)
max_probas = y_pred_proba.max(axis=1)

confidence_thresholds = {
    'very_low': float(np.percentile(max_probas, 5)),
    'low':      float(np.percentile(max_probas, 15)),
    'medium':   float(np.percentile(max_probas, 35)),
    'high':     float(np.percentile(max_probas, 65))
}
print(f"    - Very Low : {confidence_thresholds['very_low']:.3f}")
print(f"    - Low      : {confidence_thresholds['low']:.3f}")
print(f"    - Medium   : {confidence_thresholds['medium']:.3f}")
print(f"    - High     : {confidence_thresholds['high']:.3f}")

# Text statistics
print("\n[6] Computing text statistics...")
review_lengths = [len(str(r).split()) for r in X_train]
text_stats = {
    'avg_review_length': float(np.mean(review_lengths)),
    'std_review_length': float(np.std(review_lengths)),
    'min_review_length': int(np.min(review_lengths)),
    'percentile_5':      int(np.percentile(review_lengths, 5)),
    'percentile_10':     int(np.percentile(review_lengths, 10))
}
print(f"    - Avg length : {text_stats['avg_review_length']:.1f} words")
print(f"    - Min recommended: {text_stats['percentile_10']} words")

# Save
ood_stats = {
    'in_distribution_vocab': list(in_distribution_vocab),
    'vocab_size':            len(in_distribution_vocab),
    'condition_stats':       condition_stats,
    'confidence_thresholds': confidence_thresholds,
    'text_stats':            text_stats,
    'tfidf_feature_names':   tfidf.get_feature_names_out().tolist(),
    'training_samples':      len(df_train),
    'conditions':            encoder.classes_.tolist(),
    'created_date':          datetime.now().isoformat(),
    'model_used':            'tuned_pipeline.pkl'
}

with open('models/ood_statistics.json', 'w') as f:
    json.dump(ood_stats, f, indent=2)

print(f"\n[7] ✓ OOD statistics saved to: models/ood_statistics.json")
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)