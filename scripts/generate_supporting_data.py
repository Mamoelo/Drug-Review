"""
Generate Supporting Datasets from Training Data
File: scripts/generate_supporting_data.py
Purpose: Extract medical terms, conditions vocabulary, and patterns from training data
         No hard-coded words - everything learned from the dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ============================================================================
# 1. LOAD TRAINING DATA
# ============================================================================

print("=" * 80)
print("GENERATING SUPPORTING DATASETS FROM TRAINING DATA")
print("=" * 80)

train_path = Path('data/processed/cleaned_train_data.csv')
df_train = pd.read_csv(train_path)

print(f"\n[1] Loaded training data: {len(df_train):,} reviews")
print(f"    Conditions: {df_train['condition'].unique().tolist()}")

# ============================================================================
# 2. EXTRACT MEDICAL VOCABULARY (Learned from data)
# ============================================================================

print("\n[2] Extracting medical vocabulary from all reviews...")

# Combine all reviews
all_reviews = ' '.join(df_train['review'].fillna('').astype(str).tolist()).lower()

# Tokenize and clean
stop_words = set(stopwords.words('english'))
words = word_tokenize(all_reviews)

# Filter words: alphabetic, length > 3, not stopword
filtered_words = [
    word for word in words 
    if word.isalpha() and len(word) > 3 and word not in stop_words
]

# Get word frequencies
word_freq = Counter(filtered_words)

# Extract top words as medical vocabulary
medical_vocabulary = {
    'unigrams': dict(word_freq.most_common(1000)),
    'total_unique_words': len(word_freq),
    'total_words_processed': len(filtered_words)
}

# Extract bigrams
bigram_list = list(ngrams(filtered_words, 2))
bigram_freq = Counter(['_'.join(bg) for bg in bigram_list])
medical_vocabulary['bigrams'] = dict(bigram_freq.most_common(500))

# Extract trigrams
trigram_list = list(ngrams(filtered_words, 3))
trigram_freq = Counter(['_'.join(tg) for tg in trigram_list])
medical_vocabulary['trigrams'] = dict(trigram_freq.most_common(200))

print(f"    - Unigrams extracted: {len(medical_vocabulary['unigrams'])}")
print(f"    - Bigrams extracted: {len(medical_vocabulary['bigrams'])}")
print(f"    - Trigrams extracted: {len(medical_vocabulary['trigrams'])}")

# ============================================================================
# 3. EXTRACT CONDITION-SPECIFIC PATTERNS
# ============================================================================

print("\n[3] Extracting condition-specific patterns...")

condition_patterns = {}

for condition in df_train['condition'].unique():
    print(f"    Processing: {condition}")
    
    condition_df = df_train[df_train['condition'] == condition]
    condition_reviews = ' '.join(condition_df['review'].fillna('').astype(str).tolist()).lower()
    
    # Tokenize
    cond_words = word_tokenize(condition_reviews)
    cond_filtered = [
        word for word in cond_words 
        if word.isalpha() and len(word) > 3 and word not in stop_words
    ]
    
    # Get distinctive words using TF-IDF
    # Create a balanced sample - use all condition data but limit other data
    cond_sample_size = len(condition_df)
    
    # Get other conditions data
    other_df = df_train[df_train['condition'] != condition]
    other_sample_size = min(len(other_df), cond_sample_size)
    
    # Sample from other conditions (without replacement, limited by available data)
    if other_sample_size > 0:
        other_sample = other_df['review'].fillna('').sample(
            n=other_sample_size, 
            random_state=42, 
            replace=False
        )
    else:
        other_sample = pd.Series([])
    
    # Combine samples
    cond_sample = condition_df['review'].fillna('')
    all_samples = pd.concat([cond_sample, other_sample])
    
    if len(all_samples) > 0:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(all_samples)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF for this condition vs others
        cond_tfidf = tfidf_matrix[:len(cond_sample)].mean(axis=0)
        
        if len(other_sample) > 0:
            other_tfidf = tfidf_matrix[len(cond_sample):].mean(axis=0)
            tfidf_diff = np.array(cond_tfidf - other_tfidf).flatten()
        else:
            tfidf_diff = np.array(cond_tfidf).flatten()
        
        # Find terms with highest difference
        top_indices = np.argsort(tfidf_diff)[-50:][::-1]
        distinctive_terms = [feature_names[i] for i in top_indices if tfidf_diff[i] > 0]
    else:
        distinctive_terms = []
    
    # Word frequencies for this condition
    word_freq = Counter(cond_filtered)
    
    # Bigrams for this condition
    cond_bigrams = list(ngrams(cond_filtered, 2))
    bigram_freq = Counter(['_'.join(bg) for bg in cond_bigrams])
    
    # Rating statistics
    rating_stats = {
        'mean': float(condition_df['rating'].mean()),
        'std': float(condition_df['rating'].std()) if len(condition_df) > 1 else 0.0,
        'median': float(condition_df['rating'].median()),
        'high_rating_pct': float((condition_df['rating'] >= 8).mean() * 100),
        'low_rating_pct': float((condition_df['rating'] <= 3).mean() * 100)
    }
    
    # Drug statistics
    drug_stats = condition_df.groupby('drug_name').agg({
        'rating': ['mean', 'count'],
        'useful_count': 'sum'
    }).round(2)
    drug_stats.columns = ['avg_rating', 'review_count', 'total_useful']
    drug_stats = drug_stats[drug_stats['review_count'] >= 5].reset_index()
    drug_stats['score'] = drug_stats['avg_rating'] * np.log1p(drug_stats['review_count'])
    top_drugs = drug_stats.nlargest(20, 'score')['drug_name'].tolist()
    
    condition_patterns[condition] = {
        'sample_count': len(condition_df),
        'distinctive_terms': distinctive_terms[:30],
        'common_unigrams': dict(word_freq.most_common(100)),
        'common_bigrams': dict(bigram_freq.most_common(50)),
        'rating_statistics': rating_stats,
        'top_drugs': top_drugs,
        'avg_review_length': float(condition_df['review'].str.len().mean()),
        'avg_useful_count': float(condition_df['useful_count'].mean())
    }

print(f"    - Extracted patterns for {len(condition_patterns)} conditions")

# ============================================================================
# 4. EXTRACT SENTIMENT PATTERNS
# ============================================================================

print("\n[4] Extracting sentiment patterns...")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

sentiment_patterns = {
    'by_condition': {},
    'by_rating': {},
    'overall': {}
}

# By condition
for condition in df_train['condition'].unique():
    condition_df = df_train[df_train['condition'] == condition]
    sentiments = condition_df['review'].fillna('').apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    
    sentiment_patterns['by_condition'][condition] = {
        'mean': float(sentiments.mean()),
        'std': float(sentiments.std()) if len(sentiments) > 1 else 0.0,
        'positive_pct': float((sentiments >= 0.05).mean() * 100),
        'negative_pct': float((sentiments <= -0.05).mean() * 100),
        'neutral_pct': float(((sentiments > -0.05) & (sentiments < 0.05)).mean() * 100)
    }

# By rating
for rating in range(1, 11):
    rating_df = df_train[df_train['rating'] == rating]
    if len(rating_df) > 0:
        sentiments = rating_df['review'].fillna('').apply(
            lambda x: analyzer.polarity_scores(str(x))['compound']
        )
        sentiment_patterns['by_rating'][str(rating)] = {
            'mean': float(sentiments.mean()),
            'count': len(rating_df)
        }

# Overall
all_sentiments = df_train['review'].fillna('').apply(
    lambda x: analyzer.polarity_scores(str(x))['compound']
)
sentiment_patterns['overall'] = {
    'mean': float(all_sentiments.mean()),
    'std': float(all_sentiments.std()),
    'percentiles': {
        '25': float(np.percentile(all_sentiments, 25)),
        '50': float(np.percentile(all_sentiments, 50)),
        '75': float(np.percentile(all_sentiments, 75))
    }
}

print(f"    - Sentiment patterns extracted")

# ============================================================================
# 5. EXTRACT DRUG EFFECTIVENESS PATTERNS
# ============================================================================

print("\n[5] Extracting drug effectiveness patterns...")

drug_effectiveness = {}

for condition in df_train['condition'].unique():
    condition_df = df_train[df_train['condition'] == condition]
    
    drug_stats = condition_df.groupby('drug_name').agg({
        'rating': ['mean', 'count', 'std'],
        'useful_count': ['sum', 'mean']
    }).round(3)
    
    drug_stats.columns = ['avg_rating', 'review_count', 'rating_std', 'total_useful', 'avg_useful']
    drug_stats = drug_stats[drug_stats['review_count'] >= 3].reset_index()
    
    if len(drug_stats) > 0:
        # Calculate effectiveness score
        drug_stats['effectiveness_score'] = (
            drug_stats['avg_rating'] * 0.4 +
            np.log1p(drug_stats['review_count']) * 0.3 +
            np.log1p(drug_stats['total_useful']) * 0.2 +
            (1 / (1 + drug_stats['rating_std'].fillna(0))) * 0.1
        )
        
        # Categorize effectiveness
        score_80 = drug_stats['effectiveness_score'].quantile(0.8) if len(drug_stats) >= 5 else drug_stats['effectiveness_score'].max()
        score_50 = drug_stats['effectiveness_score'].quantile(0.5) if len(drug_stats) >= 2 else drug_stats['effectiveness_score'].median()
        score_20 = drug_stats['effectiveness_score'].quantile(0.2) if len(drug_stats) >= 5 else drug_stats['effectiveness_score'].min()
        
        def categorize_effectiveness(score):
            if score >= score_80:
                return 'Highly Effective'
            elif score >= score_50:
                return 'Effective'
            elif score >= score_20:
                return 'Moderately Effective'
            else:
                return 'Limited Effectiveness'
        
        drug_stats['effectiveness_category'] = drug_stats['effectiveness_score'].apply(categorize_effectiveness)
        drug_effectiveness[condition] = drug_stats.to_dict('records')

print(f"    - Extracted effectiveness for {len(drug_effectiveness)} conditions")

# ============================================================================
# 6. EXTRACT CONVERSATIONAL PATTERNS
# ============================================================================

print("\n[6] Extracting conversational patterns from data...")

# Find common sentence starters
all_text = ' '.join(df_train['review'].fillna('').astype(str).tolist()).lower()
sentences = re.split(r'[.!?]+', all_text)
sentence_starters = Counter()
for sent in sentences:
    words = sent.strip().split()
    if len(words) >= 2:
        sentence_starters[f"{words[0]} {words[1]}"] += 1

# Find words that appear in short texts (potential conversational)
short_texts = df_train[df_train['review'].str.len() < 100]['review'].fillna('').astype(str).tolist()
short_text_words = ' '.join(short_texts).lower().split()
short_word_freq = Counter([w for w in short_text_words if len(w) > 2 and w.isalpha()])

conversational_patterns = {
    'common_starters': dict(sentence_starters.most_common(50)),
    'min_word_count': 2,
    'max_word_count': int(df_train['review'].str.split().str.len().max()),
    'avg_word_count': float(df_train['review'].str.split().str.len().mean()),
    'common_short_words': dict(short_word_freq.most_common(30))
}

print(f"    - Extracted {len(conversational_patterns['common_starters'])} sentence starters")

# ============================================================================
# 7. SAVE ALL SUPPORTING DATASETS
# ============================================================================

print("\n[7] Saving supporting datasets...")

supporting_data_dir = Path('data/supporting')
supporting_data_dir.mkdir(exist_ok=True)

# Save medical vocabulary
with open(supporting_data_dir / 'medical_vocabulary.json', 'w') as f:
    json.dump(medical_vocabulary, f, indent=2)
print(f"    ✓ Saved: medical_vocabulary.json")

# Save condition patterns
with open(supporting_data_dir / 'condition_patterns.json', 'w') as f:
    json.dump(condition_patterns, f, indent=2)
print(f"    ✓ Saved: condition_patterns.json")

# Save sentiment patterns
with open(supporting_data_dir / 'sentiment_patterns.json', 'w') as f:
    json.dump(sentiment_patterns, f, indent=2)
print(f"    ✓ Saved: sentiment_patterns.json")

# Save drug effectiveness
with open(supporting_data_dir / 'drug_effectiveness.json', 'w') as f:
    json.dump(drug_effectiveness, f, indent=2)
print(f"    ✓ Saved: drug_effectiveness.json")

# Save conversational patterns
with open(supporting_data_dir / 'conversational_patterns.json', 'w') as f:
    json.dump(conversational_patterns, f, indent=2)
print(f"    ✓ Saved: conversational_patterns.json")

# Save dataset statistics
dataset_stats = {
    'total_reviews': len(df_train),
    'conditions': df_train['condition'].value_counts().to_dict(),
    'unique_drugs': int(df_train['drug_name'].nunique()),
    'rating_distribution': {str(k): int(v) for k, v in df_train['rating'].value_counts().sort_index().to_dict().items()},
    'useful_count_stats': {
        'mean': float(df_train['useful_count'].mean()),
        'median': float(df_train['useful_count'].median()),
        'max': int(df_train['useful_count'].max())
    }
}

# Handle date range if date column exists
if 'review_date' in df_train.columns:
    try:
        dataset_stats['date_range'] = {
            'start': str(df_train['review_date'].min()),
            'end': str(df_train['review_date'].max())
        }
    except:
        pass

with open(supporting_data_dir / 'dataset_statistics.json', 'w') as f:
    json.dump(dataset_stats, f, indent=2)
print(f"    ✓ Saved: dataset_statistics.json")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUPPORTING DATASETS GENERATED SUCCESSFULLY")
print("=" * 80)

print(f"""
Files created in data/supporting/:
  - medical_vocabulary.json ({len(medical_vocabulary['unigrams']):,} unigrams, {len(medical_vocabulary['bigrams'])} bigrams)
  - condition_patterns.json ({len(condition_patterns)} conditions)
  - sentiment_patterns.json
  - drug_effectiveness.json ({len(drug_effectiveness)} conditions)
  - conversational_patterns.json
  - dataset_statistics.json

All data learned from training set - no hard-coded values!
Training samples: {len(df_train):,}
""")