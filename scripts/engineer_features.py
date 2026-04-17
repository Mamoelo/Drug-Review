"""
Task 2A: Feature Extraction
File: scripts/engineer_features.py
Purpose: Extract numerical features from cleaned text data
"""

import pandas as pd
import numpy as np
import re
import html
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD CLEANED DATA
# ============================================================================

print("=" * 70)
print("TASK 2A: FEATURE EXTRACTION")
print("=" * 70)

# Load cleaned datasets
train_path = Path('data/processed/cleaned_train_data.csv')
test_path = Path('data/processed/cleaned_test_data.csv')

print("\n[1] Loading cleaned datasets...")
df_train = pd.read_csv(train_path, parse_dates=['review_date'])
df_test = pd.read_csv(test_path, parse_dates=['review_date'])

print(f"    Training data: {df_train.shape}")
print(f"    Test data: {df_test.shape}")

# ============================================================================
# 2. TEXT-BASED FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("2. EXTRACTING TEXT-BASED FEATURES")
print("=" * 70)

def extract_text_features(df, text_column='review'):
    """Extract features from review text"""
    
    print("\n[1] Calculating text statistics...")
    
    # Basic text statistics
    df['char_count'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
    df['sentence_count'] = df[text_column].str.count(r'[.!?]+')
    df['avg_sentence_length'] = df['word_count'] / df['sentence_count'].replace(0, 1)
    
    # Punctuation and special characters
    df['exclamation_count'] = df[text_column].str.count('!')
    df['question_count'] = df[text_column].str.count(r'\?')
    df['capital_ratio'] = df[text_column].str.count(r'[A-Z]') / df['char_count'].replace(0, 1)
    
    print(f"    - Character count: mean={df['char_count'].mean():.0f}")
    print(f"    - Word count: mean={df['word_count'].mean():.0f}")
    print(f"    - Average word length: mean={df['avg_word_length'].mean():.2f}")
    
    return df


def extract_sentiment_features(df, text_column='review'):
    """Extract sentiment scores using VADER"""
    
    print("\n[2] Calculating sentiment scores...")
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x)))
    
    df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
    
    # Sentiment category
    df['sentiment_category'] = df['sentiment_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    print(f"    - Positive reviews: {(df['sentiment_category'] == 'positive').sum()}")
    print(f"    - Neutral reviews: {(df['sentiment_category'] == 'neutral').sum()}")
    print(f"    - Negative reviews: {(df['sentiment_category'] == 'negative').sum()}")
    
    return df


def extract_medical_keywords(df, text_column='review'):
    """Extract counts of medical/symptom keywords"""
    
    print("\n[3] Extracting medical keyword features...")
    
    # Keyword dictionaries for each condition
    depression_keywords = [
        'depressed', 'depression', 'sad', 'hopeless', 'suicidal', 'crying',
        'anxiety', 'panic', 'mood', 'unhappy', 'worthless', 'tired',
        'fatigue', 'insomnia', 'sleep', 'appetite'
    ]
    
    bp_keywords = [
        'blood pressure', 'hypertension', 'bp', 'pressure', 'systolic',
        'diastolic', 'heart', 'pulse', 'hypertensive', 'dizzy', 'headache'
    ]
    
    diabetes_keywords = [
        'diabetes', 'diabetic', 'blood sugar', 'glucose', 'insulin',
        'a1c', 'type 2', 'sugar', 'metformin', 'carb', 'diet'
    ]
    
    side_effect_keywords = [
        'nausea', 'headache', 'dizziness', 'fatigue', 'weight gain',
        'weight loss', 'insomnia', 'drowsiness', 'dry mouth', 'constipation',
        'diarrhea', 'rash', 'itching', 'vomiting', 'stomach pain',
        'muscle pain', 'joint pain', 'blurred vision', 'tremors', 'sweating'
    ]
    
    def count_keywords(text, keywords):
        text_lower = str(text).lower()
        return sum(1 for kw in keywords if kw in text_lower)
    
    df['depression_keyword_count'] = df[text_column].apply(
        lambda x: count_keywords(x, depression_keywords)
    )
    df['bp_keyword_count'] = df[text_column].apply(
        lambda x: count_keywords(x, bp_keywords)
    )
    df['diabetes_keyword_count'] = df[text_column].apply(
        lambda x: count_keywords(x, diabetes_keywords)
    )
    df['side_effect_count'] = df[text_column].apply(
        lambda x: count_keywords(x, side_effect_keywords)
    )
    
    # Keyword density
    df['depression_keyword_density'] = df['depression_keyword_count'] / df['word_count'].replace(0, 1)
    df['bp_keyword_density'] = df['bp_keyword_count'] / df['word_count'].replace(0, 1)
    df['diabetes_keyword_density'] = df['diabetes_keyword_count'] / df['word_count'].replace(0, 1)
    
    print(f"    - Avg depression keywords: {df['depression_keyword_count'].mean():.2f}")
    print(f"    - Avg BP keywords: {df['bp_keyword_count'].mean():.2f}")
    print(f"    - Avg diabetes keywords: {df['diabetes_keyword_count'].mean():.2f}")
    print(f"    - Avg side effect mentions: {df['side_effect_count'].mean():.2f}")
    
    return df


# ============================================================================
# 3. METADATA FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("3. EXTRACTING METADATA FEATURES")
print("=" * 70)

def extract_metadata_features(df):
    """Extract features from metadata columns"""
    
    print("\n[1] Processing rating features...")
    
    # Rating features
    df['rating_normalized'] = df['rating'] / 10.0
    df['rating_squared'] = df['rating'] ** 2
    df['is_high_rating'] = (df['rating'] >= 8).astype(int)
    df['is_low_rating'] = (df['rating'] <= 3).astype(int)
    
    print(f"    - High ratings (8-10): {df['is_high_rating'].sum()} ({df['is_high_rating'].mean()*100:.1f}%)")
    print(f"    - Low ratings (1-3): {df['is_low_rating'].sum()} ({df['is_low_rating'].mean()*100:.1f}%)")
    
    print("\n[2] Processing useful count features...")
    
    # Useful count features (handle outliers with log transform)
    df['useful_count_log'] = np.log1p(df['useful_count'])
    df['useful_count_sqrt'] = np.sqrt(df['useful_count'])
    df['is_useful'] = (df['useful_count'] > 0).astype(int)
    df['is_highly_useful'] = (df['useful_count'] >= 10).astype(int)
    
    print(f"    - Reviews marked useful: {df['is_useful'].sum()} ({df['is_useful'].mean()*100:.1f}%)")
    print(f"    - Highly useful (10+): {df['is_highly_useful'].sum()} ({df['is_highly_useful'].mean()*100:.1f}%)")
    
    print("\n[3] Processing date features...")
    
    # Date features
    df['review_year'] = df['review_date'].dt.year
    df['review_month'] = df['review_date'].dt.month
    df['review_day'] = df['review_date'].dt.day
    df['review_dayofweek'] = df['review_date'].dt.dayofweek
    df['review_quarter'] = df['review_date'].dt.quarter
    df['is_weekend'] = df['review_dayofweek'].isin([5, 6]).astype(int)
    
    # Days since first review (for trend analysis)
    min_date = df['review_date'].min()
    df['days_since_first'] = (df['review_date'] - min_date).dt.days
    
    print(f"    - Date range: {df['review_date'].min().date()} to {df['review_date'].max().date()}")
    print(f"    - Years covered: {df['review_year'].nunique()}")
    
    return df


def extract_drug_features(df):
    """Extract drug-level aggregated features"""
    
    print("\n[4] Creating drug-level features...")
    
    # Drug frequency encoding
    drug_counts = df['drug_name'].value_counts()
    df['drug_frequency'] = df['drug_name'].map(drug_counts)
    df['drug_frequency_log'] = np.log1p(df['drug_frequency'])
    
    # Drug average rating
    drug_avg_rating = df.groupby('drug_name')['rating'].mean()
    df['drug_avg_rating'] = df['drug_name'].map(drug_avg_rating)
    
    # Drug rating std (consistency)
    drug_rating_std = df.groupby('drug_name')['rating'].std().fillna(0)
    df['drug_rating_std'] = df['drug_name'].map(drug_rating_std)
    
    # Drug usefulness score
    drug_usefulness = df.groupby('drug_name')['useful_count'].mean()
    df['drug_avg_usefulness'] = df['drug_name'].map(drug_usefulness)
    
    print(f"    - Unique drugs: {df['drug_name'].nunique()}")
    print(f"    - Most common drug frequency: {drug_counts.max()}")
    
    return df


# ============================================================================
# 4. TF-IDF VECTORIZATION
# ============================================================================

print("\n" + "=" * 70)
print("4. TF-IDF VECTORIZATION")
print("=" * 70)

def create_tfidf_features(train_df, test_df, text_column='review'):
    """Create TF-IDF features from review text"""
    
    print("\n[1] Creating TF-IDF vectorizer...")
    
    # Parameters (these can be tuned in Task 2B)
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )
    
    print(f"    - Max features: 2000")
    print(f"    - N-gram range: (1, 2)")
    print(f"    - Min document frequency: 5")
    print(f"    - Max document frequency: 0.8")
    
    # Fit on training data
    print("\n[2] Fitting vectorizer on training data...")
    train_tfidf = vectorizer.fit_transform(train_df[text_column].fillna(''))
    
    # Transform test data
    print("[3] Transforming test data...")
    test_tfidf = vectorizer.transform(test_df[text_column].fillna(''))
    
    print(f"\n    Training TF-IDF shape: {train_tfidf.shape}")
    print(f"    Test TF-IDF shape: {test_tfidf.shape}")
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n    Sample features: {', '.join(feature_names[:10])}...")
    
    return train_tfidf, test_tfidf, vectorizer, feature_names


# ============================================================================
# 5. APPLY FEATURE EXTRACTION
# ============================================================================

print("\n" + "=" * 70)
print("5. APPLYING FEATURE EXTRACTION TO DATASETS")
print("=" * 70)

# Process training data
print("\n[Training Data]")
df_train = extract_text_features(df_train)
df_train = extract_sentiment_features(df_train)
df_train = extract_medical_keywords(df_train)
df_train = extract_metadata_features(df_train)
df_train = extract_drug_features(df_train)

# Process test data
print("\n[Test Data]")
df_test = extract_text_features(df_test)
df_test = extract_sentiment_features(df_test)
df_test = extract_medical_keywords(df_test)
df_test = extract_metadata_features(df_test)
df_test = extract_drug_features(df_test)

# Create TF-IDF features
train_tfidf, test_tfidf, tfidf_vectorizer, tfidf_feature_names = create_tfidf_features(df_train, df_test)

# ============================================================================
# 6. FEATURE CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("6. FEATURE CORRELATION ANALYSIS")
print("=" * 70)

# Select numerical features for correlation analysis
numerical_features = [
    'char_count', 'word_count', 'avg_word_length', 'sentence_count',
    'sentiment_compound', 'sentiment_positive', 'sentiment_negative',
    'depression_keyword_count', 'bp_keyword_count', 'diabetes_keyword_count',
    'side_effect_count', 'rating_normalized', 'useful_count_log',
    'drug_frequency_log', 'drug_avg_rating', 'is_high_rating'
]

# Compute correlation matrix
corr_matrix = df_train[numerical_features].corr()

# Find highly correlated features (>0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

print("\nHighly Correlated Features (|r| > 0.8):")
if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"    - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
else:
    print("    No highly correlated features found")

# ============================================================================
# 7. SAVE FEATURE-ENGINEERED DATASETS
# ============================================================================

print("\n" + "=" * 70)
print("7. SAVING FEATURE-ENGINEERED DATASETS")
print("=" * 70)

# Save processed dataframes
train_output_path = Path('data/processed/features_train.csv')
test_output_path = Path('data/processed/features_test.csv')

df_train.to_csv(train_output_path, index=False)
df_test.to_csv(test_output_path, index=False)

print(f"\n✓ Training features saved to: {train_output_path}")
print(f"    Shape: {df_train.shape}")
print(f"    Columns: {len(df_train.columns)}")

print(f"\n✓ Test features saved to: {test_output_path}")
print(f"    Shape: {df_test.shape}")
print(f"    Columns: {len(df_test.columns)}")

# Save TF-IDF vectorizer for later use
import joblib
vectorizer_path = Path('models/tfidf_vectorizer.pkl')
joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"\n✓ TF-IDF vectorizer saved to: {vectorizer_path}")

# Save TF-IDF matrices
import scipy.sparse
scipy.sparse.save_npz('data/processed/train_tfidf.npz', train_tfidf)
scipy.sparse.save_npz('data/processed/test_tfidf.npz', test_tfidf)
print(f"✓ TF-IDF matrices saved")

# ============================================================================
# 8. FEATURE SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("8. FEATURE SUMMARY")
print("=" * 70)

feature_categories = {
    'Text Statistics': ['char_count', 'word_count', 'avg_word_length', 
                       'sentence_count', 'avg_sentence_length'],
    'Punctuation': ['exclamation_count', 'question_count', 'capital_ratio'],
    'Sentiment': ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 
                  'sentiment_neutral', 'sentiment_category'],
    'Medical Keywords': ['depression_keyword_count', 'bp_keyword_count', 
                        'diabetes_keyword_count', 'side_effect_count'],
    'Rating': ['rating_normalized', 'rating_squared', 'is_high_rating', 'is_low_rating'],
    'Usefulness': ['useful_count_log', 'useful_count_sqrt', 'is_useful', 'is_highly_useful'],
    'Date': ['review_year', 'review_month', 'review_day', 'review_dayofweek', 
             'review_quarter', 'is_weekend', 'days_since_first'],
    'Drug': ['drug_frequency', 'drug_frequency_log', 'drug_avg_rating', 
             'drug_rating_std', 'drug_avg_usefulness'],
    'TF-IDF': [f'tfidf_{i}' for i in range(2000)]
}

print("\nFeature Categories Summary:")
total_features = 0
for category, features in feature_categories.items():
    count = len([f for f in features if f in df_train.columns or 'tfidf' in category])
    total_features += count
    print(f"  - {category}: {count} features")

print(f"\nTotal extracted features: {total_features}")

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)