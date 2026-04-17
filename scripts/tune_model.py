"""
Task 5: Model Tuning (Complete Production Version)
File: scripts/tune_model.py
Purpose: Optimize hyperparameters for the best model performance
         with clear progress tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
import time

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, make_scorer

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Custom feature extractors
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Collections for vocabulary extractor
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. FEATURE EXTRACTORS
# ============================================================================

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract comprehensive text features from reviews"""
    
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
        self.conditions_ = []
        
    def fit(self, X, y):
        self.conditions_ = sorted(np.unique(y))
        
        from collections import defaultdict
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
                min_df=2
            )
            
            condition_texts = texts
            other_texts = []
            for other_cond in self.conditions_:
                if other_cond != condition:
                    other_texts.extend(texts_by_condition[other_cond][:len(texts)])
            
            all_texts = condition_texts + other_texts
            if len(all_texts) > 0:
                vectorizer.fit(all_texts)
                self.condition_vocabularies[condition] = set(vectorizer.get_feature_names_out())
        
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_lower = str(text).lower()
            words = text_lower.split()
            
            row_features = []
            for condition in self.conditions_:
                vocab = self.condition_vocabularies.get(condition, set())
                matches = sum(1 for word in words if word in vocab)
                ratio = matches / max(len(words), 1)
                row_features.extend([matches, ratio])
            
            features.append(row_features)
        
        return np.array(features)


# ============================================================================
# 2. PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    """Simple progress tracker for grid search"""
    
    def __init__(self, total_fits, name):
        self.total = total_fits
        self.current = 0
        self.name = name
        self.start_time = time.time()
    
    def update(self):
        self.current += 1
        pct = self.current * 100 // self.total
        
        # Calculate ETA
        if self.current > 1:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.current
            remaining = (self.total - self.current) * avg_time
            eta_str = f" | ETA: {remaining/60:.1f}min"
        else:
            eta_str = ""
        
        print(f"\r    {self.name}: {self.current}/{self.total} fits completed ({pct}%){eta_str}", 
              end='', flush=True)
        
        if self.current == self.total:
            print()  # New line when complete


# ============================================================================
# 3. LOAD DATA
# ============================================================================

print("=" * 80)
print("TASK 5: MODEL TUNING (WITH PROGRESS TRACKING)")
print("=" * 80)
print()

overall_start = time.time()

train_path = Path('data/processed/cleaned_train_data.csv')
test_path = Path('data/processed/cleaned_test_data.csv')

print("[1] Loading data...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"    Training: {len(df_train):,} reviews")
print(f"    Test: {len(df_test):,} reviews")
print(f"    Conditions: {df_train['condition'].unique().tolist()}")
print()

X_train = df_train['review'].fillna('').values
y_train = df_train['condition'].values

X_test = df_test['review'].fillna('').values
y_test = df_test['condition'].values

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ============================================================================
# 4. BUILD BASE PIPELINE
# ============================================================================

print("[2] Building base pipeline...")

tfidf = TfidfVectorizer(stop_words='english')

feature_union = FeatureUnion([
    ('tfidf', tfidf),
    ('text_stats', TextFeatureExtractor()),
    ('sentiment', SentimentFeatureExtractor()),
    ('learned_vocab', LearnedVocabularyExtractor(max_features_per_class=50))
])

base_models = [
    ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
    ('rf', RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('gbm', GradientBoostingClassifier(random_state=42))
]

ensemble = VotingClassifier(estimators=base_models, voting='soft')

pipeline = Pipeline([
    ('features', feature_union),
    ('scaler', MaxAbsScaler()),
    ('classifier', ensemble)
])

print("    ✓ Pipeline built")
print()

# ============================================================================
# 5. RANDOMIZED SEARCH - WITH PROGRESS TRACKER
# ============================================================================

print("[3] RANDOMIZED SEARCH - Broad Parameter Exploration")
print("-" * 60)

random_params = {
    'features__tfidf__max_features': [1500, 2000, 2500],
    'features__tfidf__ngram_range': [(1, 1), (1, 2)],
    'features__tfidf__min_df': [2, 3],
    'features__tfidf__max_df': [0.7, 0.8],
    'classifier__rf__n_estimators': [100, 150, 200],
    'classifier__rf__max_depth': [15, 20, 25, None],
    'classifier__rf__min_samples_split': [2, 5],
    'classifier__xgb__n_estimators': [100, 150, 200],
    'classifier__xgb__max_depth': [3, 5, 7],
    'classifier__xgb__learning_rate': [0.05, 0.1, 0.2],
    'classifier__xgb__subsample': [0.8, 1.0],
    'classifier__lr__C': [0.1, 1.0, 10.0],
    'classifier__lr__max_iter': [1000, 2000],
    'classifier__gbm__n_estimators': [100, 150],
    'classifier__gbm__max_depth': [3, 5],
    'classifier__gbm__learning_rate': [0.05, 0.1],
    'classifier__weights': [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 2, 3, 1],
        [1, 3, 3, 1],
    ]
}

n_iter = 30
n_folds = 3
total_random_fits = n_iter * n_folds

print(f"    Testing: {n_iter} random combinations")
print(f"    Cross-validation: {n_folds}-fold")
print(f"    Total fits: {total_random_fits}")
print()

# Create progress tracker
progress = ProgressTracker(total_random_fits, "Randomized Search")

def progress_scorer(estimator, X, y):
    progress.update()
    return accuracy_score(y, estimator.predict(X))

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=random_params,
    n_iter=n_iter,
    cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
    scoring=make_scorer(progress_scorer),
    n_jobs=1,
    verbose=0,
    random_state=42
)

print("    Running...")
random_search.fit(X_train, y_train_encoded)

print(f"\n\n    ✓ Randomized search complete!")
print(f"    Best score: {random_search.best_score_:.4f}")
print()

# ============================================================================
# 6. GRID SEARCH - WITH PROGRESS TRACKER
# ============================================================================

print("[4] GRID SEARCH - Focused Fine-Tuning")
print("-" * 60)

best_params = random_search.best_params_

focused_grid = {}

if 'features__tfidf__max_features' in best_params:
    focused_grid['features__tfidf__max_features'] = [best_params['features__tfidf__max_features']]

rf_depth = best_params.get('classifier__rf__max_depth', 20)
if rf_depth is None:
    rf_depth = 25
focused_grid['classifier__rf__max_depth'] = [rf_depth]

rf_estimators = best_params.get('classifier__rf__n_estimators', 150)
focused_grid['classifier__rf__n_estimators'] = [rf_estimators, rf_estimators + 50]

xgb_depth = best_params.get('classifier__xgb__max_depth', 5)
focused_grid['classifier__xgb__max_depth'] = [xgb_depth]

xgb_lr = best_params.get('classifier__xgb__learning_rate', 0.1)
focused_grid['classifier__xgb__learning_rate'] = [xgb_lr]

xgb_estimators = best_params.get('classifier__xgb__n_estimators', 150)
focused_grid['classifier__xgb__n_estimators'] = [xgb_estimators]

focused_grid['classifier__weights'] = [
    [1, 2, 3, 1],
    [1, 2, 4, 1],
    [1, 3, 3, 1]
]

n_combinations = 1
for values in focused_grid.values():
    n_combinations *= len(values)

total_grid_fits = n_combinations * n_folds

print(f"    Combinations: {n_combinations}")
print(f"    Cross-validation: {n_folds}-fold")
print(f"    Total fits: {total_grid_fits}")
print()

grid_progress = ProgressTracker(total_grid_fits, "Grid Search")

def grid_progress_scorer(estimator, X, y):
    grid_progress.update()
    return accuracy_score(y, estimator.predict(X))

grid_search = GridSearchCV(
    pipeline,
    param_grid=focused_grid,
    cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
    scoring=make_scorer(grid_progress_scorer),
    n_jobs=1,
    verbose=0
)

print("    Running...")
grid_search.fit(X_train, y_train_encoded)

print(f"\n\n    ✓ Grid search complete!")
print(f"    Best score: {grid_search.best_score_:.4f}")
print()

# ============================================================================
# 7. EVALUATE TUNED MODEL
# ============================================================================

print("[5] EVALUATING TUNED MODEL")
print("-" * 60)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_encoded, y_pred, average='weighted'
)

print(f"\n    Test Set Performance:")
print(f"      - Accuracy:  {accuracy:.4f}")
print(f"      - Precision: {precision:.4f}")
print(f"      - Recall:    {recall:.4f}")
print(f"      - F1 Score:  {f1:.4f}")

print("\n    Per-Class Performance:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
print()

# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================

print("[6] CROSS-VALIDATION (5-Fold)")
print("-" * 60)

cv_scores = cross_val_score(
    best_model, 
    X_train, 
    y_train_encoded, 
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print(f"\n    Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print()

# ============================================================================
# 9. BASELINE COMPARISON
# ============================================================================

print("[7] BASELINE COMPARISON")
print("-" * 60)

baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

baseline.fit(X_train, y_train_encoded)
baseline_pred = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test_encoded, baseline_pred)

print(f"\n    Baseline (TF-IDF + Logistic): {baseline_accuracy:.4f}")
print(f"    Tuned Ensemble Model:         {accuracy:.4f}")
print(f"    Improvement:                  +{accuracy - baseline_accuracy:.4f}")
print(f"    Relative:                     {(accuracy/baseline_accuracy - 1)*100:.1f}%")
print()

# ============================================================================
# 10. CONFUSION MATRIX
# ============================================================================

print("[8] GENERATING CONFUSION MATRIX")
print("-" * 60)

cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f'Confusion Matrix - Tuned Model (Acc: {accuracy:.2%})', fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()

output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'confusion_matrix_tuned.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved to: outputs/figures/confusion_matrix_tuned.png")
print()

# ============================================================================
# 11. FEATURE IMPORTANCE
# ============================================================================

print("[9] FEATURE IMPORTANCE ANALYSIS")
print("-" * 60)

rf_model = best_model.named_steps['classifier'].named_estimators_['rf']

tfidf_features = list(best_model.named_steps['features'].get_params()['tfidf'].get_feature_names_out())
text_stat_features = [f'text_stat_{i}' for i in range(13)]
sentiment_features = [f'sentiment_{i}' for i in range(8)]
vocab_features = [f'vocab_{i}' for i in range(len(label_encoder.classes_) * 2)]

all_features = tfidf_features + text_stat_features + sentiment_features + vocab_features

if len(all_features) == rf_model.feature_importances_.shape[0]:
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n    Top 15 Most Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"      {i+1:2d}. {row['feature'][:45]:<45} {row['importance']:.4f}")
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Feature Importance - Tuned Model', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_tuned.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    ✓ Saved to: outputs/figures/feature_importance_tuned.png")

print()

# ============================================================================
# 12. SAVE MODEL AND RESULTS
# ============================================================================

print("[10] SAVING MODEL AND RESULTS")
print("-" * 60)

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

joblib.dump(best_model, models_dir / 'tuned_pipeline.pkl')
print(f"    ✓ Model saved to: models/tuned_pipeline.pkl")

joblib.dump(label_encoder, models_dir / 'label_encoder.pkl')
print(f"    ✓ Label encoder saved to: models/label_encoder.pkl")

tuning_results = {
    'tuning_date': datetime.now().isoformat(),
    'training_samples': len(df_train),
    'test_samples': len(df_test),
    'conditions': label_encoder.classes_.tolist(),
    'random_search_best': float(random_search.best_score_),
    'grid_search_best': float(grid_search.best_score_),
    'test_accuracy': float(accuracy),
    'test_precision': float(precision),
    'test_recall': float(recall),
    'test_f1': float(f1),
    'cv_scores': [float(s) for s in cv_scores],
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'baseline_accuracy': float(baseline_accuracy),
    'improvement': float(accuracy - baseline_accuracy),
    'improvement_pct': float((accuracy/baseline_accuracy - 1) * 100),
    'total_time_seconds': time.time() - overall_start
}

with open(models_dir / 'tuning_results.json', 'w') as f:
    json.dump(tuning_results, f, indent=2)
print(f"    ✓ Results saved to: models/tuning_results.json")

print()

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================

total_time = time.time() - overall_start

print("=" * 80)
print("MODEL TUNING COMPLETE")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TUNING SUMMARY                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Dataset:                                                                    ║
║    • Training samples:  {len(df_train):>8,}                                           ║
║    • Test samples:      {len(df_test):>8,}                                           ║
║    • Conditions:        {', '.join(label_encoder.classes_)}                              ║
║                                                                              ║
║  Performance:                                                                ║
║    • Baseline:          {baseline_accuracy:.4f}                                          ║
║    • Tuned Model:       {accuracy:.4f}                                          ║
║    • Improvement:       +{accuracy - baseline_accuracy:.4f} ({(accuracy/baseline_accuracy - 1)*100:.1f}%)                                    ║
║    • CV Mean:           {cv_scores.mean():.4f} (±{cv_scores.std():.4f})                                      ║
║                                                                              ║
║  Time:                                                                       ║
║    • Total duration:    {total_time/60:.1f} minutes                                          ║
║                                                                              ║
║  Files Saved:                                                                ║
║    • models/tuned_pipeline.pkl                                               ║
║    • models/label_encoder.pkl                                                ║
║    • models/tuning_results.json                                              ║
║    • outputs/figures/confusion_matrix_tuned.png                              ║
║    • outputs/figures/feature_importance_tuned.png                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 80)