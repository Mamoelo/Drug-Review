"""
Task 5: Model Tuning
File: scripts/tune_model.py
Purpose: Hyperparameter optimisation for the ensemble pipeline.
         Produces a tuned model that outperforms the base model,
         with learning curves and sensitivity analysis for the report.
"""

import sys
import pickle
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV,
    StratifiedKFold, cross_val_score,
    learning_curve
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Stable module path so pickle works from Flask too ─────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from web.services.custom_transformers import (
    TextFeatureExtractor,
    SentimentFeatureExtractor,
    LearnedVocabularyExtractor,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(tfidf_max_features=2000, tfidf_ngram=(1, 2),
                   tfidf_min_df=2, tfidf_max_df=0.85,
                   lr_C=1.0,
                   rf_n=150, rf_depth=20,
                   xgb_n=150, xgb_depth=5, xgb_lr=0.1,
                   weights=None):
    """Build a fresh pipeline with the given hyperparameters."""
    if weights is None:
        weights = [1, 2, 3]

    tfidf = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram,
        min_df=tfidf_min_df,
        max_df=tfidf_max_df,
        stop_words='english',
        sublinear_tf=True,
    )

    features = FeatureUnion([
        ('tfidf',        tfidf),
        ('text_stats',   TextFeatureExtractor()),
        ('sentiment',    SentimentFeatureExtractor()),
        ('learned_vocab', LearnedVocabularyExtractor(max_features_per_class=50)),
    ])

    ensemble = VotingClassifier(
        estimators=[
            ('lr',  LogisticRegression(
                C=lr_C, class_weight='balanced',
                max_iter=1000, random_state=42)),
            ('rf',  RandomForestClassifier(
                n_estimators=rf_n, max_depth=rf_depth,
                class_weight='balanced', n_jobs=-1, random_state=42)),
            ('xgb', XGBClassifier(
                n_estimators=xgb_n, max_depth=xgb_depth,
                learning_rate=xgb_lr, eval_metric='mlogloss',
                n_jobs=-1, random_state=42,
                use_label_encoder=False)),
        ],
        voting='soft',
        weights=weights,
    )

    return Pipeline([
        ('features',   features),
        ('scaler',     MaxAbsScaler()),
        ('classifier', ensemble),
    ])


def banner(title, width=78):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

banner("TASK 5: MODEL TUNING")
overall_start = time.time()

df_train = pd.read_csv('data/processed/cleaned_train_data.csv')
df_test  = pd.read_csv('data/processed/cleaned_test_data.csv')

print(f"\n  Training samples : {len(df_train):,}")
print(f"  Test samples     : {len(df_test):,}")
print(f"  Conditions       : {df_train['condition'].unique().tolist()}")

X_train = df_train['review'].fillna('').values
y_train = df_train['condition'].values
X_test  = df_test['review'].fillna('').values
y_test  = df_test['condition'].values

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc  = label_encoder.transform(y_test)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BASELINE  (simple TF-IDF + LR — serves as lower bound)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1 — Baseline model (TF-IDF + Logistic Regression)")

baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf',   LogisticRegression(max_iter=1000, random_state=42)),
])
baseline.fit(X_train, y_train_enc)
baseline_acc = accuracy_score(y_test_enc, baseline.predict(X_test))
print(f"\n  Baseline accuracy: {baseline_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RANDOMISED SEARCH  (broad exploration, n_jobs=1 because VotingClassifier
#    already parallelises internally via n_jobs=-1 on RF and XGB)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 2 — Randomised Search (broad exploration)")

pipeline = build_pipeline()

param_dist = {
    # TF-IDF
    'features__tfidf__max_features': [1500, 2000, 2500],
    'features__tfidf__ngram_range':  [(1, 1), (1, 2)],
    'features__tfidf__min_df':       [2, 3],
    'features__tfidf__max_df':       [0.8, 0.85],
    # Logistic Regression
    'classifier__lr__C':             [0.1, 1.0, 10.0],
    # Random Forest
    'classifier__rf__n_estimators':  [100, 150, 200],
    'classifier__rf__max_depth':     [15, 20, None],
    # XGBoost
    'classifier__xgb__n_estimators': [100, 150, 200],
    'classifier__xgb__max_depth':    [3, 5, 7],
    'classifier__xgb__learning_rate':[0.05, 0.1, 0.2],
    # Ensemble weights  (LR, RF, XGB)
    'classifier__weights': [
        [1, 1, 1],
        [1, 2, 2],
        [1, 2, 3],
        [1, 3, 3],
        [1, 3, 4],
    ],
}

N_ITER  = 20          # 20 × 3 folds = 60 fits  (~25–40 min)
N_FOLDS = 3
CV      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\n  Iterations : {N_ITER}")
print(f"  CV folds   : {N_FOLDS}")
print(f"  Total fits : {N_ITER * N_FOLDS}")
print("\n  Running …  (verbose=1 shows one line per fit)")

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=CV,
    scoring='accuracy',   # <-- standard sklearn scorer, no wrapper needed
    n_jobs=1,
    verbose=1,
    random_state=42,
    refit=True,           # automatically refit best params on full training set
    error_score='raise',
)
random_search.fit(X_train, y_train_enc)

print(f"\n  ✓ Randomised search complete")
print(f"  Best CV accuracy : {random_search.best_score_:.4f}")
print(f"  Best params      :")
for k, v in random_search.best_params_.items():
    print(f"    {k} = {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FOCUSED GRID SEARCH  (refine around best params)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 3 — Focused Grid Search (fine-tuning)")

bp = random_search.best_params_

# Build a tight grid around the best values found
focused_grid = {
    'features__tfidf__max_features': [bp.get('features__tfidf__max_features', 2000)],
    'features__tfidf__ngram_range':  [bp.get('features__tfidf__ngram_range',  (1, 2))],
    'classifier__lr__C':             [bp.get('classifier__lr__C', 1.0)],
    'classifier__rf__n_estimators':  [
        bp.get('classifier__rf__n_estimators', 150),
        bp.get('classifier__rf__n_estimators', 150) + 50,
    ],
    'classifier__rf__max_depth': [bp.get('classifier__rf__max_depth', 20)],
    'classifier__xgb__n_estimators':  [bp.get('classifier__xgb__n_estimators', 150)],
    'classifier__xgb__max_depth':     [bp.get('classifier__xgb__max_depth', 5)],
    'classifier__xgb__learning_rate': [bp.get('classifier__xgb__learning_rate', 0.1)],
    'classifier__weights': [
        bp.get('classifier__weights', [1, 2, 3]),
        [1, 2, 4],
        [1, 3, 4],
    ],
}

n_combo = 1
for v in focused_grid.values():
    n_combo *= len(v)

print(f"\n  Combinations : {n_combo}")
print(f"  Total fits   : {n_combo * N_FOLDS}")
print("\n  Running …")

grid_search = GridSearchCV(
    pipeline,
    param_grid=focused_grid,
    cv=CV,
    scoring='accuracy',
    n_jobs=1,
    verbose=1,
    refit=True,
    error_score='raise',
)
grid_search.fit(X_train, y_train_enc)

print(f"\n  ✓ Grid search complete")
print(f"  Best CV accuracy : {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE BEST MODEL ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 4 — Evaluate tuned model on test set")

best_model = grid_search.best_estimator_
y_pred     = best_model.predict(X_test)

accuracy  = accuracy_score(y_test_enc, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_enc, y_pred, average='weighted'
)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  vs Baseline : {baseline_acc:.4f}")
delta = accuracy - baseline_acc
sign  = '+' if delta >= 0 else ''
print(f"  Improvement : {sign}{delta:.4f} ({sign}{delta/baseline_acc*100:.2f}%)")

print("\n  Per-class report:")
print(classification_report(
    y_test_enc, y_pred,
    target_names=label_encoder.classes_,
    digits=4
))

# ─────────────────────────────────────────────────────────────────────────────
# 6. CROSS-VALIDATION  (5-fold on training set)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 5 — 5-fold cross-validation")

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train_enc,
                             cv=cv5, scoring='accuracy', n_jobs=-1)

print(f"\n  Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean ± Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 6 — Confusion matrix")

output_dir = Path('outputs/figures')
output_dir.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title(f'Confusion Matrix — Tuned Model  (Accuracy: {accuracy:.2%})',
          fontsize=13)
plt.xlabel('Predicted', fontsize=11)
plt.ylabel('Actual',    fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_tuned.png', dpi=150)
plt.close()
print(f"\n  ✓ Saved: outputs/figures/confusion_matrix_tuned.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. LEARNING CURVE  (shows bias/variance trade-off — required for Task 5)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 7 — Learning curve")

print("\n  Computing learning curve (this takes a few minutes) …")
train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_train, y_train_enc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color='#2196F3')
plt.plot(train_sizes, val_mean, 'o-', color='#FF5722', label='Validation accuracy')
plt.fill_between(train_sizes,
                 val_mean - val_std,
                 val_mean + val_std,
                 alpha=0.15, color='#FF5722')
plt.xlabel('Training samples', fontsize=12)
plt.ylabel('Accuracy',         fontsize=12)
plt.title('Learning Curve — Tuned Ensemble Model', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0.7, 1.01)
plt.tight_layout()
plt.savefig(output_dir / 'learning_curve_tuned.png', dpi=150)
plt.close()
print(f"  ✓ Saved: outputs/figures/learning_curve_tuned.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. HYPERPARAMETER SENSITIVITY  (how accuracy varies with key params)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 8 — Hyperparameter sensitivity analysis")

print("\n  Extracting sensitivity from RandomizedSearch CV results …")

cv_results = pd.DataFrame(random_search.cv_results_)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Hyperparameter Sensitivity — RandomisedSearchCV Results',
             fontsize=14, fontweight='bold')

params_to_plot = [
    ('param_features__tfidf__max_features', 'TF-IDF max features'),
    ('param_classifier__rf__n_estimators',  'RF n_estimators'),
    ('param_classifier__xgb__max_depth',    'XGB max_depth'),
    ('param_classifier__xgb__learning_rate','XGB learning_rate'),
    ('param_classifier__lr__C',             'LR C'),
    ('param_classifier__rf__max_depth',     'RF max_depth'),
]

for ax, (param, label) in zip(axes.flat, params_to_plot):
    col = cv_results[param].astype(str)
    means = cv_results.groupby(col)['mean_test_score'].mean().sort_index()
    bars = ax.bar(range(len(means)), means.values,
                  color='#14b8a6', edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean CV Accuracy', fontsize=9)
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylim(max(0, means.min() - 0.01), means.max() + 0.01)
    ax.grid(axis='y', alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'hyperparameter_sensitivity.png', dpi=150)
plt.close()
print(f"  ✓ Saved: outputs/figures/hyperparameter_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 9 — Feature importance")

try:
    rf_estimator = best_model.named_steps['classifier'].named_estimators_['rf']
    feature_pipe = best_model.named_steps['features']

    tfidf_names   = list(feature_pipe.transformer_list[0][1].get_feature_names_out())
    text_names    = [f'text_stat_{i}' for i in range(13)]
    sent_names    = [f'sentiment_{i}' for i in range(8)]
    vocab_names   = [f'vocab_{i}'     for i in range(len(label_encoder.classes_) * 2)]
    all_feat_names = tfidf_names + text_names + sent_names + vocab_names

    importances = rf_estimator.feature_importances_
    if len(all_feat_names) == len(importances):
        fi_df = pd.DataFrame({'feature': all_feat_names, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(fi_df)), fi_df['importance'].values, color='steelblue')
        plt.yticks(range(len(fi_df)), fi_df['feature'].values, fontsize=9)
        plt.xlabel('Importance', fontsize=11)
        plt.title('Top 20 Feature Importances — Tuned Model', fontsize=13)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_tuned.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: outputs/figures/feature_importance_tuned.png")

        print("\n  Top 15 features:")
        for _, row in fi_df.head(15).iterrows():
            print(f"    {row['feature'][:50]:<50}  {row['importance']:.4f}")
    else:
        print(f"  ⚠ Feature count mismatch ({len(all_feat_names)} vs {len(importances)}) — skipping plot")
except Exception as e:
    print(f"  ⚠ Feature importance skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE MODEL  (pickle with stable module path)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 10 — Save model")

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Use pickle protocol 4 — compatible, stable, and the custom classes
# are registered under web.services.custom_transformers (not __main__)
# so Flask can load them cleanly.
with open(models_dir / 'tuned_pipeline.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=4)

import joblib
joblib.dump(label_encoder, models_dir / 'label_encoder.pkl')

print(f"\n  ✓ Tuned pipeline saved : models/tuned_pipeline.pkl")
print(f"  ✓ Label encoder saved  : models/label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────

total_time = time.time() - overall_start

# Serialise best params (some values like tuples aren't JSON-safe by default)
def json_safe(v):
    if isinstance(v, tuple): return list(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    return v

best_params_clean = {k: json_safe(v) for k, v in grid_search.best_params_.items()}

tuning_results = {
    'tuning_date':             datetime.now().isoformat(),
    'training_samples':        int(len(df_train)),
    'test_samples':            int(len(df_test)),
    'conditions':              label_encoder.classes_.tolist(),
    'random_search_best_cv':   float(random_search.best_score_),
    'grid_search_best_cv':     float(grid_search.best_score_),
    'test_accuracy':           float(accuracy),
    'test_precision':          float(precision),
    'test_recall':             float(recall),
    'test_f1':                 float(f1),
    'cv_5fold_scores':         [float(s) for s in cv_scores],
    'cv_5fold_mean':           float(cv_scores.mean()),
    'cv_5fold_std':            float(cv_scores.std()),
    'baseline_accuracy':       float(baseline_acc),
    'improvement':             float(accuracy - baseline_acc),
    'improvement_pct':         float((accuracy - baseline_acc) / baseline_acc * 100),
    'best_params':             best_params_clean,
    'total_time_seconds':      round(total_time, 1),
    'total_time_minutes':      round(total_time / 60, 1),
}

with open(models_dir / 'tuning_results.json', 'w') as f:
    json.dump(tuning_results, f, indent=2)

print(f"  ✓ Results saved        : models/tuning_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

sign = '+' if accuracy >= baseline_acc else ''
print(f"""
{'='*78}
  TASK 5 — TUNING COMPLETE
{'='*78}
  Dataset
    Training  : {len(df_train):,} reviews
    Test      : {len(df_test):,} reviews
    Conditions: {', '.join(label_encoder.classes_)}

  Performance
    Baseline accuracy          : {baseline_acc:.4f}
    Tuned model accuracy       : {accuracy:.4f}
    Improvement                : {sign}{accuracy - baseline_acc:.4f}  ({sign}{(accuracy-baseline_acc)/baseline_acc*100:.2f}%)
    5-fold CV mean (± std)     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
    RandomisedSearch best CV   : {random_search.best_score_:.4f}
    GridSearch best CV         : {grid_search.best_score_:.4f}

  Duration
    Total : {total_time/60:.1f} minutes

  Outputs saved
    models/tuned_pipeline.pkl
    models/label_encoder.pkl
    models/tuning_results.json
    outputs/figures/confusion_matrix_tuned.png
    outputs/figures/learning_curve_tuned.png
    outputs/figures/hyperparameter_sensitivity.png
    outputs/figures/feature_importance_tuned.png
{'='*78}
""")
