"""
Task 2B: Feature Tuning / Feature Selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================================

print("=" * 70)
print("TASK 2B: FEATURE TUNING AND SELECTION")
print("=" * 70)

# Load feature-engineered datasets
train_path = Path('data/processed/features_train.csv')
test_path = Path('data/processed/features_test.csv')

print("\n[1] Loading feature-engineered datasets...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"    Training data: {df_train.shape}")
print(f"    Test data: {df_test.shape}")

# ============================================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================================

print("\n" + "=" * 70)
print("2. PREPARING FEATURES FOR SELECTION")
print("=" * 70)

# Encode target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train['condition'])
y_test = label_encoder.transform(df_test['condition'])

print(f"\nTarget classes: {label_encoder.classes_}")

# Select numerical features (exclude non-feature columns)
exclude_cols = ['unique_id', 'drug_name', 'condition', 'review', 'review_date', 
                'sentiment_category', 'review_year', 'review_month', 'review_day']

feature_cols = [col for col in df_train.columns 
                if col not in exclude_cols 
                and df_train[col].dtype in ['int64', 'float64']]

X_train = df_train[feature_cols].fillna(0)
X_test = df_test[feature_cols].fillna(0)

print(f"\nInitial features: {len(feature_cols)}")

# ============================================================================
# 3. FEATURE IMPORTANCE USING RANDOM FOREST
# ============================================================================

print("\n" + "=" * 70)
print("3. FEATURE IMPORTANCE ANALYSIS (Random Forest)")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
for i, row in feature_importance.head(20).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<30} Importance: {row['importance']:.4f}")

# ============================================================================
# 4. STATISTICAL FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("4. STATISTICAL FEATURE SELECTION")
print("=" * 70)

# Method 1: ANOVA F-value
print("\n[1] ANOVA F-value Selection...")
selector_f = SelectKBest(f_classif, k=30)
selector_f.fit(X_train, y_train)
f_scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector_f.scores_
}).sort_values('score', ascending=False)

print("\nTop 15 Features (ANOVA F-value):")
for i, row in f_scores.head(15).iterrows():
    print(f"  {row['feature']:<30} Score: {row['score']:.2f}")

# Method 2: Mutual Information
print("\n[2] Mutual Information Selection...")
selector_mi = SelectKBest(mutual_info_classif, k=30)
selector_mi.fit(X_train, y_train)
mi_scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector_mi.scores_
}).sort_values('score', ascending=False)

print("\nTop 15 Features (Mutual Information):")
for i, row in mi_scores.head(15).iterrows():
    print(f"  {row['feature']:<30} Score: {row['score']:.4f}")

# ============================================================================
# 5. RECURSIVE FEATURE ELIMINATION
# ============================================================================

print("\n" + "=" * 70)
print("5. RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 70)

# Use a simpler model for RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000, random_state=42)
selector_rfe = RFECV(
    estimator=estimator,
    step=1,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=20,
    n_jobs=-1
)

print("\nRunning RFE (this may take a moment)...")
selector_rfe.fit(X_train, y_train)

# Get selected features
selected_features_rfe = [feature_cols[i] for i, selected in enumerate(selector_rfe.support_) if selected]

print(f"\nOptimal number of features: {selector_rfe.n_features_}")
print(f"Selected features: {len(selected_features_rfe)}")
print("\nSelected features:")
for i, feature in enumerate(selected_features_rfe):
    print(f"  {i+1:2d}. {feature}")

# ============================================================================
# 6. COMBINED FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 70)
print("6. COMBINED FEATURE SELECTION STRATEGY")
print("=" * 70)

# Combine top features from different methods
top_rf = feature_importance.head(25)['feature'].tolist()
top_f = f_scores.head(25)['feature'].tolist()
top_mi = mi_scores.head(25)['feature'].tolist()
top_rfe = selected_features_rfe[:25]

# Find features that appear in multiple lists
from collections import Counter
all_top_features = top_rf + top_f + top_mi + top_rfe
feature_votes = Counter(all_top_features)

# Select features that appear in at least 2 methods
final_features = [feature for feature, count in feature_votes.items() if count >= 2]

print(f"\nFeatures appearing in 2+ methods: {len(final_features)}")
print("\nFinal Selected Features:")
for i, feature in enumerate(final_features):
    votes = feature_votes[feature]
    print(f"  {i+1:2d}. {feature:<35} (Votes: {votes}/4)")

# ============================================================================
# 7. FEATURE VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("7. GENERATING FEATURE VISUALIZATIONS")
print("=" * 70)

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest Importance
ax1 = axes[0]
top_15_rf = feature_importance.head(15)
ax1.barh(range(len(top_15_rf)), top_15_rf['importance'].values)
ax1.set_yticks(range(len(top_15_rf)))
ax1.set_yticklabels(top_15_rf['feature'].values)
ax1.set_xlabel('Importance')
ax1.set_title('Top 15 Features - Random Forest')
ax1.invert_yaxis()

# Feature correlation heatmap (selected features)
ax2 = axes[1]
selected_corr = X_train[final_features[:10]].corr()
sns.heatmap(selected_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, ax=ax2)
ax2.set_title('Feature Correlations (Top 10 Selected)')

plt.tight_layout()
output_path = Path('outputs/figures/feature_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Feature visualization saved to: {output_path}")
plt.close()

# ============================================================================
# 8. SAVE SELECTED FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("8. SAVING TUNED DATASETS")
print("=" * 70)

# Create datasets with only selected features
X_train_tuned = X_train[final_features]
X_test_tuned = X_test[final_features]

print(f"\nFinal feature set shape:")
print(f"  - Training: {X_train_tuned.shape}")
print(f"  - Test: {X_test_tuned.shape}")

# Save the tuned datasets
train_tuned = pd.concat([
    df_train[['unique_id', 'drug_name', 'condition', 'review']],
    X_train_tuned
], axis=1)

test_tuned = pd.concat([
    df_test[['unique_id', 'drug_name', 'condition', 'review']],
    X_test_tuned
], axis=1)

train_tuned_path = Path('data/processed/train_tuned.csv')
test_tuned_path = Path('data/processed/test_tuned.csv')

train_tuned.to_csv(train_tuned_path, index=False)
test_tuned.to_csv(test_tuned_path, index=False)

print(f"\n✓ Tuned training data saved to: {train_tuned_path}")
print(f"✓ Tuned test data saved to: {test_tuned_path}")

# Save feature list
feature_list_path = Path('models/selected_features.txt')
with open(feature_list_path, 'w') as f:
    for feature in final_features:
        f.write(f"{feature}\n")
print(f"✓ Selected features list saved to: {feature_list_path}")

# Save label encoder
encoder_path = Path('models/label_encoder.pkl')
joblib.dump(label_encoder, encoder_path)
print(f"✓ Label encoder saved to: {encoder_path}")

# ============================================================================
# 9. SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("9. FEATURE TUNING SUMMARY")
print("=" * 70)

print(f"\nInitial features: {len(feature_cols)}")
print(f"Features after tuning: {len(final_features)}")
print(f"Reduction: {len(feature_cols) - len(final_features)} features ({((len(feature_cols) - len(final_features))/len(feature_cols)*100):.1f}%)")

print("\nFeature Categories in Final Set:")
category_counts = {
    'Text Statistics': 0,
    'Sentiment': 0,
    'Medical Keywords': 0,
    'Rating': 0,
    'Usefulness': 0,
    'Drug': 0
}

for feature in final_features:
    if feature in ['char_count', 'word_count', 'avg_word_length', 'sentence_count']:
        category_counts['Text Statistics'] += 1
    elif 'sentiment' in feature:
        category_counts['Sentiment'] += 1
    elif 'keyword' in feature or 'side_effect' in feature:
        category_counts['Medical Keywords'] += 1
    elif 'rating' in feature or 'is_high' in feature or 'is_low' in feature:
        category_counts['Rating'] += 1
    elif 'useful' in feature:
        category_counts['Usefulness'] += 1
    elif 'drug' in feature:
        category_counts['Drug'] += 1

for category, count in category_counts.items():
    if count > 0:
        print(f"  - {category}: {count} features")

print("\n" + "=" * 70)
print("FEATURE TUNING COMPLETE")
print("=" * 70)