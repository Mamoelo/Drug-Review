"""
Task 1: Data Gathering and Cleaning - Training Dataset
File: clean_train_data.py
Purpose: Clean drugsComTrain_raw.csv and output cleaned_train_data.csv
"""

import pandas as pd
import html
import re
from datetime import datetime

# ============================================================================
# 1. DATA GATHERING
# ============================================================================

print("=" * 60)
print("TASK 1: DATA GATHERING AND CLEANING - TRAINING DATASET")
print("=" * 60)

# Load the training dataset
df_train = pd.read_csv('data/raw/drugsComTrain_raw.csv')

print(f"\n[1] Original Training Dataset Shape: {df_train.shape}")
print(f"    Columns: {list(df_train.columns)}")

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("2. MISSING VALUES ANALYSIS")
print("=" * 60)

missing_values = df_train.isnull().sum()
missing_percentage = (df_train.isnull().sum() / len(df_train)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage (%)': missing_percentage.round(2)
})

print("\nMissing Values Summary:")
print(missing_df[missing_df['Missing Count'] > 0])

if missing_values.sum() == 0:
    print("\n✓ No missing values found in training dataset")

# ============================================================================
# 3. DATA CLEANING FUNCTIONS
# ============================================================================

def clean_review_text(text):
    """Clean HTML entities and special characters from review text"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove extra quotes that wrap the entire review
    text = text.strip('"')
    
    while text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    # Replace common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#039;', "'")
    text = text.replace('&rsquo;', "'")
    text = text.replace('&ldquo;', '"')
    text = text.replace('&rdquo;', '"')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def parse_date(date_str):
    """Parse date from '28-Feb-12' format to datetime"""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(date_str, format='%d-%b-%y')
    except:
        return None


def clean_condition(condition):
    """Standardize condition names"""
    if pd.isna(condition):
        return condition
    
    condition = str(condition).strip()
    condition_lower = condition.lower()
    
    if 'depression' in condition_lower:
        return 'Depression'
    elif 'high blood pressure' in condition_lower or 'hypertension' in condition_lower:
        return 'High Blood Pressure'
    elif 'diabetes' in condition_lower and 'type 2' in condition_lower:
        return 'Diabetes, Type 2'
    elif 'type 2 diabetes' in condition_lower:
        return 'Diabetes, Type 2'
    else:
        return condition


# ============================================================================
# 4. APPLY CLEANING
# ============================================================================

print("\n" + "=" * 60)
print("3. APPLYING DATA CLEANING")
print("=" * 60)

# Create a copy for cleaning
df_train_clean = df_train.copy()

# Clean review text
print("\n[1] Cleaning review text (HTML entities, quotes, special characters)...")
df_train_clean['review_clean'] = df_train_clean['review'].apply(clean_review_text)

# Parse dates
print("[2] Parsing dates to datetime format...")
df_train_clean['date_parsed'] = df_train_clean['date'].apply(parse_date)

# Standardize condition names
print("[3] Standardizing condition names...")
df_train_clean['condition_clean'] = df_train_clean['condition'].apply(clean_condition)

# Check unique conditions before filtering
print("\nUnique conditions in training dataset (top 20):")
print(df_train_clean['condition_clean'].value_counts().head(20))

# ============================================================================
# 5. FILTER FOR TARGET CONDITIONS (Business Objective)
# ============================================================================

print("\n" + "=" * 60)
print("4. FILTERING FOR TARGET CONDITIONS")
print("=" * 60)

target_conditions = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']

print("\nChecking for target conditions:")
for condition in target_conditions:
    count = len(df_train_clean[df_train_clean['condition_clean'] == condition])
    print(f"  - {condition}: {count} reviews")

# Filter the dataset
df_train_filtered = df_train_clean[df_train_clean['condition_clean'].isin(target_conditions)].copy()

print(f"\n[Filtered] Original: {len(df_train_clean)} rows")
print(f"           Kept: {len(df_train_filtered)} rows")
print(f"           Removed: {len(df_train_clean) - len(df_train_filtered)} rows")

# ============================================================================
# 6. OUTLIER DETECTION (IQR Method)
# ============================================================================

print("\n" + "=" * 60)
print("5. OUTLIER DETECTION - USEFULCOUNT")
print("=" * 60)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound

outliers, lower, upper = detect_outliers_iqr(df_train_filtered, 'usefulCount')

print(f"\nUsefulCount Statistics:")
print(f"  - Q1 (25th percentile): {df_train_filtered['usefulCount'].quantile(0.25):.2f}")
print(f"  - Q3 (75th percentile): {df_train_filtered['usefulCount'].quantile(0.75):.2f}")
print(f"  - IQR: {df_train_filtered['usefulCount'].quantile(0.75) - df_train_filtered['usefulCount'].quantile(0.25):.2f}")
print(f"  - Lower bound: {lower:.2f}")
print(f"  - Upper bound: {upper:.2f}")
print(f"  - Outliers detected: {len(outliers)} rows")
print(f"  - Percentage: {(len(outliers)/len(df_train_filtered)*100):.2f}%")

# Rating validation
print("\nRating Validation:")
print(f"  - Min rating: {df_train_filtered['rating'].min()}")
print(f"  - Max rating: {df_train_filtered['rating'].max()}")
invalid_ratings = df_train_filtered[(df_train_filtered['rating'] < 1) | (df_train_filtered['rating'] > 10)]
print(f"  - Invalid ratings (<1 or >10): {len(invalid_ratings)}")

# ============================================================================
# 7. DESCRIPTIVE ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("6. DESCRIPTIVE ANALYSIS")
print("=" * 60)

print("\n[1] Numerical Columns Summary:")
print(df_train_filtered[['rating', 'usefulCount']].describe())

print("\n[2] Condition Distribution (Target Conditions Only):")
condition_counts = df_train_filtered['condition_clean'].value_counts()
for condition, count in condition_counts.items():
    percentage = (count / len(df_train_filtered)) * 100
    print(f"  - {condition}: {count} reviews ({percentage:.1f}%)")

print("\n[3] Rating Distribution:")
rating_counts = df_train_filtered['rating'].value_counts().sort_index()
for rating, count in rating_counts.items():
    percentage = (count / len(df_train_filtered)) * 100
    print(f"  - Rating {int(rating)}: {count} reviews ({percentage:.1f}%)")

print(f"\n[4] Date Range:")
print(f"  - Earliest review: {df_train_filtered['date_parsed'].min().strftime('%d %B %Y')}")
print(f"  - Latest review: {df_train_filtered['date_parsed'].max().strftime('%d %B %Y')}")

print("\n[5] Average Rating by Condition:")
avg_ratings = df_train_filtered.groupby('condition_clean')['rating'].mean().round(2)
for condition, avg in avg_ratings.items():
    print(f"  - {condition}: {avg}/10")

print("\n[6] Top 5 Most Useful Reviews:")
top_useful = df_train_filtered.nlargest(5, 'usefulCount')[
    ['drugName', 'condition_clean', 'rating', 'usefulCount', 'review_clean']
]
for idx, row in top_useful.iterrows():
    print(f"\n  Drug: {row['drugName']} | Condition: {row['condition_clean']}")
    print(f"  Rating: {row['rating']}/10 | Useful Count: {row['usefulCount']}")
    print(f"  Review: {row['review_clean'][:150]}...")

# ============================================================================
# 8. DUPLICATE CHECK
# ============================================================================

print("\n" + "=" * 60)
print("7. DUPLICATE CHECK")
print("=" * 60)

duplicates = df_train_filtered.duplicated(subset=['uniqueID']).sum()
print(f"\nDuplicate uniqueID values: {duplicates}")

if duplicates == 0:
    print("✓ No duplicates found")

# ============================================================================
# 9. SAVE CLEANED DATASET
# ============================================================================

print("\n" + "=" * 60)
print("8. SAVING CLEANED DATASET")
print("=" * 60)

# Select final columns
df_train_final = df_train_filtered[[
    'uniqueID', 
    'drugName', 
    'condition_clean', 
    'review_clean', 
    'rating', 
    'date_parsed', 
    'usefulCount'
]].copy()

# Rename columns to clean names
df_train_final.columns = [
    'unique_id', 
    'drug_name', 
    'condition', 
    'review', 
    'rating', 
    'review_date', 
    'useful_count'
]

# Save to CSV
output_path = 'data/processed/cleaned_train_data.csv'
df_train_final.to_csv(output_path, index=False)

print(f"\n✓ Cleaned training dataset saved to: {output_path}")
print(f"  Final shape: {df_train_final.shape}")
print(f"  Final columns: {list(df_train_final.columns)}")

print("\n" + "=" * 60)
print("TRAINING DATASET CLEANING COMPLETE")
print("=" * 60)