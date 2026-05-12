**MODEL CREATION DETAIL**

Drug Review Classification System

scripts/train_model.py | scripts/tune_model.py | web/services/custom_transformers.py

# 1\. Overview

The classification model is a soft-voting ensemble combining three complementary classifiers: Logistic Regression, Random Forest, and XGBoost. Each classifier operates on the same feature matrix, but contributes a different decision-making strategy. The ensemble aggregates their individual class probability outputs and selects the class with the highest mean probability across all three members.

All feature extraction and classification steps are encapsulated in a single scikit-learn Pipeline object. This ensures that the exact same transformations applied during training are automatically applied at inference time, preventing data leakage and making deployment straightforward.

# 2\. Feature Engineering

The feature matrix fed to the ensemble is built using a FeatureUnion of four parallel transformers, each contributing a different type of signal from the review text:

## 2.1 TF-IDF Vectoriser

The primary text representation. Each review is converted into a sparse vector of 2,500 weighted term scores (unigrams and bigrams). The sublinear_tf=True setting applies log scaling to term frequency, preventing reviews that repeat the same word many times from dominating the feature space. English stop words are removed. Terms appearing in fewer than 2 documents are excluded.

TfidfVectorizer(max_features=2500, ngram_range=(1,2), min_df=2, max_df=0.85,

stop_words='english', sublinear_tf=True)

## 2.2 TextFeatureExtractor (custom transformer)

A custom scikit-learn transformer that extracts 13 statistical features from the raw text of each review, capturing structural and stylistic properties that TF-IDF cannot represent:

| **Feature**         | **Description**                              |
| ------------------- | -------------------------------------------- |
| char_count          | Total character length of the review         |
| ---                 | ---                                          |
| word_count          | Number of whitespace-delimited tokens        |
| ---                 | ---                                          |
| unique_words        | Vocabulary size (distinct tokens)            |
| ---                 | ---                                          |
| avg_word_length     | Mean character count per word                |
| ---                 | ---                                          |
| sentence_count      | Number of sentence-ending punctuation marks  |
| ---                 | ---                                          |
| avg_sentence_length | word_count / sentence_count                  |
| ---                 | ---                                          |
| exclamation_count   | Number of ! characters                       |
| ---                 | ---                                          |
| question_count      | Number of ? characters                       |
| ---                 | ---                                          |
| uppercase_ratio     | Proportion of uppercase characters           |
| ---                 | ---                                          |
| ttr                 | Type-Token Ratio - unique_words / word_count |
| ---                 | ---                                          |
| log_char_count      | log1p(char_count) - compressed scale         |
| ---                 | ---                                          |
| log_word_count      | log1p(word_count) - compressed scale         |
| ---                 | ---                                          |
| sqrt_word_count     | sqrt(word_count) - alternative scale         |
| ---                 | ---                                          |

These features capture that Depression reviews tend to be longer and more detailed (higher word count, lower TTR) than High Blood Pressure or Type 2 Diabetes reviews - a pattern confirmed in the descriptive analysis.

## 2.3 SentimentFeatureExtractor (custom transformer)

Uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon to score the emotional tone of each review. VADER is specifically calibrated for social media and consumer-generated text, making it well-suited to drug review language. Eight sentiment-derived features are extracted per review:

| **Feature**         | **Description**                                 |
| ------------------- | ----------------------------------------------- |
| compound_normalised | (compound + 1) / 2 - rescaled to \[0, 1\]       |
| ---                 | ---                                             |
| pos                 | Proportion of positive sentiment tokens         |
| ---                 | ---                                             |
| neg                 | Proportion of negative sentiment tokens         |
| ---                 | ---                                             |
| neu                 | Proportion of neutral sentiment tokens          |
| ---                 | ---                                             |
| abs_compound        | Absolute value of compound - measures intensity |
| ---                 | ---                                             |
| is_positive         | Binary flag: compound >= 0.05                   |
| ---                 | ---                                             |
| is_negative         | Binary flag: compound <= -0.05                  |
| ---                 | ---                                             |
| pos_plus_neu        | pos + neu - positive or neutral combined score  |
| ---                 | ---                                             |

The descriptive analysis confirms that sentiment differs meaningfully between conditions: High Blood Pressure reviews show the most negative compound scores (mean -0.130), Depression reviews show moderate negativity (-0.038), and Type 2 Diabetes reviews are closest to neutral (-0.003). These differences make sentiment a discriminative signal for the classifier.

## 2.4 LearnedVocabularyExtractor (custom transformer)

Learns the top 50 most characteristic unigrams and bigrams per condition from the training set. At inference time, it counts how many of these condition-specific terms appear in the input text, producing a 3-column binary/count feature matrix (one column per condition). This gives the model an explicit signal about condition-specific medical terminology that TF-IDF may dilute across the full vocabulary.

Examples of learned vocabulary per condition:

- Depression: sertraline, antidepressant, anxiety, hopeless, mood, sleep, motivation
- High Blood Pressure: lisinopril, blood pressure, hypertension, dizziness, amlodipine, metoprolol
- Type 2 Diabetes: metformin, glucose, insulin, blood sugar, hba1c, jardiance, weight

## 2.5 Combined Feature Matrix

The four transformers run in parallel and their outputs are horizontally stacked by FeatureUnion into a single sparse matrix:

| **Component**              | **Features** | **Representation**     |
| -------------------------- | ------------ | ---------------------- |
| TF-IDF                     | 2,500        | Sparse float matrix    |
| ---                        | ---          | ---                    |
| TextFeatureExtractor       | 13           | Dense float matrix     |
| ---                        | ---          | ---                    |
| SentimentFeatureExtractor  | 8            | Dense float matrix     |
| ---                        | ---          | ---                    |
| LearnedVocabularyExtractor | ~150         | Dense count matrix     |
| ---                        | ---          | ---                    |
| Total                      | ~2,671       | Combined sparse matrix |
| ---                        | ---          | ---                    |

# 3\. Classifiers

## 3.1 Logistic Regression

Learns a linear decision boundary in the high-dimensional TF-IDF feature space. Chosen because:

- It natively outputs calibrated class probabilities, required for soft voting.
- It performs well on sparse, high-dimensional text data where linear separability is often sufficient.
- class_weight='balanced' compensates for the Depression majority class without requiring oversampling.

LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, random_state=42)

## 3.2 Random Forest

An ensemble of 100 decision trees trained using bootstrap aggregation (bagging). Chosen because:

- It handles the mixed sparse/dense feature matrix robustly without feature scaling.
- It reduces variance by averaging predictions across many trees, reducing overfitting on minority classes.
- Feature importance scores from the fitted forest were used in feature selection (tune_features.py) to identify and remove low-predictive features.

RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced',

n_jobs=-1, random_state=42)

## 3.3 XGBoost

A gradient-boosted tree ensemble that sequentially corrects the residual errors of prior trees. Chosen because:

- It achieves the highest individual accuracy (97.19%) of the three classifiers.
- It handles class imbalance through the scale_pos_weight mechanism.
- Its sequential boosting strategy complements the parallel bagging of Random Forest.

XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.2,

eval_metric='mlogloss', random_state=42)

## 3.4 Soft Voting Ensemble

The three classifiers are combined using scikit-learn's VotingClassifier with voting='soft'. At prediction time, each classifier returns a probability vector over the three condition classes. These vectors are averaged (with weights \[1, 2, 3\] giving XGBoost the highest influence) and the class with the highest mean probability is selected.

VotingClassifier(estimators=\[('lr', lr), ('rf', rf), ('xgb', xgb)\],

voting='soft', weights=\[1, 2, 3\])

Soft voting was chosen over hard voting (majority vote) because it uses the full confidence distribution from each model rather than just their top prediction, making the ensemble more robust when two models disagree strongly.

# 4\. Training Procedure

Script: scripts/train_model.py

- Data loaded from data/processed/cleaned_train_data.csv and cleaned_test_data.csv.
- Labels encoded with LabelEncoder and saved to models/label_encoder.pkl for consistent decoding at inference.
- Training set: 14,111 reviews. Test set: 4,751 reviews. Split is stratified to preserve the 65/18/17 class distribution.
- The full Pipeline (FeatureUnion + VotingClassifier) is fitted on the training set in a single call to pipeline.fit(X_train, y_train).
- 5-fold cross-validation is run on the training set before evaluating on the held-out test set.
- The fitted pipeline is saved to models/pipeline.pkl using joblib. Metadata (accuracy, precision, recall, F1, training date, sample counts) is written to models/model_metadata.json.

## 4.1 Base Model Performance

| **Metric** | **Depression** | **High Blood Pressure** | **Type 2 Diabetes** | **Overall** |
| ---------- | -------------- | ----------------------- | ------------------- | ----------- |
| Accuracy   | -              | -                       | -                   | 94.63%      |
| ---        | ---            | ---                     | ---                 | ---         |
| Precision  | -              | -                       | -                   | 94.76%      |
| ---        | ---            | ---                     | ---                 | ---         |
| Recall     | -              | -                       | -                   | 94.63%      |
| ---        | ---            | ---                     | ---                 | ---         |
| F1 Score   | -              | -                       | -                   | 94.54%      |
| ---        | ---            | ---                     | ---                 | ---         |

# 5\. Hyperparameter Tuning

Script: scripts/tune_model.py

Tuning was performed in two sequential stages to balance search quality against computational cost. Total tuning time: 154.7 minutes.

## 5.1 Stage 1 - RandomizedSearchCV

An initial broad random search over a wide parameter space was run with 5-fold cross-validation. This stage rapidly identifies promising regions of the hyperparameter space without exhaustively evaluating every combination.

RandomizedSearchCV(pipeline, param_distributions, n_iter=30, cv=5,

scoring='accuracy', n_jobs=-1, random_state=42)

Best CV accuracy from random search: 96.17%

## 5.2 Stage 2 - GridSearchCV

A targeted grid search around the best parameters found in Stage 1, using 5-fold cross-validation. This refines the solution within the promising region identified by random search.

GridSearchCV(pipeline, refined_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

Best CV accuracy from grid search: 96.19%

## 5.3 Optimal Hyperparameters

| **Parameter**                    | **Optimal Value**               |
| -------------------------------- | ------------------------------- |
| classifier*\_lr*\_C              | 10.0                            |
| ---                              | ---                             |
| classifier*\_rf*\_n_estimators   | 100                             |
| ---                              | ---                             |
| classifier*\_rf*\_max_depth      | 15                              |
| ---                              | ---                             |
| classifier*\_xgb*\_n_estimators  | 200                             |
| ---                              | ---                             |
| classifier*\_xgb*\_max_depth     | 7                               |
| ---                              | ---                             |
| classifier*\_xgb*\_learning_rate | 0.2                             |
| ---                              | ---                             |
| classifier\_\_weights            | \[1, 2, 3\] (LR:1, RF:2, XGB:3) |
| ---                              | ---                             |
| features*\_tfidf*\_max_features  | 2,500                           |
| ---                              | ---                             |
| features*\_tfidf*\_ngram_range   | (1, 2)                          |
| ---                              | ---                             |

## 5.4 Tuned Model Performance

| **Metric**                | **Value**                        |
| ------------------------- | -------------------------------- |
| Test Accuracy             | 97.39%                           |
| ---                       | ---                              |
| Test Precision            | 97.38%                           |
| ---                       | ---                              |
| Test Recall               | 97.39%                           |
| ---                       | ---                              |
| Test F1                   | 97.38%                           |
| ---                       | ---                              |
| 5-Fold CV Mean            | 96.61%                           |
| ---                       | ---                              |
| 5-Fold CV Std             | 0.28%                            |
| ---                       | ---                              |
| Improvement over baseline | +2.46 percentage points (+2.59%) |
| ---                       | ---                              |

The low CV standard deviation (0.28%) confirms the model generalises stably across different data splits and is not overfitting to any particular partition.

# 6\. Model Persistence and Loading

The tuned pipeline is saved to models/tuned_pipeline.pkl using joblib. The label encoder is saved separately to models/label_encoder.pkl. At inference time, web/model.py loads both files and uses them to transform raw text input and decode numeric predictions back to condition names.

The custom transformers (TextFeatureExtractor, SentimentFeatureExtractor, LearnedVocabularyExtractor) are defined in web/services/custom_transformers.py. This module must be importable at load time for pickle to reconstruct the pipeline correctly - which is why the project root is added to sys.path in run.py.

# 7\. Iterative Decisions During Model Development

| **Step**          | **Initial Decision**        | **Problem**                         | **Final Decision**                          |
| ----------------- | --------------------------- | ----------------------------------- | ------------------------------------------- |
| Missing values    | Placeholder text injection  | Accuracy loss - noise introduced    | Listwise deletion                           |
| ---               | ---                         | ---                                 | ---                                         |
| Outliers          | 3-sigma removal             | Lost informative high-count reviews | 99th percentile capping                     |
| ---               | ---                         | ---                                 | ---                                         |
| Text vectoriser   | CountVectorizer             | Overweighted common words           | TF-IDF with sublinear scaling               |
| ---               | ---                         | ---                                 | ---                                         |
| Feature selection | Keep all features           | Overfitting on training set         | RF importance scoring - drop low-predictive |
| ---               | ---                         | ---                                 | ---                                         |
| Model selection   | Individual classifiers only | Lower accuracy ceiling              | Soft voting ensemble                        |
| ---               | ---                         | ---                                 | ---                                         |
| Train-test split  | 70/30 random split          | Unstable metrics across runs        | Stratified 80/20                            |
| ---               | ---                         | ---                                 | ---                                         |
| Tuning strategy   | Full GridSearchCV           | Excessive runtime (>6 hours)        | RandomSearch then refined GridSearch        |
| ---               | ---                         | ---                                 | ---                                         |
| Ensemble weights  | Equal weights \[1,1,1\]     | Under-utilised XGBoost strength     | Weighted \[1,2,3\] favouring XGBoost        |
| ---               | ---                         | ---                                 | ---                                         |