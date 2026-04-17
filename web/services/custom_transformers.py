"""
Custom Transformers for Model Loading
services/custom_transformer.py
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
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
    def __init__(self, max_features_per_class=50):
        self.max_features_per_class = max_features_per_class
        self.condition_vocabularies = {}
        self.condition_bigrams = {}
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
                stop_words='english', ngram_range=(1, 1), min_df=2
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
            all_words = ' '.join(texts).split()
            bigram_counter = Counter()
            for i in range(len(all_words) - 1):
                bigram_counter[f"{all_words[i]}_{all_words[i+1]}"] += 1
            self.condition_bigrams[condition] = set([bg for bg, _ in bigram_counter.most_common(30)])
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