# Technical Report: Fake News Detection
**Scope:** Phase 1 (Data Preprocessing & EDA) and Phase 2 (Traditional Machine Learning)

---

## Phase 1: Data Preprocessing & Exploratory Data Analysis (EDA)

### 1. Dataset Overview
The project processes a subset of the Kaggle Fake and Real News dataset. 
- **Initial Count:** 7,139 total articles
- **Class Balance:** Extremely well-balanced (3,436 Fake / 3,703 Real)
- **Features:** 	itle, 	ext, subject, date, label

### 2. Data Cleaning & Feature Engineering
- **Length Analysis:** The mean article length prior to cleaning is 422 words (median 406), indicating substantial text for NLP models to analyze.
- **Deduplication:** 14 duplicated articles and a few missing rows (subject, date) were removed, leaving **7,125 highly clean samples**.
- **Text Normalization:** A robust pipeline was applied (lowercasing, stripping URLs/HTML, punctuation removal, stopword removal, lemmatization).
- **Result:** The mean text length was compressed from 422 raw words down to **238 clean tokens**. This aggressive token reduction removes noise and drastically speeds up training.

---

## Phase 2: Traditional Machine Learning

### 1. Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Logistic Regression** | **0.9930** | 0.9930 | 0.9930 | 0.9930 | 0.9998 | 1.85s |
| **Naive Bayes** | **0.9747** | 0.9748 | 0.9747 | 0.9747 | 0.9981 | **0.01s** |

### 2. Analysis & Key Takeaways
1. **Linearly Separable Vocabulary:** Both models achieved staggering performance (>97%), proving that the vocabulary used by fake news authors and real news authors is highly distinct.
2. **Efficiency:** Logistic Regression achieved 99.30% accuracy in just 1.85 seconds. Naive Bayes is blisteringly fast (0.01s) making it perfect for real-time edge processing.



Key Takeaways from the Analysis:

Excellent Cleaning: Your Phase 1 preprocessing perfectly compressed the raw articles from 422 words down to 238 clean tokens, removing massive amounts of noise.
Phenomenal Baseline: The vocabulary used by "Fake News" writers vs "Real News" writers in this dataset is highly distinct. Because of this, your simple Logistic Regression model in Phase 2 achieved a staggering 99.30% accuracy in under 2 seconds!
The Challenge for Phase 3/4: Because the traditional Machine Learning models set the bar so incredibly high (99.3%), your Deep Learning and Transformer models won't be able to show much absolute improvement in basic accuracy. Their real value will be in capturing contextual nuance.