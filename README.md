# Spam-SMS-Classification-Feature-Engineering-Study

**Author:** Riddhi Sanghvi

---

## Overview

This project explores feature engineering techniques for SMS spam detection. Using a Logistic Regression classifier as the base model, three different text vectorization strategies are compared to understand how feature representation affects classification performance.

The goal is to classify SMS messages as **spam** or **ham (not spam)** and evaluate how increasingly sophisticated feature extraction impacts accuracy, precision, recall, and F1 score.

---

## Dataset

- **Source:** [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- **Size:** 5,572 SMS messages
- **Class Distribution:**
  - Ham (legitimate): 4,825 messages
  - Spam: 747 messages
- **Split:** 80% training (4,457 samples) / 20% test (1,115 samples), stratified

---

## Project Structure

```
spam_classification.ipynb
│
├── Section 1: Setup & Data Loading
├── Section 2: CountVectorizer (Baseline)
├── Section 3: TF-IDF Vectorizer
├── Section 4: N-grams (Bi-grams)
├── Section 5: Model Comparison & Visualizations
└── Section 6: Conclusion & Analysis
```

---

## Methodology

### Section 1 — Setup & Data Loading
Loads the SMS dataset, encodes labels (`ham → 0`, `spam → 1`), and performs a stratified train/test split.

### Section 2 — CountVectorizer (Baseline)
Transforms messages into raw word-count vectors using `CountVectorizer` with English stop words removed. Trains a Logistic Regression model and evaluates on the test set.

- Vocabulary size: **7,403 tokens**

### Section 3 — TF-IDF Vectorizer
Replaces raw counts with TF-IDF (Term Frequency–Inverse Document Frequency) weights, which up-weight rare, message-specific tokens and down-weight common words. This amplifies distinctive spam vocabulary (e.g., "prize", "winner", "claim").

### Section 4 — N-grams with CountVectorizer
Extends the feature space to include **bi-grams** (`ngram_range=(1,2)`), capturing two-word phrases like "free prize" or "call now" that are strong spam signals when considered together rather than as individual words.

- Feature matrix grows ~5× larger, but phrase-level context improves recall.

### Section 5 — Comparison & Visualizations
Plots confusion matrices and a grouped bar chart comparing all three models across all four metrics.

### Section 6 — Conclusion
Summarizes the trade-offs of each approach and identifies the best-performing configuration.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CountVectorizer (Baseline) | — | — | — | — |
| TF-IDF | — | — | — | — |
| N-grams (Bi-grams) | — | — | — | — |

> *Run the notebook to populate exact metric values from your environment.*

**Key Takeaway:** The progression CountVectorizer → TF-IDF → N-grams generally yields incremental F1 improvements. `TfidfVectorizer(ngram_range=(1,2))` — combining IDF re-weighting with phrase-level context — is typically the best production choice, balancing detection accuracy with manageable feature dimensionality.

---

## Requirements

```
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone this repository or download the notebook.
2. Open `S26_AISec_Assignment1_Riddhi_Sanghvi.ipynb` in Jupyter or Google Colab.
3. Run all cells sequentially (Runtime → Run All in Colab).
4. Metrics for each model will be printed in dictionary format for easy comparison.

---

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy** — Overall correctness
- **Precision** — Of predicted spam, how much is actually spam (minimizes false positives)
- **Recall** — Of actual spam, how much was caught (minimizes false negatives)
- **F1 Score** — Harmonic mean of precision and recall (primary metric for imbalanced classes)
