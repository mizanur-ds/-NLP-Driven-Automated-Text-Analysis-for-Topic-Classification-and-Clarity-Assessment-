# 🧠 NLP-Driven Automated Text Analysis for Topic Classification and Clarity Assessment

## 📋 Project Overview

This project leverages Natural Language Processing (NLP) techniques for automated text analysis across two main tasks:

- **Topic Classification**: Classify paragraphs into predefined topics using a Multilayer Perceptron (MLP) classifier.
- **Text Clarity Classification**: Prototype to assess the clarity of text submissions.


---

## ✨ Key Features

### 1. Topic Classification
- **Model**: Multilayer Perceptron (MLP)
- **Optimization**: Grid search with 5-fold cross-validation
- **Metrics**: Precision, Recall, F1-score
- **Performance**:  
  - Macro average F1-score: **90%**  
  - Trivial baseline: **18%**

### 2. Text Clarity Classification (Prototype)
- **Model**: MLP classifier
- **Data Labeling**: Based on text length  
  - `≤ 800` words: *Clear Enough*  
  - `> 800` words: *Not Clear Enough*
- **Performance**:  
  - F1-score (*Clear Enough*): **0.94**  
  - F1-score (*Not Clear Enough*): **0.75**  
  - Baseline: Random guess

---

## 🧾 Dataset

**Total Entries**: 9,347  
**Features**:

- **Numerical**: `par_id`, `lexicon_count`, `difficult_words`
- **Categorical**: `paragraph`, `has_entity`, `last_editor_gender`, `category`, `text_clarity`

### 🔧 Preprocessing Steps

- Imputation of missing values (mean for numerical, most frequent for categorical)
- Text standardization (lowercasing, removal of special characters, etc.)
- Label encoding of categorical features
- Word embeddings using **SpaCy**

---

## ⚖️ Ethical Considerations

- **Bias**: Ensures training data is diverse and representative
- **Fairness**: Implements bias mitigation strategies
- **Transparency**: Provides clear user guidelines for improving text clarity

---

## 📊 Results

| Task                  | Metric        | Score | Baseline     |
|-----------------------|---------------|-------|--------------|
| Topic Classification  | F1 (macro)    | 90%   | 18%          |
| Text Clarity          | F1            | 94%   | Random Guess |

<!-- Figure 1: Confusion matrix heatmap -->
<div align="center">
  <img src="https://github.com/user-attachments/assets/4e075d7d-cf79-45b2-b9a3-050680e03c55" alt="Confusion matrix heatmap" width="600"/>
  <p><strong>Figure:</strong> Heat map of confusion matrix</p>
</div>

<!-- Figure 2: Confusion matrix heatmap for random guess -->
<div align="center">
  <img src="https://github.com/user-attachments/assets/8cf757cc-84e6-43ac-b02d-9c15d09e4512" alt="Confusion matrix heatmap for random guess" width="600"/>
  <p><strong>Figure:</strong> Heat map of confusion matrix (for random guess)</p>
</div>


- Confusion matrices and classification reports are available in the full report.

---

## 📌 Summary Discussion

This project successfully addresses:
- **Robust topic classification**, achieving high performance with minimal misclassification (<10% in most cases).
- **Effective clarity assessment prototype**, showing strong performance even with limited labeled data.

The use of **MLP** models demonstrates flexibility and effectiveness in handling non-linear, high-dimensional NLP tasks. Future improvements include enhanced labeling strategies for better clarity evaluation.

---

