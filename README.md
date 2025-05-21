# NLP-Driven Automated Text Analysis for Topic Classification and Clarity Assessment

## Project Overview
This project focuses on two key tasks:
1. **Topic Classification**: Using a Multilayer Perceptron (MLP) classifier to categorize paragraphs into predefined topics.
2. **Text Clarity Assessment**: Developing a prototype MLP classifier to evaluate the clarity of text submissions.

The models were optimized via grid search with 5-fold cross-validation and evaluated using precision, recall, and F1-score metrics.

## Table of Contents
- [Executive Summary](#executive-summary)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [References](#references)

---

## Executive Summary
### Task 1: Topic Classification
- **Model**: MLP classifier with hyperparameter tuning (hidden layers, learning rate).
- **Performance**: Achieved **90% macro-average F1-score**, outperforming a trivial baseline (18%).
- **Key Metrics**: Precision, recall, and F1-score were used to address class imbalance.

### Task 2: Text Clarity Prototype
- **Model**: MLP classifier trained on a labeled subset (100 samples).
- **Performance**: **95% accuracy** on validation data, significantly better than random guessing.
- **Labeling**: Clarity labels were inferred based on paragraph length due to limited ground truth.

---

## Data Exploration
### Dataset Overview
- **Size**: 9,347 entries, 8 columns.
- **Features**:
  - Numerical: `lexicon_count`, `difficult_words`.
  - Categorical: `paragraph`, `category`, `text_clarity`, etc.
- **Issues**: Missing values, duplicates, and inconsistent capitalization (e.g., "Biography" vs. "biography").

### Key Insights
- **Imbalance**: Categorical features (e.g., `category`, `last_editor_gender`) were skewed.
- **Outliers**: Observed in `lexicon_count` and `difficult_words` (see histograms and boxplots below).

<div align="center">
  <img src="https://github.com/user-attachments/assets/8913048e-b22b-4353-9727-ad8a38b96ac4" alt="Image 1" width="500" height="300" style="display:inline-block; margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/0e00813c-99ce-4758-9a2d-25deb57ef48c" alt="Image 2" width="500" height="300" style="display:inline-block;"/>
  <strong>Fig-1:</strong> Distribution of numerical features
</div> 

---

## Data Preprocessing
1. **Cleaning**:
   - Filled missing values using mean (numerical) and mode (categorical).
   - Standardized text (lowercase, removed special characters).
2. **Splitting**:
   - 90% training, 10% test (stratified by target).
3. **Encoding**:
   - Label encoding for categories.
   - SpaCy word embeddings for text (`paragraph` column).

---

## Model Implementation
### Topic Classification
- **Architecture**: MLP with grid-searched hyperparameters:
  - `hidden_layer_sizes`: `(20, 20)` and `(30,)`.
  - `learning_rate_init`: `0.0001` and `0.001`.
- **Evaluation**:
  - Confusion matrix and classification report (below).
 <div align="center">
  <img src="https://github.com/user-attachments/assets/413fedb9-4a45-4751-ae64-a51dface6ca5" alt="Image 1" width="500" height="300" style="display:inline-block; margin-right: 10px;"/>
  <strong>Fig-2:</strong> Heatmap of topic classification results
</div> 


### Text Clarity Prototype
- **Labeling**: Assigned clarity based on paragraph length (`>800 words = unclear`).
- **Results**: 95% validation accuracy, no overfitting.

---

## Results
### Topic Classification
| Class                  | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| Artificial Intelligence| 0.92      | 0.88   | 0.90     |
| Philosophy             | 0.89      | 0.85   | 0.87     |

### Text Clarity
- **F1-Score**: 0.94 (clear) vs. 0.18 (trivial baseline).

---

## Ethical Considerations
- **Bias Risk**: Clarity judgments based on length may disadvantage long-form content.
- **Mitigation**:
  - Use diverse training data.
  - Provide user feedback for rejected submissions.

---

## Conclusion
- Both models met client requirements with **<10% misclassification** (topic) and **95% accuracy** (clarity).
- **Recommendation**: Expand labeled data for clarity assessment to reduce reliance on heuristic labeling.

---

## References
1. Windeatt, T. (2006). *Accuracy/Diversity and Ensemble MLP Classifier Design*. IEEE.  
   [Link](https://ieeexplore.ieee.org/abstract/document/1687930)
2. Shekar, B. H., & Dagnew, G. (2019). *Grid Search-Based Hyperparameter Tuning*. IEEE.  
   [Link](https://ieeexplore.ieee.org/abstract/document/8882943)
