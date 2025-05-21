
![image](https://github.com/user-attachments/assets/9c5db9bb-2524-4a0f-87cf-4758d655aea7)
![image](https://github.com/user-attachments/assets/0f18030c-23c5-4bae-9e56-92e96a7f682c)
![image](https://github.com/user-attachments/assets/de94d2f6-2ef3-495e-b143-c88fffede3ec)
![image](https://github.com/user-attachments/assets/3335d3d1-ab0b-4ba6-b7f6-5e4ec2431ec8)


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

![Histograms](figures/histograms.png)  
*Fig-3/4: Distribution of numerical features.*

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

![Confusion Matrix](figures/confusion_matrix.png)  
*Fig-11: Heatmap of topic classification results.*

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
