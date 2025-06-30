# Emotion Classification with EfficientNetCNN on FER2013

## Overview
This project addresses the challenge of emotion recognition from facial images using the FER2013 dataset. I developed and evaluated a CNN model based on EfficientNetB0 to classify grayscale images into seven distinct emotional categories.

The primary goal: reach or exceed human-level accuracy while maintaining interpretability and building a reproducible, resource-efficient pipeline suitable for real-world applications.

---

## Problem Statement
Facial emotion recognition is a multi-class classification problem where the model must identify human emotion from a single face image. The task is non-trivial due to:

- High intra-class variance — individuals express the same emotion differently.
- Low inter-class separation — expressions like fear and surprise often look similar.
- Low-quality input — the original dataset provides only 48x48 grayscale images.

Despite the limitations, this problem has valuable real-world implications in:

- Healthcare and mental state monitoring  
- Surveillance and crowd analysis  
- Human-Computer Interaction

---

## Dataset

- **Source**: FER2013 (Kaggle)
- **Classes**: 7 — `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- **Training Images**: 20,179  
- **Validation Images**: 2,117  
- **Image Resolution**: 48x48 grayscale (upsampled ×3 → 224x224 RGB)

### Preprocessing
- Images upscaled to 224×224 and converted to 3 channels for EfficientNet compatibility
- One-hot encoding for categorical labels
- Image augmentation: horizontal flips, zooms, and shifts using `ImageDataGenerator`
- Class weights applied to reduce the effect of imbalance

---

## Model Pipeline & Workflow

### Workflow
1. Preprocess and augment input images
2. Load base EfficientNetB0 without top layers
3. Add custom dense layers with dropout for regularization
4. Compile using categorical crossentropy and Adam optimizer
5. Train using early stopping and learning rate scheduling
6. Evaluate with classification metrics and interpret predictions

---

## Why EfficientNet?

I chose EfficientNet because:

---

## Model Architecture

- **Base**: EfficientNetB0 (ImageNet pretrained, include_top=False)
- **Top Layers**:
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.35)
  - Dense(128, activation='relu')
  - Dropout(0.35)
  - Dense(7, activation='softmax')

Training was in two phases:
- Frozen base for initial epochs to train the dense layers
- Unfrozen base for fine-tuning the entire model

---

## Evaluation Results

### Metrics on Validation Set

| Metric                  | Value      |
|------------------------|------------|
| Top-1 Accuracy          | 67.0%      |
| Top-3 Accuracy          | 92.65%      |
| Weighted F1 Score          | 0.6677      |
| Weighted Precision         | 0.6739      |
| Weighted Recall            | 0.67      |
| Log Loss                | 0.958      |
| Matthews Corr. Coef.    | 0.603      |
| ROC-AUC (avg)     | 0.916      |

### Per-Class Performance

| Class     | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Angry     | 0.56     | 0.60  | 0.58    |
| Disgust   | 0.59     | 0.72  | 0.65    |
| Fear      | 0.57     | 0.46  | 0.48    |
| Happy     | 0.91     | 0.83  | 0.87    |
| Neutral   | 0.58     | 0.72  | 0.64    |
| Sad       | 0.56     | 0.51  | 0.53    |
| Surprise  | 0.71     | 0.85  | 0.77    |

---

## Misclassification Insights

- `happy`, `surprise`, and `disgust` were consistently predicted with high confidence.
- `fear`, `sad`, and `angry` were frequently confused with `neutral`.
- Class imbalance remains a challenge, though mitigated via weighting and augmentation.

---

![image](https://github.com/user-attachments/assets/fac1bae2-4adc-4903-af98-dae29a0ac8bd)
![image](https://github.com/user-attachments/assets/8dbec29f-30e2-474d-ab4c-7495f6649cf0)
![image](https://github.com/user-attachments/assets/9b0f4a24-1e1b-411e-a4e3-5f8ca16562a5)

---

## Conclusion & Takeaways

This EfficientNetB0-based model achieved human-comparable performance on a difficult, low-resolution emotion classification task. With a modest number of parameters, it reached:

- 67.0% Top-1 accuracy
- 92.65% Top-3 accuracy
- 0.67 average F1 score
without ensembling, external data, or hyperparameter grid search.

---

## Appendix

- **Framework**: TensorFlow / Keras
- **Hardware**: Kaggle Notebook (P100 GPU)
- **Notebook**: [model_interpretation.ipynb](./model_interpretation.ipynb) for inference, [actual.ipynb](./actual.ipynb) for training
- **Author**: Pratham Shah  
- **ID**: 240905614  
- **Task**: Cryptonite Research Taskphase

