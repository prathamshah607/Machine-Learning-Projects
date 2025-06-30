# Emotion Classification with EfficientNetCNN on FER2013

## Overview
This project tackles the challenge of emotion recognition from facial images using the FER2013 subset. I built and evaluated a CNN model based on EfficientNet for classifying images into seven emotional categories. The goal was to approach or exceed human-level performance while maintaining interpretability and generalizability.

## Problem Statement
Facial emotion recognition is a multi-class image classification problem where the model must identify the emotional expression from a human face image. This has real-world applications in healthcare, surveillance, affective computing, and HCI.

The task is challenging due to:
- High intra-class variance (people show emotions differently)
- Inter-class similarity (fear vs. surprise, sad vs. neutral)
- Low-resolution grayscale data

## Dataset
- **Source**: FER2013, Kaggle
- **Classes**: 7 (happy, sad, surprise, disgust, anger, fear, neutral)
- **Training Images**: ~28,821  
- **Validation Images**: ~7,066  
- **Image Resolution**: 48x48 (upsampled ×3)
- **Preprocessing**:
  - Upscaled grayscale images to RGB 224x224
  - One-hot encoding of labels
  - Class weights to handle imbalance
  - ImageDataGenerator used for augmentation (shift, flip, etc.)

## Model Pipeline & Workflow

### General Workflow
1. **Preprocessing**: Loaded and transformed grayscale images into RGB.
2. **Model Architecture**: Transfer learning with `EfficientNet` backbone.
3. **Regularization Techniques**:
   - EarlyStopping
   - ReduceLROnPlateau
   - Dropout layers
4. **Training**: Used categorical cross-entropy and Adam optimizer.
5. **Evaluation**: Top-1 and Top-3 accuracy, confusion matrix, ROC-AUC, MCC.

### Why EfficientNet?
- Lightweight yet powerful
- Scales well across width/depth
- Good baseline for image classification without overfitting

### Results

| Metric                | Score    |
|----------------------|----------|
| Top-1 Accuracy        | **67%**  |
| Top-3 Accuracy        | **93%**  |
| F1 Score (macro avg)  | ~0.67    |
| Precision (macro avg) | ~0.67    |
| MCC                   | ~0.60    |
| Log Loss              | ~0.96  |
| ROC-AUC (avg)         | ~0.92 |

These are strong metrics considering no ensembles, no extra training data, and a severely imbalanced dataset.

### Misclassification Insights
- Emotions like **happy**, **surprise**, and **disgust** were classified with high confidence.
- More ambiguous emotions like **sad** and **fear** had lower precision.
- **Class imbalance** was partially mitigated by weighting and augmentation.

![image](https://github.com/user-attachments/assets/fac1bae2-4adc-4903-af98-dae29a0ac8bd)
![image](https://github.com/user-attachments/assets/8dbec29f-30e2-474d-ab4c-7495f6649cf0)
![image](https://github.com/user-attachments/assets/9b0f4a24-1e1b-411e-a4e3-5f8ca16562a5)

## Conclusion & Takeaways
- This model delivers **human-comparable accuracy** on a tough problem.
- Careful regularization and augmentation made the difference.
- Despite limitations (no ensemble, small grayscale images), the model generalizes well.
- In future iterations, attention mechanisms or ensemble techniques could push performance further.

## Appendix
- **Trained in**: TensorFlow / Keras
- **Hardware**: P100 GPU-backed runtime, Kaggle
- **Notebook**: [`model_interpretation.ipynb`](./model_interpretation.ipynb)
- **Implemented by**: Pratham Shah, 240905614, for Cryptonite Research Taskphase

