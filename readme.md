# Cryptonite Research AI Subsystem Taskphase by Pratham Shah

## Task 1:

**Heart Dataset:**
1. Linear Regression
2. vanilla Logistic Regression
3. Hierarchichal classification using cross val and multiple models - ***winner***.

**Electricity Dataset:**
1. LSTM
2. XGBoost - ***winner***.
3. RandomForestRegressor

Because the EDA and feature engineering are same for all the files, I have commented their details majorly in the LSTM file for electricity and the Hierarchical classifier in the heart dataset.

---

## Task 2:

**DeepWeeds Dataset:**
- Transfer learning through ResNet 50 trainable architecture
- Sequential training
- Robust image augmentation and regularisation
- High fidelity overfitting prevention `(Early stopping, LR reduction at plateau, dropout)`
- Delivers state-of-the-art 99% accuracy across train and validation with no overfitting

**Fashion MNIST:**
- No pretrained model usage or transfer learning
- Data augmentation
- Custom CNN
- Delivers near state-of-the-art accuracy at 92.8% (max 94%)

**Facial Expressions:**
- Transfer Learning through EfficientNet Architecture (trainable)
- Robust Image Augmentation to replicate facial distortion
- Very protective overfitting and regularisation methods `(Early stopping, LR reduction at plateau, dropout)`
- Model saving at ideal weights and loading for evaluation
- Dealing with high class imbalance elegantly
- Delivers ***92% Top-3*** accuracy, 67% test accuracy and 74% train accuracy (best values, slight but controlled overfit)

**Imagenet100:**
- ResNet50 Transfer learner + dense layers achieve ballpark standard 86% accuracy
- Great balanced f1, precision, recall and MCC scores
- Top3 accuracy jumps to 99-96%, showing richness in feature detection
- misclassifications are usually between semantically similar classes (eg. types of sharks)


___

## Task 3:

Specialisation, Model and Reports on Natural Language Processing:

### Specialisation:
The Coursera specialisation certificates can be found in NLP/Certifications

### Model:
The BERT and RoBERTa Large Encoder models achieve near-state of the art performance. They are inside the NLP/Projects subdirectory.

RoBERTa:
| Metric                | Train Score       | Test Score        |
|-----------------------|-------------------|-------------------|
| Eval Loss             | 0.03287           | 0.11486           |
| Eval Precision        | 95.93%            | 91.97%            |
| Eval Recall           | 96.60%            | 92.96%            |
| Eval F1               | 96.26%            | 92.46%            |
| Eval Runtime (sec)    | 41.39             | 43.80             |
| Samples per Second    | 83.75             | 84.11             |
| Steps per Second      | 2.63              | 2.65              |
| Epoch                 | 5.0               | 5.0               |

BERT:
| Metric                | Train Score      | Test Score       |
|-----------------------|------------------|------------------|
| Eval Loss             | 0.03397          | 0.10361          |
| Eval Precision        | 95.559%          | 91.475%          |
| Eval Recall           | 96.191%          | 92.677%          |
| Eval F1               | 95.862%          | 92.046%          |
| Eval Runtime (sec)    | 40.69            | 43.14            |
| Samples per Second    | 85.18            | 85.39            |
| Steps per Second      | 2.68             | 2.69             |
| Epoch                 | 5.0              | 5.0              |

The projects subdirectory also consists of 
- tokenizers,
- word embedding generators, and
- a Wikipedia based RAG application.

### Reports:

The reports for Attention, BERT and RAGs are present as pdf files in the NLP/Reports subdirectory.




