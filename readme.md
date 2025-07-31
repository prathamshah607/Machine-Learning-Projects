# Pratham Shah's Taskphase for the Cryptonite Research AI Subsystem

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
The BERT Encoder model achieves near-state of the art performance. It is inside the NLP/Projects subdirectory.

The projects subdirectory also consists of 
- tokenizers,
- word embedding generators, and
- a Wikipedia based RAG application.

### Reports:

The reports for Attention, BERT and RAGs are present as pdf files in the NLP/Reports subdirectory.




