# Image Classification on ImageNet100 Using EfficientNetV2

## Overview
This project implements a EfficientNet-based image classification pipeline on a 100-class subset of the ImageNet dataset — known as **ImageNet100**. The goal was to build a scalable image classifier that can generalize well across diverse object categories while keeping computational cost competitive.

## Problem Statement
ImageNet100 is a balanced subset of 100 classes, making it ideal for rapid experimentation while still maintaining real-world difficulty. The goal is to correctly classify input images into one of the 100 object categories.

This task presents:
- High inter-class diversity (animals, tools, plants, etc.)
- Varying object scales, perspectives, and lighting
- Need for transfer learning due to dataset size and complexity

## Dataset
- **Source**: [ImageNet100 on Kaggle](https://www.kaggle.com/datasets/ambelkarimadhu/imagenet100)
- **Classes**: 100  
- **Images**: 130,000 train + 5,000 val  
- **Image Size**: Varied, resized during preprocessing to 256x256 RGB
- **Format**: JPEG, organized by class folders  
- **Preprocessing**:
- Combined all training folders into a unified `train_combined` directory.
- Applied the following augmentations using `ImageDataGenerator`:
  - Rotation: +/-30°
  - Zoom: +/-20%
  - Shear: +/-10%
  - Width & height shift: +/-10%
- Used `tf.keras.applications.efficientnet_v2.preprocess_input` for normalization
- Validation data was not shuffled to preserve index-label alignment

## Model Pipeline & Workflow

### General Workflow
1. **Data Loading**: Images loaded using `ImageDataGenerator.flow_from_directory()` with class mode `'categorical'`.
2. **Transfer Learning**: EfficientNetV2S (not pretrained, with random starting weights) was used as the backbone. The classification head is dropped.
3. **Model Structure**:
   - Base: `EfficientNetV2S(input_shape=(256, 256, 3), include_top=False)`
   - Custom top layers:
     - Batch Normalisation
     - GlobalAveragePooling2D
    
     - Dense(512, relu, l2-regularised)
     - Batch Normalisation
     - Dropout(0.3)
       
     - Dense(256, relu, l2-regularised)
     - Batch Normalisation
     - Dropout(0.3)
       
     - Dense(100, activation='softmax')
4. **Compilation**:
   - Optimizer: Adam
   - Loss: Categorical crossentropy
   - Metrics: Accuracy, Top3 Accuracy
5. **Callbacks**:
   - EarlyStopping (patience=5)
   - ModelCheckpoint (best weights)
   - ReduceLROnPlateau (factor=0.5, patience=3)
    (All these monitor val accuracy)
6. **Training**:
   - Batch size: 32
   - Epochs: 12
   - Trained on Kaggle GPU environment

### Why EfficientNetV2S?
- It handles feature scaling, depth scaling and resolution in a predetermined way
- It is suitable for a 100 image classification subset of ImageNet
- It is a lightweight upgraded version of the tried and tested EfficientNet, using upgraded Fused-MBConv layers and 40 convolutional layers rather than the original's 16

## Evaluation & Results

### Example Metrics for different Semantic Classes

| Class Name             | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| great white shark      | 0.85      | 0.92   | 0.88     |
| hammerhead             | 0.92      | 0.90   | 0.91     |
| chickadee              | 0.94      | 1.00   | 0.97     |
| bald eagle             | 0.94      | 0.98   | 0.96     |
| great grey owl         | 0.96      | 0.98   | 0.97     |
| common newt            | 0.78      | 0.80   | 0.79     |
| spotted salamander     | 0.89      | 0.94   | 0.91     |
| mud turtle             | 0.75      | 0.48   | 0.59     |
| green lizard           | 0.73      | 0.88   | 0.80     |
| Komodo dragon          | 0.94      | 0.94   | 0.94     |

### Final Metrics (Validation Set)

| Metric                | Value     |
|------------------------|-----------|
| **Top-1 Accuracy**     | **86.82%** |
| **Top-3 Accuracy**     | **97.5%** |
| **F1 Score (weighted)**   | 0.867       |
| **Precision (weighted)**  | 0.87       |
| **Recall (weighted)**     | 0.8682       |
| **Log Loss**           | 0.6017       |
| **ROC AUC (avg)**| 0.99       |
| **Matthews Corr. Coef.** | 0.867    |

These metrics reflect strong generalization and predictive power.

**Most accurate categories**:
- birds like chickadees, great grey owls 

**Most confused categories**:
- mud turtles, leatherback turtles, water reptiles

![image](https://github.com/user-attachments/assets/5ebf5e2a-e32c-4ba4-acdc-d0251e9d37e4)

![image](https://github.com/user-attachments/assets/29916d31-9492-4f15-9a09-53e4c543211f)

### Misclassification Analysis

Misclassifications, as seen from this report, occur largely between semantically similar classes (eg. same coloured fish). This is also shows that the model is detecting rich features, and the correct classification is almost always among the top 3 predictions.

![image](https://github.com/user-attachments/assets/7037529f-5798-46c2-8310-748f596a9b73)

Sample misclassification:

![image](https://github.com/user-attachments/assets/3cd3768f-c992-4fd5-b5de-857945176606)

## Conclusion

- The EfficientNet-based model performs reliably across 100 diverse classes.
- Effective data augmentation and transfer learning helped achieve **87% top-1 and 96% top-3 accuracy**.
- Results suggest the model generalizes well, with confidence and separability.

## Appendix
- **Notebook**: [`model_training.ipynb`](./model_training.ipynb), [`model_analysis.ipynb`](./model_analysis.ipynb)
- **Environment**: Kaggle Notebook (P100 GPU, TensorFlow/Keras)
- **Author**: Pratham Shah, 240905614
