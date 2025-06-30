# Fashion Classification with CNN on Fashion MNIST

## Overview
This project addresses image classification of fashion items using the Fashion MNIST dataset. I trained a simple convolutional neural network to categorize grayscale clothing images into 10 different fashion classes. The goal was to achieve high test accuracy with efficient architecture and precise interpretability.

## Problem Statement
Fashion MNIST is a well-known benchmark dataset that contains 28x28 grayscale images of fashion items. Each image must be classified into one of ten classes. The task is challenging due to low resolution, grayscale limitations, and class similarity (e.g., Shirt vs. T-shirt/top).

## Dataset
- **Source**: [Fashion MNIST - Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- **Classes** (10): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Resolution**: 28×28 grayscale
- **Preprocessing**:
  - Normalized to [0, 1] pixel values
  - Labels one-hot encoded for training
  - Simple augmentation using `ImageDataGenerator`:
    1. Rotation by a range of 10 degrees
    2. Zooming by a range of 10%
    3. Width/Height transposing by a range of 10%
    4. Slanting (shearing) by a range of 10%

## Model Pipeline & Workflow

### General Workflow
1. **Data Normalization** and basic augmentation
2. **Model Architecture**: Custom CNN with multiple Conv2D, MaxPooling, Dropout layers
3. **Training**: Categorical cross-entropy loss, Adam optimizer
4. **Evaluation**: Used accuracy, precision, recall, F1-score, MCC, and log loss

### Model Architecture

| Layer Type        | Details                                                      |
|------------------|--------------------------------------------------------------|
| Conv2D           | 32 filters, 4×4 kernel, He init, padding='same'              |
| BatchNormalization | Normalization layer                                        |
| ReLU             | Activation function                                          |
| MaxPooling2D     | 2×2 pool                                                     |
| Conv2D           | 64 filters, 3×3 kernel, padding='same'                       |
| BatchNormalization | —                                                          |
| ReLU             | —                                                            |
| MaxPooling2D     | 2×2 pool                                                     |
| Conv2D           | 128 filters, 3×3 kernel, padding='same'                      |
| BatchNormalization | —                                                          |
| ReLU             | —                                                            |
| Flatten          | Converts feature maps to 1D                                  |
| Dense            | 128 units                                                    |
| BatchNormalization | —                                                          |
| ReLU             | —                                                            |
| Dropout          | 35% dropout                                                  |
| Dense (Output)   | 10 units, softmax activation                                 |

The model was compiled with **Adam** optimizer and **categorical crossentropy** loss, using accuracy as the  evaluation metric.

![image](https://github.com/user-attachments/assets/5251dd3f-a7f4-43d2-a44d-07a69dc7c58a)


## Results

| Metric                | Score     |
|------------------------|-----------|
| Accuracy               | **92.80%** |
| Precision (weighted)   | **92.94%** |
| Recall (weighted)      | **92.80%** |
| F1 Score (weighted)    | **92.84%** |
| MCC                    | **0.92**   |
| Log Loss               | **0.20**   |


| Class         | Precision | Recall | F1-Score | 
|---------------|-----------|--------|----------|
| T-shirt/top   | 0.89      | 0.86   | 0.88     |
| Trouser       | 0.98      | 0.99   | 0.99     |
| Pullover      | 0.93      | 0.86   | 0.89     |
| Dress         | 0.94      | 0.91   | 0.93     |
| Coat          | 0.87      | 0.90   | 0.88     | 
| Sandal        | 0.99      | 0.99   | 0.99     | 
| Shirt         | 0.75      | 0.81   | 0.78     |
| Sneaker       | 0.96      | 0.98   | 0.97     |
| Bag           | 0.99      | 1.00   | 0.99     |
| Ankle boot    | 0.98      | 0.96   | 0.97     | 

**Macro and weighted average**: Precision 0.93, Recall 0.93, F1-score 0.93

Note: The rows are the actual labels, the columns the predicted labels.

![image](https://github.com/user-attachments/assets/ab80a057-8269-44b2-814a-1f184a02aa45)

![image](https://github.com/user-attachments/assets/a43d716a-099c-40d8-9536-71547a33b30f)

- The model confidently classifies **Sneakers**, **Bags**, and **Sandal** categories.
- Some confusion persists between visually similar classes like **Shirts** vs. **T-shirts**.
- Dropout and normalization helped avoid overfitting.

## Misclassification Insights
The misclassifications from the model are largely due to semanitc similarity between classes.
Most misclassifications are between visually similar items like t-shirts and shirts, coats and pullovers, etc.

![image](https://github.com/user-attachments/assets/289c75e3-7e1b-4976-9df3-16384bde5852)

An example of a misclassification:

![image](https://github.com/user-attachments/assets/16d9bed4-43b1-4f0a-a001-9a51ca36c26e)


## Conclusion
The model delivers strong performance on Fashion MNIST using a custom CNN trained from scratch. The accuracy and advanced metrics show that even simple architectures can perform reliably on clean, structured datasets like this one.

## Appendix
- **Trained in**: TensorFlow / Keras  
- **Hardware**: P100 GPU-backed Kaggle runtime  
- **Notebook**: [`FashionMNIST.ipynb`](./FashionMNIST.ipynb)  
- **Implemented by**: Pratham Shah, 240905614, for Cryptonite Research Taskphase
