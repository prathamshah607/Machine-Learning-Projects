# DeepWeeds Classification with ResNet50

## Overview  
This project applies a ResNet50-based CNN to classify 9 weed species from the DeepWeeds dataset. The dataset contains real-world field imagery, and the model was trained sequentially across five pre-split subsets provided by the dataset creator. The objective was to build a robust model capable of generalizing across varied environmental conditions, class imbalance, and visual noise.

## Problem Statement  
The task is to perform multi-class classification of weed species from RGB images taken in the field. The challenge stems from high inter-class similarity, occlusion, varying lighting conditions, and imbalanced data. A reliable model should not only distinguish between species but also avoid misclassifying background vegetation as target classes.

## Dataset  
- **Source**: [DeepWeeds Dataset (Kaggle)](https://www.kaggle.com/datasets/imsparsh/deepweeds)
- **Classes**: 9  
  `Chinee Apple`, `Lantana`, `Parkinsonia`, `Parthenium`, `Prickly Acacia`, `Rubber Vine`, `Siam Weed`, `Snake Weed`, `Negative`
- **Structure**: Dataset split into 5 pre-separated CSV subsets (train/val per split). Model was trained on all these sequentially, and tested on test subset 0.
- **Image Dimensions**: Resized to 224x224 RGB
- **Preprocessing**:
  - Normalization with `rescale=1./255`
  - Rotation augmentation up to ±30°
  - One-hot encoding for classification
  - Per-split `class_weight` to address imbalance

## Model Pipeline & Workflow

1. **Data Loading**: Sequentially loaded all five train/val splits.
2. **Augmentation**: Applied real-time image augmentations during training.
3. **Architecture**: 
   - Pretrained `ResNet50` base (ImageNet weights, `include_top=False`)
   - Added `GlobalAveragePooling2D`, `Dropout(0.1)`, and a `Dense(9, softmax)` layer
4. **Training**: 
   - Used Adam optimizer and `categorical_crossentropy` loss
   - Trained each subset for 10 epochs with:
     - `EarlyStopping(patience=7)`
     - `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)`
5. **Evaluation**: Final scores computed on test subset 0 (after full training)

![image](https://github.com/user-attachments/assets/c6c29c98-f5b9-43f5-82f2-7eb8c1bc77e4)


## Results

On the Test subset 0:

| Metric                  | Value      |
|------------------------|------------|
| Accuracy          | 98.97%      |
| Weighted F1 Score          | 0.9895      |
| Weighted Precision         | 0.9898      |
| Weighted Recall            | 0.9879      |
| Log Loss                | 0.03139      |
| Matthews Corr. Coef.    | 0.9853      |
| ROC-AUC (avg)     | 1.00      |

| Class           | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| Chinee Apple   | 0.99      | 0.99   | 0.99     |
| Lantana        | 0.96      | 1.00   | 0.98     |
| Parkinsonia    | 1.00      | 1.00   | 1.00     |
| Parthenium     | 1.00      | 1.00   | 1.00     |
| Prickly Acacia | 1.00      | 1.00   | 1.00     |
| Rubber Vine    | 0.99      | 0.99   | 0.99     |
| Siam Weed      | 0.98      | 1.00   | 0.99     |
| Snake Weed     | 0.99      | 0.87   | 0.92     |
| Negative       | 0.99      | 1.00   | 1.00     |

![image](https://github.com/user-attachments/assets/4da6fbae-bf44-4f5c-a2fb-1b5ac7132c60)


---

## Misclassification Insights

- **Snake Weed** consistently showed the lowest recall (87%), likely due to visual overlap with surrounding vegetation and Siam Weed.
- **Negative** class maintained strong scores, proving the model distinguishes real weeds from background foliage well.
- All other classes hovered between 98-100% F1, suggesting the model generalizes very effectively despite natural image noise and imbalance.

![image](https://github.com/user-attachments/assets/570b0612-7920-4616-974b-f791874a61fe)


---

## Why ResNet50 with ImageNet
- Extremely rich feature detection through 50 convolutional layers
- Already good at discriminating against ImageNet plants species, it can adapt well to DeepWeeds if further tuned
- Very deep model which allows for extremely niche feature detection and classification
- Its use of skip connections and residual blocks focusses on value addition and does not fall into the vanishing/exploding gradient problem
- Its results go beyond traditional models to acheive SOTA accuracy

---

## Appendix  
- **Framework**: TensorFlow / Keras  
- **Runtime**: Kaggle (P100 GPU)  
- **Notebook**: [`actual.ipynb`](./actual.ipynb)  
- **Author**: Pratham Shah, 240905614, for Cryptonite Research Taskphase
