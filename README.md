# Alzheimer's Disease Detection Model with Vision Transformer (ViT) Head

## Overview

This project involves the development of a deep learning model for Alzheimer's disease detection using brain MRI scans. The model utilizes a MobileNetV2 backbone for feature extraction, which is enhanced with a Vision Transformer (ViT) head to classify MRI images into four categories: **Cognitively Normal (CN), Early Mild Cognitive Impairment (EMCI), Late Mild Cognitive Impairment (LMCI), and Alzheimer's Disease (AD)**.

The model is trained using a mixed dataset from **ADNI** (Alzheimer's Disease Neuroimaging Initiative) and **OASIS** datasets. The training process consists of two phases: 
1. **Phase 1:** Training the model with a frozen backbone (MobileNetV2).
2. **Phase 2:** Fine-tuning the entire model.

## Progression and challenges
To tackle this problem of AD Disease progression multi class classification we first did a thorough and deep research within this space coming across multiple papers we came across Vision transformers being used in this field,
We analyzed 2 papers published in 2025

1.HybridViT: An Approach for Alzheimer’s Disease Classification with ADNI Neuroimaging Data

2.Hybrid-RViT: Hybridizing ResNet-50 and Vision Transformer for Enhanced Alzheimer’s disease detection

and set them as our base and inspiration for this model, Both HybridViT and Hybrid-RViT showcase the power of hybrid CNN–ViT architectures for classifying Alzheimer’s disease stages using MRI, and they can be adapted to the user’s context of detecting AD progression (particularly within the ADNI dataset). In terms of ADNI utilization, HybridViT already directly uses ADNI data, demonstrating that a hybrid model can distinguish Normal, EMCI, LMCI, MCI, and AD with high accuracy​. Hybrid-RViT, while developed on OASIS, was conceptually aligned to ADNI’s categories and could be retrained on ADNI’s much larger sample to potentially achieve even better performance on detecting subtle progression.

We weren't getting significant results or were able to meet our target accuracies
- in the process we tried multiple other approaches and architectures including SWIN and DiET transformers and we also tried different data augmentation techniques and a mixture of ADNI+OASIS custom dataset as well, we got the best results on our MobileNetV2+ViT architecture achieving 95% acc on OASIS, we also got similarly good results on DIET transformer but we had our failures like it took time for us to get the datasets/data augmentation right , we failed with our SWIN tranformer approach and ResNet as backbone approach citing lack of enough images for training leading to poor perfomances.

## Dataset

### ADNI Dataset
The **ADNI-4C-Alzheimers-MRI-Classification** dataset from Kaggle is used for this project, containing MRI images categorized into four classes: **CN, EMCI, LMCI, and AD**.

### OASIS Dataset
The **OASIS Dataset** is also used, providing additional MRI images labeled with **Non-Demented, Very Mild Dementia, Mild Dementia, and Moderate Dementia** categories.

The datasets are preprocessed and augmented to ensure efficient training using TensorFlow's data pipeline (`tf.data`).

## Model Architecture

- **MobileNetV2 Backbone:**
  - Pretrained on ImageNet (without the top classification layer).
  - Extracts features from 160×160-sized images.
  - About 90% of MobileNetV2 layers are frozen during the initial phase of training.

- **Vision Transformer (ViT) Head:**
  - A custom ViT head is applied after the MobileNetV2 backbone.
  - The ViT head uses Transformer blocks, a class token, and positional embeddings to learn from the image patches.
  - The final output is a classification of the MRI images into one of four classes.

## Training Strategy

### Phase 1 (Head Training)
- **Freezing Backbone:** The MobileNetV2 backbone is frozen to prevent its weights from being updated during the first phase of training.
- **Training Head:** Only the ViT head (the projection, class token, and positional embeddings) is trained in this phase.
- **Early Stopping:** Training is monitored for validation loss, and early stopping is applied after a specified number of epochs if no improvement is observed.

### Phase 2 (Fine-tuning)
- **Unfreezing Backbone:** During phase 2, the MobileNetV2 backbone is unfrozen, and the entire model is fine-tuned with a lower learning rate.
- **Distillation Token:** The ViT head includes an optional distillation token, which allows the model to focus on both the class token and distillation token outputs for better performance.

### Early Stopping and Callbacks
- **Model Checkpoints**: Best-performing models are saved during training, based on validation loss.
- **Early Stopping**: The training process halts early if the model stops improving on the validation set.

## Requirements

- **TensorFlow**: Used for building, training, and evaluating the model.
- **Kaggle API**: For downloading datasets from Kaggle.
- **matplotlib**, **seaborn**: For visualizing training metrics, accuracy plots, and confusion matrices.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/alzheimers-detection.git
   cd alzheimers-detection
# DL_Project_AD_Progression_model
