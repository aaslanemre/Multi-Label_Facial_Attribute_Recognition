# Multi-Label_Facial_Attribute_Recognition
## 🌟 Project Overview

This repository contains the implementation for a deep learning model designed to perform **Multi-Label Facial Attribute Recognition**. The goal is to accurately predict the presence or absence of multiple, non-mutually exclusive binary attributes (e.g., 'Smiling', 'Wearing Eyeglasses', 'Male') from a single input facial image.

Our approach utilizes **Transfer Learning** on a pre-trained Convolutional Neural Network (CNN) backbone and employs a carefully designed two-stage fine-tuning strategy to achieve high accuracy and robustness.

## 💻 Technical Stack

| Category | Component | Description |
| :--- | :--- | :--- |
| **Framework** | PyTorch / TensorFlow | *Specify one: e.g., PyTorch* |
| **Backbone** | ResNet-50 | Efficient and deep architecture for image feature extraction. |
| **Dataset** | CelebA (Celebrity Faces Attributes) | Over 200k images with 40 binary attribute annotations. |
| **Metrics** | F1-Score (per attribute), Mean Average Precision (mAP) | Key metrics for multi-label task evaluation. |


