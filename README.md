# Lab-3: AQI Classification using Deep Learning

A comprehensive deep learning project for **Air Quality Index (AQI) image classification** using PyTorch. This notebook implements and compares two approaches:
1. **Basic CNN** — A custom convolutional neural network trained from scratch
2. **Transfer Learning** — Fine-tuned EfficientNet-B0 pretrained on ImageNet

---

## Overview

This notebook covers the complete machine learning pipeline:

### Task 1-2: Data Preparation & Preprocessing
- Load AQI classification data from CSV (columns: `Filename`, `AQI_Class`)
- Validate image paths and remove broken entries
- Encode class labels using `LabelEncoder`
- Split dataset into **70% train, 15% validation, 15% test** (stratified)
- Implement custom `AQIDataset` class for PyTorch DataLoaders
- Apply image transformations:
  - **Training**: RandomHorizontalFlip, RandomRotation, ColorJitter, Normalization
  - **Validation/Test**: Resize and Normalization only
- Images resized to 224×224 pixels, normalized using ImageNet statistics

### Task 3: Model Building
- **Model 1 — BasicCNN**: Custom 3-block CNN with batch normalization and dropout
  - Blocks: 32 → 64 → 128 channels
  - Adaptive average pooling + fully connected classifier head
  - ~150K trainable parameters

- **Model 2 — EfficientNet-B0 Transfer Learning**:
  - Phase 1: Train classifier head only (frozen backbone) — 5 epochs
  - Phase 2: Fine-tune entire network with differential learning rates — 15 epochs
  - Backbone LR: 1e-5, Classifier LR: 1e-4
  - Cosine annealing scheduler for smooth convergence

### Task 4: Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrices (raw and normalized)
- Comparison charts between models

### Task 5: Training Curves
- Loss and accuracy plots over epochs
- Visualization of convergence behavior
- Transfer learning phase markers
---
