# CIFAR-10 Classifier: Achieving 94% Accuracy

This project demonstrates how to achieve high accuracy on the CIFAR-10 image classification benchmark using a deep learning approach inspired by AlexNet. The code is implemented in PyTorch and includes data augmentation, training scripts, and experiment tracking with MLflow.

## Overview
The goal of this project is to achieve at least 90% accuracy on the CIFAR-10 dataset using a convolutional neural network (CNN) inspired by AlexNet. With data augmentation and learning rate scheduling, the model achieves up to 94% accuracy.

## Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes
  - 50,000 training images
  - 10,000 test images
- [CIFAR-10 dataset info](https://www.cs.toronto.edu/~kriz/cifar.html)

## Model Architecture
The model is a deep CNN inspired by AlexNet, featuring:
- Multiple convolutional layers with increasing channels
- Batch normalization and ReLU activations
- Max pooling and dropout for regularization
- Fully connected layers for classification

See `model.py` for implementation details.

## Training Setup
- **Framework**: PyTorch
- **Optimizer**: SGD (lr=0.01, momentum=0.9, weight decay=5e-4)
- **Loss**: Cross-Entropy
- **Batch size**: 128
- **Epochs**: 100
- **Hardware**: NVIDIA GTX 3060 (12GB VRAM)
- **Experiment tracking**: MLflow
- **Model analysis**: FiftyOne

## Data Augmentation
To improve generalization, the following augmentations are applied:
- Color jitter (brightness, contrast, saturation, hue)
- Perspective transform
- Random horizontal flip

## Results
- **Vanilla training**: 88.3% accuracy after 100 epochs
- **With data augmentation**: 92.5% accuracy after 100 epochs
- **With learning rate scheduling**: 94% accuracy

See the `presentation/` folder for detailed plots and the LaTeX presentation.

## Usage
1. **Install dependencies** (see `pyproject.toml`):
   ```bash
   uv sync
   ```
2. **Train the model**:
   ```bash
   uv run train.py
   ```
3. **Track experiments**: MLflow is used for logging. Start the MLflow server if needed.
    ```bash
   mlflow server
   ```
4. **Visualize results**: Use `visualize.py` for FiftyOne model analysis.
    ```bash
    uv run visualize.py
    ```
