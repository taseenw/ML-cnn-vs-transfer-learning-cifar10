# CNN vs Transfer Learning on CIFAR-10

## Overview
This project compares the performance of a custom Convolutional Neural Network (CNN) with two popular transfer learning architectures — **ResNet50** and **VGG16** — on the CIFAR-10 image classification task.  
The goal is to **explore the trade-offs between building models from scratch vs leveraging pre-trained networks**, and to analyze how transfer learning impacts accuracy, convergence speed, and generalization.

This work was done as part of my deep learning practice and is intended to demonstrate both **hands-on implementation skills** in PyTorch and **analytical thinking** about model performance.

---

## Project Objectives
- Implement and train a custom CNN for CIFAR-10 classification.
- Fine-tune ResNet50 and VGG16 using transfer learning.
- Compare:
  - Training speed and convergence behavior.
  - Test accuracy and generalization.
  - Overfitting tendencies.
- Visualize and interpret loss/accuracy trends.

---

## Dataset
- **CIFAR-10**: 60,000 color images (32×32 pixels) in 10 classes.
- Standard train/test split (50,000 / 10,000).

---

## Models
1. **SimpleCNN**  
   - 3 convolutional layers + pooling  
   - Trained from scratch for 5 epochs.

2. **ResNet50 Transfer Learning**  
   - Pre-trained on ImageNet, frozen convolutional base, custom classifier head.  
   - Fine-tuned for 5 epochs.

3. **VGG16 Transfer Learning**  
   - Pre-trained on ImageNet, frozen convolutional base, custom classifier head.  
   - Fine-tuned for 5 epochs.

---

## Key Results
| Model      | Final Test Accuracy | Notes |
|------------|--------------------|-------|
| SimpleCNN  | ~60%               | Good generalization, no overfitting, slower learning curve. |
| ResNet50   | ~81%               | Large jump in performance, strong generalization. |
| VGG16      | ~86.5% (peak)      | Highest accuracy but clear signs of overfitting after early epochs. |

---

## Insights
- **Transfer learning drastically improves initial performance** — both ResNet50 and VGG16 start with >77% accuracy on CIFAR-10 after the first epoch.
- **VGG16 outperforms ResNet50 in accuracy**, but overfits faster — requiring early stopping or regularization.
- **SimpleCNN generalizes well** but requires more training and a deeper architecture to compete with pre-trained models.
- Visualization of training/test curves was critical for identifying overfitting.

---

## Tech Stack
- **Language:** Python 3
- **Framework:** PyTorch
- **Libraries:** torchvision, matplotlib, tqdm, numpy

---

## How to Run
```bash
# Clone the repository
git clone https://github.com/<your-username>/cnn-vs-transfer-learning-cifar10.git
cd cnn-vs-transfer-learning-cifar10

# Install dependencies
pip install -r requirements.txt

# Open the Jupyter Notebook
jupyter notebook cnn-vs-transfer-learning-cifar10.ipynb
