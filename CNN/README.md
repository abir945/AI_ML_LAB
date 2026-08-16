# CNN Image Classification with PyTorch — FashionMNIST

## Assignment 8 — FashionMNIST with Custom Smartphone Images

A complete Convolutional Neural Network (CNN) image-classification project developed with **PyTorch**. The model is trained on the **FashionMNIST** dataset and then used to classify **10 custom smartphone images** representing the FashionMNIST classes.

**Student:** Md Abir Hossain  
**Student ID:** 220144  
**Course:** AI/ML Lab  
**Repository:** [AI_ML_LAB](https://github.com/abir945/AI_ML_LAB)  
**CNN Project Folder:** [CNN](https://github.com/abir945/AI_ML_LAB/tree/main/CNN)  
**Custom Dataset:** [CNN/dataset](https://github.com/abir945/AI_ML_LAB/tree/main/CNN/dataset)

---

## 📌 Project Overview

This assignment implements an end-to-end CNN image classification pipeline using PyTorch.

The notebook automatically:

1. Clones the public GitHub repository in Google Colab.
2. Loads the 10 custom smartphone images from `CNN/dataset/`.
3. Downloads the standard FashionMNIST dataset using `torchvision.datasets`.
4. Applies preprocessing and data augmentation.
5. Splits the FashionMNIST training data into training and validation sets.
6. Builds and trains a custom three-block CNN.
7. Tracks training and validation loss and accuracy.
8. Saves the trained model as `CNN/model/220144.pth`.
9. Generates loss/accuracy curves, confusion matrix, classification report, and error analysis.
10. Preprocesses the real-world smartphone images.
11. Predicts the class of each custom image.
12. Calculates prediction probabilities using Softmax.
13. Displays the custom-image predictions in a gallery.

---

## 🔗 Important Repository Links

| Resource | Link |
|---|---|
| Main Repository | https://github.com/abir945/AI_ML_LAB |
| CNN Folder | https://github.com/abir945/AI_ML_LAB/tree/main/CNN |
| Custom Images | https://github.com/abir945/AI_ML_LAB/tree/main/CNN/dataset |

---

## 📂 Project Structure

```text
AI_ML_LAB/
│
├── CNN/
│   ├── dataset/
│   │   ├── 01_t-shirt.png
│   │   ├── 02_trouser.jpg
│   │   ├── 03_pullover.jpg
│   │   ├── 04_dress.jpg
│   │   ├── 05_coat.jpg
│   │   ├── 06_sandal.jpg
│   │   ├── 07_shirt.jpg
│   │   ├── 08_sneaker.jpg
│   │   ├── 09_bag.jpg
│   │   └── 10_ankle_boot.jpg
│   │
│   ├── model/
│   │   └── 220144.pth
│   │
│   └── assets/
│       └── 220144_history.json
│
└── 220144_CNN.ipynb
```

The notebook also creates `CNN/data/` for the downloaded FashionMNIST data and `CNN/assets/` for generated training history and visual assets.

---

# 🗂 FashionMNIST Dataset

FashionMNIST contains 10 clothing and footwear categories. The class mapping used in the notebook is:

| Label | Class Name |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

# 📱 Custom Smartphone Dataset

The project uses exactly **10 custom smartphone images**, one for each FashionMNIST class.

| No. | Image | Expected Class |
|---:|---|---|
| 01 | `01_t-shirt.png` | T-shirt/top |
| 02 | `02_trouser.jpg` | Trouser |
| 03 | `03_pullover.jpg` | Pullover |
| 04 | `04_dress.jpg` | Dress |
| 05 | `05_coat.jpg` | Coat |
| 06 | `06_sandal.jpg` | Sandal |
| 07 | `07_shirt.jpg` | Shirt |
| 08 | `08_sneaker.jpg` | Sneaker |
| 09 | `09_bag.jpg` | Bag |
| 10 | `10_ankle_boot.jpg` | Ankle boot |

All custom images are available here:

**[Open Custom Dataset on GitHub](https://github.com/abir945/AI_ML_LAB/tree/main/CNN/dataset)**

---

# 🧠 CNN Architecture

The model contains three convolutional blocks followed by a fully connected classifier.

```text
Input: 1 × 28 × 28
        │
        ▼
┌──────────────────────────────┐
│ Block 1                      │
│ Conv2D: 1  → 32             │
│ BatchNorm2D                 │
│ ReLU                         │
│ Conv2D: 32 → 32             │
│ BatchNorm2D                 │
│ ReLU                         │
│ MaxPool 2×2                 │
│ Dropout 0.25               │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Block 2                      │
│ Conv2D: 32 → 64             │
│ BatchNorm2D                 │
│ ReLU                         │
│ Conv2D: 64 → 64             │
│ BatchNorm2D                 │
│ ReLU                         │
│ MaxPool 2×2                 │
│ Dropout 0.25               │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Block 3                      │
│ Conv2D: 64 → 128            │
│ BatchNorm2D                 │
│ ReLU                         │
│ MaxPool 2×2                 │
│ Dropout 0.25               │
└──────────────────────────────┘
        │
        ▼
Flatten
        │
        ▼
Linear: 1152 → 256
        │
      ReLU
        │
  Dropout 0.50
        │
        ▼
Linear: 256 → 10
        │
        ▼
10-Class Output
```

### Model Components

- `Conv2d` for feature extraction
- `BatchNorm2d` for stable training
- `ReLU` activation
- `MaxPool2d` for spatial downsampling
- `Dropout` for regularization
- `Flatten` before classification
- `Linear` layers for final classification

---

# 🔄 Image Preprocessing

## FashionMNIST Images

The evaluation transform used for standard images is:

```text
Grayscale
   ↓
Resize to 28 × 28
   ↓
Convert to Tensor
   ↓
Normalize using FashionMNIST statistics
```

The notebook uses:

```text
Mean = 0.2860
Std  = 0.3530
```

## Training Augmentation

Training images additionally use:

- Random Horizontal Flip
- Random Affine Transformation
- Small Rotation/Translation/Scale changes
- Tensor conversion
- Normalization

## Custom Smartphone Images

Real-world images need additional preparation because they are different from FashionMNIST images in resolution, background, lighting, and composition.

```text
EXIF Rotation Correction
        ↓
Grayscale Conversion
        ↓
Background Adjustment
        ↓
Object Cropping
        ↓
Contrast Enhancement
        ↓
Resize to 28 × 28
        ↓
Tensor Conversion
        ↓
Normalization
        ↓
CNN Prediction
```

---

# 📊 Dataset Split

The original FashionMNIST training set contains **60,000 images**. In this project, it is split into:

| Split | Images |
|---|---:|
| Training | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |
| **Total** | **70,000** |

The official FashionMNIST test set contains 10,000 images.

---

# ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.001 initially |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Validation Fraction | 10% |
| Random Seed | 42 |
| Device in recorded run | CUDA / NVIDIA T4 |

The learning-rate scheduler reduced the learning rate during training.

---

# 📈 Training Results

The recorded training run achieved the following validation performance:

| Epoch | Train Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 74.44% | 85.33% |
| 2 | 82.20% | 87.02% |
| 3 | 84.33% | 87.55% |
| 4 | 85.39% | 89.00% |
| 5 | 86.34% | 89.75% |
| 6 | 86.98% | 90.50% |
| 7 | 88.39% | 90.85% |
| 8 | 88.97% | 91.52% |
| 9 | 88.90% | 91.50% |
| 10 | 89.01% | 91.70% |

### Final Test Result

```text
Test Accuracy : 91.75%
```

---

# 📋 Custom Image Results

The final notebook report recorded:

```text
Custom images      : 10
Known-label correct: 8/10
Custom accuracy    : 80.00%
```

So the model achieved **80.00% accuracy on the custom images whose known labels were available for comparison**.

> Note: performance on smartphone images can be lower than FashionMNIST test accuracy because real-world photos contain different backgrounds, lighting, scale, orientation, and object composition.

---

# 📊 Evaluation

The notebook produces the following evaluation outputs:

### 1. Loss vs Epochs

Training and validation loss are plotted across the 10 training epochs.

### 2. Accuracy vs Epochs

Training and validation accuracy are plotted across the 10 training epochs.

### 3. Confusion Matrix

The confusion matrix shows how predictions are distributed across the 10 FashionMNIST classes.

### 4. Classification Report

The notebook generates a classification report containing:

- Precision
- Recall
- F1-score
- Support

### 5. Error Analysis

Three randomly selected misclassified FashionMNIST test images are displayed for visual error analysis.

### 6. Custom Prediction Gallery

The 10 custom smartphone images are displayed in a **2 × 5 gallery** with their predicted class and confidence.

---

# 💾 Saved Model

The trained model is saved as:

```text
CNN/model/220144.pth
```

The training history is saved as:

```text
CNN/assets/220144_history.json
```

---

# 🚀 How to Run in Google Colab

## Step 1 — Open the Notebook

Open the CNN notebook from the repository:

**[Open CNN Folder](https://github.com/abir945/AI_ML_LAB/tree/main/CNN)**

## Step 2 — Run All Cells

The notebook is designed to automatically clone:

```text
https://github.com/abir945/AI_ML_LAB.git
```

## Step 3 — Automatic Dataset Loading

The notebook finds the custom images from:

```text
AI_ML_LAB/CNN/dataset/
```

It also downloads FashionMNIST automatically using `torchvision.datasets`.

## Step 4 — Train the Model

The CNN is trained for 10 epochs.

## Step 5 — Evaluate

After training, the notebook generates the requested evaluation metrics and visualizations.

## Step 6 — Predict Custom Images

The 10 smartphone images are automatically preprocessed and classified.

---

# 📦 Required Libraries

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

Main libraries used:

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

---

# 🎯 Learning Objectives

This project demonstrates practical understanding of:

- Convolutional Neural Networks
- Image classification
- PyTorch model building
- Data preprocessing
- Data augmentation
- Train/validation/test splitting
- Model training
- Model evaluation
- Confusion matrix analysis
- Classification reports
- Error analysis
- Real-world image preprocessing
- Softmax confidence prediction
- Saving and loading model weights

---

# 👨‍💻 Author

**Md Abir Hossain**  
**Student ID:** 220144  
**Course:** AI/ML Lab  
**University:** Jashore University of Science and Technology (JUST)

---

# 📌 Repository

**Main Repository:**  
https://github.com/abir945/AI_ML_LAB

**CNN Project:**  
https://github.com/abir945/AI_ML_LAB/tree/main/CNN

**Custom Image Dataset:**  
https://github.com/abir945/AI_ML_LAB/tree/main/CNN/dataset

---

## 📄 Academic Use

This project was prepared as an academic AI/ML Lab assignment.
