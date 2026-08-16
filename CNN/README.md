# CNN Image Classification — Rock, Paper, Scissors

## Student Information

- **Name:** MD Abir Hossain
- **Roll:** 220144
- **University:** Jashore University of Science and Technology (JUST)
- **Repository:** https://github.com/abir945/AI_ML_LAB

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** for **Rock, Paper, Scissors (RPS) image classification**.

The notebook is designed to run the complete workflow automatically:

1. Clone the GitHub repository.
2. Download the standard Rock-Paper-Scissors dataset automatically.
3. Preprocess the images.
4. Split the standard dataset into training, validation, and test sets.
5. Build and train a CNN model.
6. Plot training/validation loss and accuracy.
7. Save the trained model as `220144.pth`.
8. Load the saved model.
9. Load the 10 custom smartphone images from `CNN/dataset/`.
10. Predict the class and confidence for each custom image.
11. Display a confusion matrix and perform visual error analysis.

## Dataset

### Standard Dataset

The standard **Rock-Paper-Scissors** dataset is downloaded automatically during notebook execution.

The three classes are:

- `paper`
- `rock`
- `scissors`

No manual upload of the standard dataset is required.

### Custom Dataset

The custom images are stored in:

```text
CNN/dataset/
```

There are **10 custom images**:

- 4 Paper images
- 3 Rock images
- 3 Scissors images

The filenames follow this format:

```text
paper_01.jpg
paper_02.jpg
...
rock_01.jpg
...
scissors_01.jpg
...
```

The notebook automatically detects the images from the GitHub repository.

## CNN Architecture

The model is implemented in PyTorch and contains:

- 4 convolutional layers
- ReLU activation
- Max Pooling after each convolution block
- Fully connected layer
- Dropout (`0.30`)
- Final 3-class output layer

### Architecture Flow

```text
Input Image (128 × 128 × 3)
        ↓
Conv2D (3 → 32) + ReLU + MaxPool
        ↓
Conv2D (32 → 64) + ReLU + MaxPool
        ↓
Conv2D (64 → 128) + ReLU + MaxPool
        ↓
Conv2D (128 → 256) + ReLU + MaxPool
        ↓
Flatten
        ↓
Fully Connected (256)
        ↓
ReLU + Dropout
        ↓
Output (3 Classes)
```

### Training Configuration

- **Image Size:** `128 × 128`
- **Batch Size:** `64`
- **Epochs:** `15`
- **Optimizer:** Adam
- **Learning Rate:** `0.001`
- **Loss Function:** Cross Entropy Loss

## Preprocessing

Each image is processed using:

```text
Resize → ToTensor → Normalize
```

The normalization values are:

```python
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

The same preprocessing pipeline is used for the standard dataset and the custom images.

## Training and Evaluation

The notebook evaluates the model using:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Standard test-set accuracy
- Confusion matrix

Training graphs are generated for:

- Training vs Validation Loss
- Training vs Validation Accuracy

## Model File

After training, the model is saved as:

```text
CNN/model/220144.pth
```

The notebook can load this saved model for reproducible evaluation and prediction.

## Custom Image Prediction

The notebook automatically loads all 10 images from:

```text
CNN/dataset/
```

For each image, it displays:

- Actual class
- Predicted class
- Prediction confidence

Example:

```text
paper_01.jpg | Actual: paper | Predicted: rock | Confidence: 78.50%
```

## Real-World Testing

The 10 custom smartphone images are used as a separate real-world evaluation set.

In the current run, the custom images produced:

- **Correct predictions:** 4 / 10
- **Incorrect predictions:** 6 / 10
- **Custom-image accuracy:** 40%

This shows that performance on real-world/custom images can differ significantly from performance on the standard test set because of differences in background, lighting, angle, object position, and image appearance.

## Visual Error Analysis

The assignment asks for 3 incorrectly classified images from the **standard test set**.

The notebook checks the standard test set and selects up to 3 random misclassified images when such errors exist. Each displayed error includes:

- True label
- Predicted label

If the standard test set contains zero misclassified images, the notebook reports that no misclassified standard-test images were found rather than fabricating incorrect predictions.

The custom dataset has 6 incorrect predictions in the current run, and these can also be inspected through the custom-image prediction gallery.

## Repository Structure

```text
AI_ML_LAB/
└── CNN/
    ├── 220144.ipynb
    ├── dataset/
    │   ├── paper_01.jpg
    │   ├── paper_02.jpg
    │   ├── paper_03.jpg
    │   ├── paper_04.jpg
    │   ├── rock_01.jpg
    │   ├── rock_02.jpg
    │   ├── rock_03.jpg
    │   ├── scissors_01.jpg
    │   ├── scissors_02.jpg
    │   └── scissors_03.jpg
    └── model/
        └── 220144.pth
```

## How to Run

### 1. Open the Notebook

Open:

```text
CNN/220144.ipynb
```

in **Google Colab** or Jupyter Notebook.

### 2. Run All Cells

The notebook automatically:

- clones the repository,
- downloads the standard RPS dataset,
- trains the CNN,
- evaluates the model,
- saves the model,
- loads the custom images,
- generates predictions and confidence values.

No manual image upload is required.

## Requirements

Main Python libraries used:

```text
Python
PyTorch
Torchvision
NumPy
Matplotlib
Pillow
scikit-learn
```

## Conclusion

This project demonstrates an end-to-end CNN image classification workflow for **Rock, Paper, Scissors**, including dataset preparation, preprocessing, model training, evaluation, model saving/loading, confusion matrix analysis, and real-world custom-image prediction.

---

**Author:** MD Abir Hossain  
**Roll:** 220144  
**JUST**
