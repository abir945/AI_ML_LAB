# Logistic Regression (Binary Classification)

## Overview

This project builds a **Logistic Regression** classifier for a binary classification problem.  
It includes data loading, preprocessing, model training, evaluation metrics, and standard classification visualizations.

The main objective of this project is to predict a binary outcome using numeric input features and evaluate the model performance using common classification metrics.

---

## Dataset

- **Source file:** `dataset/data.csv`
- **Target:** Binary class (`0/1`)
- **Features:** Numeric predictors used for classification

The dataset is loaded from the `dataset/` folder and used to train and test the logistic regression model.

---

## Workflow

1. Load data from `dataset/data.csv`
2. Perform basic preprocessing
3. Split the dataset into training and testing sets
4. Train a Logistic Regression model
5. Generate predictions on the test data
6. Evaluate the model using classification metrics
7. Visualize the confusion matrix and ROC curve
8. Display sample predictions with probability scores

---

## Results

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.8784107228338918 |
| Precision | 0.8820116054158608 |
| Recall | 0.8735632183908046 |
| F1 Score | 0.87776708373436 |
| AUC | 0.9611541916442099 |

The model achieved strong performance on the current dataset, especially with a very high AUC score.

---

## Visuals

The notebook generates and saves the following outputs:

- Confusion Matrix
- Evaluation Metrics
- ROC Curve
- Sample Predictions Table

---

## Screenshots

### Confusion Matrix

![Confusion Matrix]
https://github.com/abir945/AI_ML_LAB/blob/main/Logistic_Regression/screenshots/Confusion%20Matrix.png?raw=true

### Evaluation Metrics

![Evaluation Metrics]
https://github.com/abir945/AI_ML_LAB/blob/main/Logistic_Regression/screenshots/Evaluation%20Metrics.png?raw=true

### ROC Curve

![ROC Curve]
https://github.com/abir945/AI_ML_LAB/blob/main/Logistic_Regression/screenshots/ROC%20Curve.png?raw=true

### Sample Predictions

![Sample Predictions]
https://github.com/abir945/AI_ML_LAB/blob/main/Logistic_Regression/screenshots/Sample%20Predictions.png?raw=true

---

## How to Run

1. Open `210137_logistic.ipynb` in Jupyter Notebook or Google Colab.
2. Make sure the dataset is available at:

```text
dataset/data.csv
```

3. Install or import the required libraries:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

4. Run all cells from top to bottom.
5. Check the generated metrics, confusion matrix, ROC curve, and sample prediction outputs.

---

## Requirements

The following Python libraries are used:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Model Used

- **Algorithm:** Logistic Regression
- **Problem Type:** Binary Classification
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, AUC

---

## Notes

The evaluation metrics indicate strong class separation on the current dataset.  
However, the model should also be validated using a holdout test set or cross-validation to confirm that it generalizes well to unseen data.

Possible improvements:

- Apply cross-validation
- Tune hyperparameters
- Check class imbalance
- Try feature scaling
- Compare with other classification models

---

