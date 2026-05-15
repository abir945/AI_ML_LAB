# Linear Regression - Software Developer Salary Prediction

## Overview

This project implements a **Linear Regression model** to predict software developer salary using a tabular dataset. The project includes the full machine learning workflow, including data loading, preprocessing, feature selection, normalization, model training, evaluation, and visualization.

The main goal of this project is to understand how Linear Regression works on real-world data and how well it can predict continuous values such as salary in USD.

---

## Dataset

- **Dataset file:** `dataset/train.csv`
- **Target column:** `salary_usd`
- **Problem type:** Regression
- **Features:** Developer-related attributes such as experience, country, education, programming languages, frameworks, and company size.

The model uses the available input features to predict the final software developer salary in USD.

---

## Workflow

The project follows these steps:

1. Load the dataset from `dataset/train.csv`.
2. Check and clean the dataset.
3. Remove missing values.
4. Select input features and target variable.
5. Encode categorical features using OneHotEncoder.
6. Normalize numerical features using StandardScaler.
7. Split the data into training and testing sets.
8. Train a Linear Regression model.
9. Predict salary values on the test data.
10. Evaluate the model using regression metrics.
11. Visualize the loss curve, actual vs predicted values, and sample predictions.

---

## Model Used

The machine learning model used in this project is:

```python
LinearRegression()
```

from the `scikit-learn` library.

---

## Evaluation Metrics

The model is evaluated using the following regression metrics:

- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **R2 Score** - Coefficient of Determination

---

## Results

Results from the notebook run:

| Metric | Value |
|---|---:|
| MAE | 1.324827e+04 |
| MSE | 2.611949e+08 |
| RMSE | 1.616153e+04 |
| R2 Score | 8.761469e-01 |

The evaluation metrics show how close the predicted salary values are to the actual salary values. A lower MAE, MSE, and RMSE indicate better prediction performance, while a higher R2 Score means the model explains more variation in the target variable.

---

## Visualizations

The notebook generates the following visual outputs:

- Loss curve
- Error metrics table
- Actual vs predicted salary graph
- Sample predictions table

---

## Screenshots

Add your screenshots inside the `screenshots/` folder and update the image paths below.

### Evaluation Metrics

![Evaluation Metrics]
https://github.com/abir945/AI_ML_LAB/blob/main/Linear_Regression/screenshots/Actual%20&%20Predicted%20Salary%20Graph.png?raw=true

### Loss Curve

![Loss Curve]
https://github.com/abir945/AI_ML_LAB/blob/main/Linear_Regression/screenshots/Loss%20Curve.png?raw=true

### Actual vs Predicted Salary

![Actual vs Predicted]
https://github.com/abir945/AI_ML_LAB/blob/main/Linear_Regression/screenshots/Actual%20&%20Predicted%20Salary%20Graph.png?raw=true

### Sample Predictions

![Sample Predictions]
https://github.com/abir945/AI_ML_LAB/blob/main/Linear_Regression/screenshots/Sample%20Predictions.png?raw=true

---

### File Description

- `linear_regression.ipynb` - Main Google Colab / Jupyter Notebook containing the full implementation.
- `dataset/train.csv` - Dataset used for training and testing the model.
- `screenshots/` - Folder containing output screenshots from the notebook.
- `README.md` - Project documentation file.

---

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Required libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## How to Run

1. Open the project folder.
2. Open `linear_regression.ipynb` in Jupyter Notebook or Google Colab.
3. Make sure the dataset is available at:

```text
dataset/train.csv
```

4. If using Google Colab, upload the dataset to Google Drive and load it using:

```python
df = pd.read_csv("/content/drive/MyDrive/ML/train.csv")
```

5. Run all notebook cells from top to bottom.
6. Check the evaluation results and generated visualizations.

---

## Sample Output

The model predicts software developer salary based on the selected input features. A sample prediction table is generated in the notebook showing:

- Actual salary
- Predicted salary
- Error difference

This helps compare how close the model predictions are to the real salary values.

---

## Notes

The prediction performance may vary depending on the dataset quality and feature relationships. Possible improvements include:

- Removing outliers from the dataset
- Applying better feature selection
- Adding more useful features
- Trying polynomial regression
- Using advanced models such as Decision Tree, Random Forest, or XGBoost

Even if the performance is not perfect, this project is useful for understanding the basic implementation of Linear Regression and regression evaluation metrics.

---

