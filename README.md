# AI/ML Internship Tasks

This repository contains the first three completed AI/ML internship tasks.

## Repository Structure

```text
ai-ml-internship/
|-- task1_iris_exploration/       # Iris dataset EDA and visualization
|-- task2_stock_prediction/       # Stock price forecasting with yfinance
|-- task3_heart_disease/          # Heart disease binary classification
|-- requirements.txt
`-- README.md
```

## Setup

```bash
cd ai-ml-internship
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Task Summary

| Task | Objective | Dataset | Models / Tools | Key Results |
|------|-----------|---------|----------------|-------------|
| **1** | Explore and visualize tabular data | Iris via seaborn | pandas, matplotlib, seaborn | Feature distributions, species clusters, outlier patterns |
| **2** | Predict next-day closing price | AAPL via yfinance | Linear Regression, Random Forest | Actual vs predicted price plots, MAE/RMSE/R2 comparison |
| **3** | Predict heart disease risk | UCI Heart Disease Cleveland | Logistic Regression, Decision Tree | Accuracy, ROC-AUC, confusion matrix, feature importance |

## Task 1: Iris Dataset Exploration

**Objective:** Load, inspect, and visualize the Iris dataset to understand trends and distributions.

**Dataset:** Iris (`sns.load_dataset("iris")`)

**Models:** None. This is an exploratory data analysis task.

**Key Results:**
- 150 samples, 4 numeric features, and 3 species classes.
- Setosa is clearly separated by petal length and petal width.
- Versicolor and virginica overlap slightly, especially in sepal measurements.

**Notebook:** [`task1_iris_exploration/task1_iris_exploration.ipynb`](task1_iris_exploration/task1_iris_exploration.ipynb)

## Task 2: Stock Price Prediction

**Objective:** Use historical OHLCV data to predict the next day's closing price.

**Dataset:** Apple stock data (`AAPL`) loaded with `yfinance`.

**Models:** Linear Regression and Random Forest Regressor.

**Key Results:**
- Uses Open, High, Low, Volume, and lagged Close/Volume features.
- Uses a chronological train/test split to avoid future-data leakage.
- Compares actual and predicted closing prices with regression metrics.

**Notebook:** [`task2_stock_prediction/task2_stock_prediction.ipynb`](task2_stock_prediction/task2_stock_prediction.ipynb)

## Task 3: Heart Disease Prediction

**Objective:** Predict heart disease risk from patient health metrics.

**Dataset:** UCI Heart Disease Cleveland dataset.

**Models:** Logistic Regression and Decision Tree Classifier.

**Key Results:**
- Handles missing values with median imputation.
- Converts the original multi-class diagnosis label into a binary risk target.
- Evaluates models with accuracy, ROC-AUC, confusion matrices, ROC curves, and feature importance.

**Notebook:** [`task3_heart_disease/task3_heart_disease.ipynb`](task3_heart_disease/task3_heart_disease.ipynb)

## Submission Notes

- The notebooks include problem statements, dataset loading, preprocessing or EDA, visualizations, model training where applicable, evaluation, and final insights.
- Tasks 2 and 3 may require internet access to download fresh stock or dataset files when rerun.
