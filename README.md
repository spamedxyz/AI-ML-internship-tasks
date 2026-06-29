# AI/ML Internship Tasks

This repository contains six AI/ML internship projects.

## Repository Structure

```text
ai-ml-internship/
|-- task1_iris_exploration/       # Iris dataset EDA and visualization
|-- task2_stock_prediction/       # Stock price forecasting with yfinance
|-- task3_heart_disease/          # Heart disease binary classification
|-- task4_health_chatbot/         # Prompt-engineered health Q&A chatbot
|-- task5_mental_health_chatbot/  # Empathetic mental health support chatbot
|-- task6_house_prices/           # House price regression
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

For Task 4 with OpenAI, set an API key before running the OpenAI backend:

```bash
set OPENAI_API_KEY=your_key_here
```

## Task Summary

| Task | Objective | Dataset | Models / Tools | Key Results |
|------|-----------|---------|----------------|-------------|
| **1** | Explore and visualize tabular data | Iris via seaborn | pandas, matplotlib, seaborn | Feature distributions, species clusters, outlier patterns |
| **2** | Predict next-day closing price | AAPL via yfinance | Linear Regression, Random Forest | Actual vs predicted price plots, MAE/RMSE/R2 comparison |
| **3** | Predict heart disease risk | UCI Heart Disease Cleveland format | Logistic Regression, Decision Tree | Accuracy, ROC-AUC, confusion matrix, feature importance |
| **4** | Build a health information chatbot | Prompt-only workflow | Optional OpenAI API, offline fallback | Safety filtering, health disclaimers, sample responses |
| **5** | Build empathetic support chatbot workflow | EmpatheticDialogues workflow | DistilGPT2 optional fine-tuning, Streamlit | Supportive responses, crisis-safety handling, offline demo |
| **6** | Predict house prices | California Housing or Kaggle CSV | Linear Regression, Gradient Boosting | MAE/RMSE/R2 and predicted vs actual plots |

## Task 1: Iris Dataset Exploration

**Objective:** Load, inspect, and visualize the Iris dataset to understand trends and distributions.

**Dataset:** Iris (`sns.load_dataset("iris")`)

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

**Dataset:** UCI Heart Disease Cleveland format, with an offline fallback dataset for reproducible execution.

**Models:** Logistic Regression and Decision Tree Classifier.

**Key Results:**
- Handles missing values with median imputation.
- Converts the original multi-class diagnosis label into a binary risk target.
- Evaluates models with accuracy, ROC-AUC, confusion matrices, ROC curves, and feature importance.

**Notebook:** [`task3_heart_disease/task3_heart_disease.ipynb`](task3_heart_disease/task3_heart_disease.ipynb)

## Task 4: General Health Query Chatbot

**Objective:** Answer general health questions using prompt engineering and safety filters.

**Tools:** Optional OpenAI chat API and a local offline fallback.

**Key Results:**
- Uses a clear medical-information system prompt.
- Blocks unsafe requests before model/API calls.
- Adds a medical disclaimer to every response.
- Runs offline for review without requiring an API key.

**Files:** [`task4_health_chatbot/health_chatbot.py`](task4_health_chatbot/health_chatbot.py), [`task4_health_chatbot/task4_health_chatbot.ipynb`](task4_health_chatbot/task4_health_chatbot.ipynb)

## Task 5: Mental Health Support Chatbot

**Objective:** Build a supportive chatbot workflow for stress, anxiety, and emotional wellness.

**Dataset:** EmpatheticDialogues workflow, with a small local notebook sample for offline execution.

**Models / Tools:** DistilGPT2 optional fine-tuning, Hugging Face Trainer, Streamlit fallback app.

**Key Results:**
- Formats dialogue examples into user-emotion, user-message, and supportive-response training text.
- Includes a real fine-tuning script for EmpatheticDialogues.
- Provides a Streamlit app that uses a fine-tuned model when available or a safe local fallback otherwise.
- Adds crisis-language handling.

**Files:** [`task5_mental_health_chatbot/finetune_empathetic.py`](task5_mental_health_chatbot/finetune_empathetic.py), [`task5_mental_health_chatbot/app.py`](task5_mental_health_chatbot/app.py), [`task5_mental_health_chatbot/task5_mental_health_chatbot.ipynb`](task5_mental_health_chatbot/task5_mental_health_chatbot.ipynb)

## Task 6: House Price Prediction

**Objective:** Predict house prices from property features.

**Dataset:** California Housing from scikit-learn by default, or Kaggle House Prices `train.csv` placed in `data/`.

**Models:** Linear Regression and Gradient Boosting Regressor.

**Key Results:**
- Performs preprocessing, feature scaling, model training, and regression evaluation.
- Reports MAE, RMSE, and R2.
- Visualizes predicted prices compared with actual prices and feature importance.

**Notebook:** [`task6_house_prices/task6_house_prices.ipynb`](task6_house_prices/task6_house_prices.ipynb)

## Submission Notes

- Notebooks include problem statements, dataset loading or workflow setup, preprocessing/EDA, modeling or chatbot logic, evaluation, and final insights.
- Tasks 4 and 5 include offline fallbacks so the notebooks and demos can be reviewed without API keys or model downloads.
- Task 5 fine-tuning can take significant time without a GPU, so the reusable training script is included alongside the notebook workflow.
