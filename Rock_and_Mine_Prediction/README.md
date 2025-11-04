🪨 Rock vs Mine Detection using Logistic Regression

This project uses machine learning (Logistic Regression) to classify sonar signals as either Rock (R) or Mine (M) based on data collected from sonar signals reflected off various objects.

📘 Project Overview

The goal of this project is to build a binary classification model that can predict whether an object detected by sonar is a rock or a metal mine.
We use the Sonar dataset, which contains 208 samples with 60 numerical features representing energy levels at different frequencies.

📂 Dataset

Dataset Name: Sonar Dataset (from UCI Machine Learning Repository)

File: Sonar data.csv

Shape: 208 rows × 61 columns

Columns 0–59 → numerical features (sonar readings)

Column 60 → label (R for Rock, M for Mine)

⚙️ Project Workflow
1️⃣ Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

2️⃣ Loading and Exploring the Data

Load data using pandas.read_csv()

Check the first few rows with .head()

Get data summary with .describe()

Check label distribution and dataset shape

3️⃣ Data Preprocessing

Separate features (X) and labels (y)

Split into training and testing sets using train_test_split()

4️⃣ Model Training

Train a Logistic Regression model:

model = LogisticRegression()
model.fit(X_train, y_train)

5️⃣ Model Evaluation

Compute the accuracy on training and test data:

from sklearn.metrics import accuracy_score

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(train_pred, y_train))
print("Test Accuracy:", accuracy_score(test_pred, y_test))


Sample Results:

Training Accuracy → 83.4%

Test Accuracy → 76.2%

6️⃣ Making Predictions

Use the trained model to predict new data:

input_data = (0.0286, 0.0453, 0.0277, 0.0174, ..., 0.0062)
input_data_reshaped = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")


Example Output:

['R']
The object is a Rock

🧠 Algorithm Details

Algorithm Used: Logistic Regression

Type: Supervised Learning (Binary Classification)

Performance Metric: Accuracy

Libraries: scikit-learn, numpy, pandas, matplotlib

📊 Results Summary
Metric	Training Set	Test Set
Accuracy	83.4%	76.2%
🧩 Requirements

You can install the required Python libraries using:

pip install numpy pandas matplotlib scikit-learn

🚀 How to Run the Project

Clone this repository or copy the code into a Jupyter Notebook / Python file.

Place the dataset file Sonar data.csv in the same directory.

Run the script step-by-step or execute all cells in the notebook.

Enter new input values to make predictions.

📚 References
UCI Machine Learning Repository – Sonar Dataset
Scikit-learn Documentation

👨‍💻 Author
Aatir Ali
Machine Learning Enthusiast
