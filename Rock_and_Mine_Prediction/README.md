🪨 Rock vs Mine Detection using Logistic Regression

This project uses Machine Learning (Logistic Regression) to classify sonar signals as either Rock (R) or Mine (M) based on reflected sonar signal data.

📘 Overview

The Sonar Dataset contains 208 samples of sonar readings collected from rocks and metal mines.
Each sample has 60 numeric features, representing the energy levels of sonar signals at different frequencies.
The final column (label) represents whether the object is a Rock (R) or a Mine (M).

This project builds and trains a Logistic Regression model to predict the object type based on these sonar readings.

📂 Dataset Information
Property	Description
Dataset Name	Sonar Dataset
Source	UCI Machine Learning Repository
File Used	Sonar data.csv
Rows	208
Columns	61
Features	Columns 0–59 (numeric sonar readings)
Target Label	Column 60 (R for Rock, M for Mine)
⚙️ Project Workflow
1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

2️⃣ Load and Explore the Data

Load dataset using pandas.read_csv()

Display first few rows, shape, and summary statistics

Count class distribution (R vs M)

Example:

Shape: (208, 61)
Classes: 111 Mines (M), 97 Rocks (R)

3️⃣ Data Preprocessing

Separate features and target variable

Split dataset into training and test sets (90% / 10%)

Stratify the split to maintain class balance

4️⃣ Model Training

Initialize and train Logistic Regression model on the training set.

model = LogisticRegression()
model.fit(X_train, y_train)

5️⃣ Model Evaluation

Predict on training and test data

Calculate accuracy using accuracy_score()

Dataset	Accuracy
Training Data	83.4%
Test Data	76.2%
6️⃣ Making Predictions

Example prediction on a single sonar reading:

input_data = (0.0286, 0.0453, 0.0277, 0.0174, ..., 0.0062)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_as_numpy_array)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")


Example Output:

['R']
The object is a Rock

🧠 Algorithm Details
Detail	Description
Algorithm Used	Logistic Regression
Model Type	Supervised Learning (Binary Classification)
Performance Metric	Accuracy
Libraries Used	scikit-learn, numpy, pandas, matplotlib
🧩 Requirements

Install dependencies using:

pip install numpy pandas matplotlib scikit-learn

🚀 How to Run the Project

Clone this repository or download the script.

Place Sonar data.csv in the same directory.

Open the project in Jupyter Notebook, VS Code, or any Python IDE.

Run the cells step-by-step to:

Load and explore the data

Train the model

Test accuracy

Make predictions

📊 Results Summary
Metric	Training Set	Test Set
Accuracy	83.4%	76.2%
📚 References

UCI Sonar Dataset

Scikit-learn Documentation

👨‍💻 Author

Your Name
Machine Learning Enthusiast
📧 your.email@example.com
