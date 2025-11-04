# 🪨 Rock vs Mine Detection using Logistic Regression

This project applies **Machine Learning (Logistic Regression)** to classify sonar signals as either **Rock (R)** or **Mine (M)** based on the pattern of reflected sonar waves.

---

## 📘 Overview

The **Sonar Dataset** contains 208 samples of sonar signal data collected by bouncing sonar signals off various surfaces — both rocks and metal cylinders (mines).  
Each sample consists of **60 numeric features** that represent the energy levels of sonar signals at different frequencies, and the final label (`R` or `M`) denotes whether the object is a **Rock** or a **Mine**.

This project builds and evaluates a **Logistic Regression model** to predict the type of object based on these features.

---

## 📂 Dataset Information

| Property | Description |
|-----------|--------------|
| **Dataset Name** | Sonar Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) |
| **File Used** | `Sonar data.csv` |
| **Rows** | 208 |
| **Columns** | 61 |
| **Columns 0–59** | Sonar signal energy readings |
| **Column 60** | Target label (`R` = Rock, `M` = Mine) |

---

## ⚙️ Project Workflow

### 1️⃣ Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

### 2️⃣ Load and Explore the Data
df = pd.read_csv('/content/Sonar data.csv', header=None)

print(df.head())
print(df.describe())
print(df.shape)
print(df[60].value_counts())

Output Example:
Shape: (208, 61)
Classes: 111 Mines (M), 97 Rocks (R)


### 3️⃣ Data Analysis

Statistical summary of all 60 numeric features was obtained using df.describe().
Data distribution by target class (R vs M) was verified using value_counts().
Average feature values were also compared group-wise using:

df.groupby(60).mean()

## 4️⃣ Data Preprocessing
# Separating features and labels
X = df.drop(columns=60, axis=1)
y = df[60]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1
)

Output Example:

X Shape: (208, 60)
Train Shape: (187, 60)
Test Shape: (21, 60)

## 5️⃣ Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

The Logistic Regression model is trained on the sonar signal data to distinguish between Rocks and Mines.

## 6️⃣ Model Evaluation
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print("Accuracy on training data:", training_data_accuracy)
print("Accuracy on test data:", test_data_accuracy)

Results:
Dataset	Accuracy
Training Data	83.42%
Test Data	76.19%


## 7️⃣ Making Predictions
# Example input data
input_data = (0.0286, 0.0453, 0.0277, 0.0174, 0.0384, 0.0990, 0.1201, 0.1833,
              0.2105, 0.3039, 0.2988, 0.4250, 0.6343, 0.8198, 1.0000, 0.9988,
              0.9508, 0.9025, 0.7234, 0.5122, 0.2074, 0.3985, 0.5890, 0.2872,
              0.2043, 0.5782, 0.5389, 0.3750, 0.3411, 0.5067, 0.5580, 0.4778,
              0.3299, 0.2198, 0.1407, 0.2856, 0.3807, 0.4158, 0.4054, 0.3296,
              0.2707, 0.2650, 0.0723, 0.1238, 0.1192, 0.1089, 0.0623, 0.0494,
              0.0264, 0.0081, 0.0104, 0.0045, 0.0014, 0.0038, 0.0013, 0.0089,
              0.0057, 0.0027, 0.0051, 0.0062)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape for single instance prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")

Example Output:
['R'] --> The object is a Rock

## 🧠 Algorithm Details
Attribute	Description
Algorithm Used	Logistic Regression
Type	Supervised Learning (Binary Classification)
Target Classes	Rock (R), Mine (M)
Performance Metric	Accuracy
Libraries Used	scikit-learn, numpy, pandas, matplotlib
📊 Results Summary
Metric	Training Set	Test Set
Accuracy	83.4%	76.2%

The model generalizes reasonably well, showing moderate variance and consistent predictive power.

🧩 Requirements
To install dependencies:
pip install numpy pandas matplotlib scikit-learn

🚀 How to Run the Project
Clone this repository or download the script.
Place the file Sonar data.csv in the same directory as your script.
Open the project in Jupyter Notebook, Google Colab, or any Python IDE.
Run all cells sequentially to:
Load and explore the data
Train the Logistic Regression model
Evaluate accuracy
Make predictions

#### 📚 References
UCI Sonar Dataset
Scikit-learn Documentation

👨‍💻 Author
Aatir Ali
Machine Learning Enthusiast
