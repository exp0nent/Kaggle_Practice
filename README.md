# 🐚 Abalone Age Prediction (Regression Model)

This project predicts the **age of abalone** based on its physical measurements using **machine learning regression models**.  
The dataset used is the classic **Abalone Dataset** from the **UCI Machine Learning Repository**.

---

## 📘 Overview

The main goal of this project is to estimate the **number of rings** (a proxy for age) of abalone shells using measurable physical attributes such as length, diameter, weight, and height.  
Abalone age is approximately calculated as:

> **Age = Rings + 1.5**

---

## 📂 Dataset Description

Each row in the dataset represents one abalone sample.

| Column | Description | Type |
|---------|--------------|------|
| Sex | M (Male), F (Female), I (Infant) | Categorical |
| Length | Longest shell measurement (mm) | Numeric |
| Diameter | Perpendicular to length (mm) | Numeric |
| Height | Height with meat in shell (mm) | Numeric |
| Whole weight | Whole abalone weight (grams) | Numeric |
| Shucked weight | Weight of meat (grams) | Numeric |
| Viscera weight | Gut weight (grams) | Numeric |
| Shell weight | Weight of shell after being dried (grams) | Numeric |
| Rings | Number of rings (proxy for age) | Integer |

**Rows:** 4,177  
**Missing values:** None

---

## 🧹 Data Preprocessing

1. **Encoding:**  
   The categorical `Sex` column was encoded as:
   - `M → 0`, `F → 1`, `I → 2`

2. **Scaling:**  
   Numerical features were standardized using `StandardScaler` to normalize all feature ranges.

3. **Train-Test Split:**  
   Data was split into:
   - **Training set:** 80%  
   - **Testing set:** 20%

---

## 🔎 Exploratory Data Analysis (EDA)

Several visualizations were performed using **Seaborn** and **Matplotlib**:
- Distribution of abalone sex and rings.
- Correlation heatmap of all features.
- Scatter plots of `Length` vs `Rings` (with and without color coding by sex).
- Insights:
  - Physical dimensions and weights are strongly correlated.
  - Most abalones have between 6–12 rings (ages ~7.5–13.5 years).
  - Males and females have similar size distributions, while infants are smaller.

---

## 🧠 Model Training and Evaluation

Eight regression algorithms were trained and compared:

| Model | MSE | R² |
|--------|------|------|
| Linear Regression | 4.95 | 0.54 |
| Ridge Regression | 4.95 | 0.54 |
| Lasso Regression | 7.69 | 0.29 |
| Decision Tree Regressor | 8.94 | 0.17 |
| Random Forest Regressor | 4.94 | 0.54 |
| Support Vector Regressor (SVR) | **4.88** | **0.55** |
| Gradient Boosting Regressor | 5.10 | 0.53 |
| K-Neighbors Regressor | 5.24 | 0.52 |

### ✅ Best Performing Models
- **Support Vector Regressor (SVR)** and **Random Forest Regressor**
  - Achieved the best balance between bias and variance.
  - R² ≈ **0.55**, MSE ≈ **4.9**

### ❌ Least Performing Model
- **Decision Tree Regressor**
  - Overfitted training data and generalized poorly (R² ≈ 0.17).

---

## 📊 Model Interpretation

- The most influential features for predicting abalone age:
  - **Shell weight**
  - **Whole weight**
  - **Length**
- Abalones with heavier shells and larger sizes tend to have more rings (older).

---

## 🧩 Prediction Function

A custom function `prediction_rings()` allows real-time prediction of abalone ring count based on user inputs.

**Example input:**
- Sex = 0 (Male)
- Length = 0.6
- Diameter = 0.45
- Height = 0.15
- Whole weight = 1.0
- Shucked weight = 0.4
- Viscera weight = 0.2
- Shell weight = 0.3

**Predicted output:**
Predicted number of rings: 12
Estimated age: 13.5 years

## 🚀 Future Improvements

- Implement **hyperparameter tuning** using `GridSearchCV` or `RandomizedSearchCV`.
- Experiment with advanced models like **XGBoost** or **LightGBM**.
- Add **feature importance visualization** for ensemble models.
- Create a simple **web app** (Streamlit/Flask) for interactive predictions.

---

📍 Usage (Google Colab)
Open the notebook in Google Colab.
Upload the dataset (abalone.csv).
Run all cells to:
Perform EDA
Train models
Evaluate performance
Make predictions

🧑‍💻 Author
Aatir Ali
Machine Learning Enthusiast | Data Science Learner

📫 Feel free to reach out for collaboration or feedback!

## 🧾 Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn



