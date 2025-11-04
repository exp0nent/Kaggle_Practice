# 🐚 Abalone Age Prediction (Regression Model)

This project predicts the **age of abalone** based on its physical measurements using **machine learning regression models**.  
The dataset used is the classic **Abalone Dataset** from the **UCI Machine Learning Repository**.

## 📘 Overview

The main objective of this project is to estimate the **number of rings** (a proxy for age) of abalone shells using measurable physical attributes such as length, diameter, weight, and height.  
Abalone age is approximately calculated as:

> **Age = Rings + 1.5**

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

## 🧹 Data Preprocessing

1. **Encoding:**  
   The categorical column `Sex` was encoded numerically as:  
   - `M → 0`, `F → 1`, `I → 2`

2. **Scaling:**  
   All numerical features were standardized using `StandardScaler` to normalize feature ranges.

3. **Train-Test Split:**  
   The dataset was divided into:
   - **Training set:** 80%  
   - **Testing set:** 20%

## 🔎 Exploratory Data Analysis (EDA)

Visualizations were performed using **Seaborn** and **Matplotlib** to understand relationships between variables.

**Key observations:**
- Physical measurements such as **length**, **diameter**, and **weights** are highly correlated.
- The number of **Rings** (hence age) is roughly centered around 8–12.
- **Infant abalones (I)** are typically smaller and lighter compared to males and females.

Plots generated:
- Distribution of `Sex` and `Rings`
- Correlation heatmap
- Scatter plots:
  - `Length` vs `Rings`
  - `Length` vs `Rings` by `Sex`

## 🧠 Model Training and Evaluation

Eight different regression algorithms were trained and compared to predict the number of rings.

| Model | MSE | R² |
|--------|------|------|
| Linear Regression | 4.95 | 0.54 |
| Ridge Regression | 4.95 | 0.54 |
| Lasso Regression | 7.69 | 0.29 |
| **Decision Tree Regressor** | **8.94** | **0.17** |
| Random Forest Regressor | 4.94 | 0.54 |
| Support Vector Regressor (SVR) | 4.88 | 0.55 |
| Gradient Boosting Regressor | 5.10 | 0.53 |
| K-Neighbors Regressor | 5.24 | 0.52 |

## 🎯 Chosen Model: Decision Tree Regressor
Although the **Decision Tree Regressor** did not achieve the lowest error, it was chosen as the final model because of its:
- **Simplicity** and interpretability.
- Ability to handle **nonlinear relationships**.
- **Ease of visualization** (clear decision paths).

### Model Performance:
- **Mean Squared Error (MSE):** 8.94  
- **R² Score:** 0.17

The model was trained on scaled features and then used to predict the number of rings for new abalone samples.

## 🧩 Prediction Function

A custom function `prediction_rings()` was created to predict the **number of rings** given user-inputted features.

**Example Input:**
- Sex = 0 (Male)  
- Length = 0.6  
- Diameter = 0.45  
- Height = 0.15  
- Whole weight = 1.0  
- Shucked weight = 0.4  
- Viscera weight = 0.2  
- Shell weight = 0.3  

**Predicted Output:**
Predicted number of rings: 12
Estimated age: 13.5 years

## 📊 Model Interpretation

- Abalones with **higher shell and whole weights** tend to have **more rings** (older).  
- The **Sex** of the abalone influences growth characteristics, but its impact on age is moderate.
- The **Decision Tree** model can be visualized to interpret feature splits and thresholds.

## 🚀 Future Improvements

- Implement **ensemble methods** (Random Forest, Gradient Boosting) for higher accuracy.
- Use **cross-validation** and **hyperparameter tuning** to improve model generalization.
- Develop a **Streamlit or Flask web app** for real-time predictions.
- Add **feature importance visualization** for better explainability.

📍 Usage (Google Colab)

Open the notebook in Google Colab.
Upload the dataset file: abalone.csv.
Run all cells to:
Explore the dataset (EDA)
Train and evaluate models
Predict abalone age using the trained model

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
