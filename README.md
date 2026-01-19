# 💎 Diamond Price Prediction using KNN from Scratch

## 📌 Project Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch to predict the **price of diamonds** using the famous `diamonds.csv` dataset.  

The objective is to:

- Preprocess mixed-type data (numerical + categorical)
- Implement KNN regression **without using sklearn’s KNN**
- Evaluate the scratch model
- Compare it with sklearn’s optimized KNN implementation

This project demonstrates both **conceptual understanding** and **practical implementation** of KNN.

---

## 📂 Dataset Information

- **Dataset Name:** diamonds.csv  
- **Total Instances:** ~54,000  
- **Total Features:** 10  
- **Target Variable:** `price` (7th column)

### Features Description

| Feature   | Description |
|-----------|-------------|
| carat     | Weight of the diamond |
| cut       | Quality of the cut (Fair, Good, Very Good, Premium, Ideal) |
| color     | Diamond color from J (worst) to D (best) |
| clarity   | Clarity grade (I1 to IF) |
| x         | Length in mm |
| y         | Width in mm |
| z         | Depth in mm |
| depth     | Total depth percentage |
| table     | Width of top of diamond |
| price     | Price in US dollars (Target) |

---

## 🛠️ Technologies & Libraries Used

- Python 3.x  
- Jupyter Notebook  

### Python Libraries

- `pandas` – Data loading and manipulation  
- `numpy` – Numerical computations  
- `matplotlib` – Visualization  
- `scikit-learn` –  
  - Train-test split  
  - StandardScaler  
  - OneHotEncoder  
  - Model evaluation metrics  
  - Sklearn KNN for comparison  

---

## 🧪 Steps Performed

### Step 1: Load the Dataset
- Loaded `diamonds.csv` using pandas.

### Step 2: Identify Input and Output Variables
- Input Features: All columns except `price`  
- Output Variable: `price`

### Step 3: Train-Test Split
- Split ratio: **75% Train : 25% Test**
- Used `train_test_split` from sklearn.

### Step 4: Data Preprocessing on Training Data

- **Categorical Encoding**
  - Encoded `cut`, `color`, and `clarity` using `OneHotEncoder`.

- **Numerical Rescaling**
  - Applied `StandardScaler` to scale all features.

### Step 5: Data Preprocessing on Test Data
- Applied the same encoding and scaling transformations on test data.

### Step 6: KNN Implementation from Scratch

- Implemented:
  - Euclidean distance function  
  - KNN prediction logic manually  
- Used a **vectorized NumPy version** for faster execution.

Due to the large dataset size, a **subset of training data** was used for scratch KNN to reduce computation time.

### Step 7: Model Evaluation

Evaluated scratch KNN using:

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- R² Score  

### Step 8: Sklearn KNN and Comparison

- Trained `KNeighborsRegressor` from sklearn.
- Compared performance with scratch KNN.

---

## 📊 Evaluation Metrics Used

- **MAE (Mean Absolute Error)**  
- **MSE (Mean Squared Error)**  
- **R² Score**

These metrics help measure prediction accuracy and goodness of fit.

---

## ⏱️ Performance Note

KNN has **O(n²)** time complexity for naive implementation.

Since the dataset has ~54,000 samples:

- Scratch KNN becomes computationally expensive.
- Therefore, a reduced subset of training data was used for scratch implementation.
- Full dataset was used with sklearn’s optimized KNN.

This approach balances **correctness** and **practical feasibility**.

---

## 📈 Results Summary

| Model        | MAE   | MSE   | R² Score |
|--------------|-------|-------|----------|
| Scratch KNN  | (varies) | (varies) | (varies) |
| Sklearn KNN  | Better | Lower | Higher |

Sklearn KNN performs better due to internal optimizations and efficient data structures.

---

## ✅ Key Learnings

- Understood KNN working principle deeply  
- Learned handling mixed data (categorical + numerical)  
- Understood time complexity limitations of brute-force KNN  
- Learned the importance of optimized libraries for large datasets  

---

## 🚀 How to Run the Project

1. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
